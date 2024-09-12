# RankMatch
import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import torch.nn.functional as F

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus_modify
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed

from itertools import permutations
import numpy as np


parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def prob2rank(prob, prob_s, k=4):
    """
    input: prob(probability) [b, h, w, n]
    return: rank [b, h, w, k!]
    To save the computing resources, use top-k ranther than n
    """
    full_permutation = [c for c in permutations(range(k))]
    full_permutation = torch.from_numpy(np.stack(full_permutation)) # [k!, k]

    _, prob_topk_index = prob.topk(k, dim=-1) # [b, h, w, k]
    A = prob_topk_index[:, :, :, full_permutation] # [b, h, w, k!, k]
    B = prob.unsqueeze(3).expand(-1, -1, -1, full_permutation.shape[0], -1) # [b, h, w, k!, n]
    B_s = prob_s.unsqueeze(3).expand(-1, -1, -1, full_permutation.shape[0], -1)
    C = torch.gather(input=B, dim=-1, index=A) # [b, h, w, k!, k]
    C_s = torch.gather(input=B_s, dim=-1, index=A)

    rank = C[:, :, :, :, 0] / (C[:, :, :, :, 0:].sum(dim=-1) + 1e-10) # [b, h, w, k!]
    rank_s = C_s[:, :, :, :, 0] / (C_s[:, :, :, :, 0:].sum(dim=-1) + 1e-10)

    for i in range(1, k):
        rank *= C[:, :, :, :, i] / (C[:, :, :, :, i:].sum(dim=-1) + 1e-10)
        rank_s *= C_s[:, :, :, :, i] / (C_s[:, :, :, :, i:].sum(dim=-1) + 1e-10)

    return rank, rank_s


def orthogonal_landmarks(q, q_s, num_landmarks=64, subsample_fraction=1.0):
    """
    Construct set of landmarks by recursively selecting new landmarks 
    that are maximally orthogonal to the existing set.
    Returns near orthogonal landmarks with shape (B, M, D).
    """
    B, D, H, W = q.shape
    N = H * W
    q = q.permute(0, 2, 3, 1).reshape(B, -1, D)
    q_s = q_s.permute(0, 2, 3, 1).reshape(B, -1, D)

    # ignore_mask = F.interpolate(ignore_mask.unsqueeze(1), size=(H, W), mode="nearest").squeeze(1)
    # nonignore_mask = nonignore_mask.reshape(B, -1)

    if subsample_fraction < 1.0:
        # Need at least M/2 samples of queries and keys
        num_samples = max(int(subsample_fraction * q.size(-2)), num_landmarks)
        q_unnormalised = q[:, torch.randint(q.size(-2), (num_samples,), device=q.device), :] # (B, N, D)
    else:
        # (B, N, D)
        q_unnormalised = q

    # may need to change default eps to eps=1e-8 for mixed precision compatibility
    qk = F.normalize(q_unnormalised, p=2, dim=-1)
    # B, N, D = qk.shape

    selected_mask = torch.zeros((B, N, 1), device=qk.device)
    landmark_mask = torch.ones((B, 1, 1), dtype=selected_mask.dtype, device=qk.device)

    # Get initial random landmark
    random_idx = torch.randint(qk.size(-2), (B, 1, 1), device=qk.device)
    # random_idx = torch.empty((B, 1, 1), dtype=torch.long, device=qk.device)
    # for i in range(B):
    #     nonignore_indices = torch.nonzero(nonignore_mask[i])
    #     selected_index = nonignore_indices[torch.randint(0, nonignore_indices.size(0), (1,))]
    #     random_idx[i, 0, 0] = selected_index.item()

    selected_landmark = qk[torch.arange(qk.size(0)), random_idx.view(-1), :].view(B, D)
    selected_mask.scatter_(-2, random_idx, landmark_mask)

    # Selected landmarks
    selected_landmarks = torch.empty((B, num_landmarks, D), device=qk.device, dtype=qk.dtype)
    selected_landmarks[:, 0, :] = selected_landmark

    # Store computed cosine similarities
    cos_sims = torch.empty((B, N, num_landmarks), device=qk.device, dtype=qk.dtype)

    for M in range(1, num_landmarks):
        # Calculate absolute cosine similarity between selected and unselected landmarks
        # (B, N, D) * (B, D) -> (B, N)
        cos_sim = torch.einsum('b n d, b d -> b n', qk, selected_landmark).abs()
        # # set cosine similarity for ignore mask to > 1
        # cos_sim.view(-1)[nonignore_mask.flatten() == False] = 10
        cos_sims[:, :, M - 1] = cos_sim
        # (B, N, M) cosine similarities of current set of landmarks wrt all queries and keys
        cos_sim_set = cos_sims[:, :, :M]

        # Get orthogonal landmark: landmark with smallest absolute cosine similarity:
        # set cosine similarity for already selected landmarks to > 1
        cos_sim_set.view(-1, M)[selected_mask.flatten().bool(), :] = 10
        # (B,) - want max for non
        selected_landmark_idx = cos_sim_set.amax(-1).argmin(-1)
        selected_landmark = qk[torch.arange(qk.size(0)), selected_landmark_idx, :].view(B, D)

        # Add most orthogonal landmark to selected landmarks: 
        selected_landmarks[:, M, :] = selected_landmark

        # Removed selected indices from non-selected mask: 
        selected_mask.scatter_(-2, selected_landmark_idx.unsqueeze(-1).unsqueeze(-1), landmark_mask)

    landmarks = torch.masked_select(q_unnormalised, selected_mask.bool()).reshape(B, -1, D) # (B, M, D)
    landmarks_s = torch.masked_select(q_s, selected_mask.bool()).reshape(B, -1, D)

    return landmarks, landmarks_s # (B, M, D)


def corr_loss(feat_w, feat_s, local_rank):

    criterion_c = torch.nn.KLDivLoss(reduction='mean').cuda(local_rank) # pixel-reference correlation criterion

    num_landmarks = 64

    refers_w, refers_s = orthogonal_landmarks(feat_w, feat_s, num_landmarks)

    p2r_w = torch.einsum('b c h w, b n c -> b h w n', feat_w, refers_w).softmax(dim=-1)
    p2r_s = torch.einsum('b c h w, b n c -> b h w n', feat_s, refers_s).softmax(dim=-1)

    p2r_w_rank, p2r_s_rank = prob2rank(p2r_w, p2r_s)

    loss = criterion_c((p2r_s_rank + 1e-10).log(), p2r_w_rank)

    return loss


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus_modify(cfg)
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    # criterion_c = nn.KLDivLoss(reduction='none').cuda(local_rank) # pixel-reference correlation criterion

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_c = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):
            
            torch.cuda.empty_cache()
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            iters = epoch * len(trainloader_u) + i

            with torch.no_grad():
                model.eval()

                pred_u_w_mix, feat_u_w_mix = model(img_u_w_mix)
                pred_u_w_mix, feat_u_w_mix = pred_u_w_mix.detach(), feat_u_w_mix.detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)
                _, H, W =conf_u_w_mix.shape
                _, _, h, w = feat_u_w_mix.shape

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            preds, preds_fp, feats = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            _, feat_u_w = feats.split([num_lb, num_ulb])
            
            pred_u_w_fp = preds_fp[num_lb:]

            preds_u_s, feats_u_s = model(torch.cat((img_u_s1, img_u_s2)))
            pred_u_s1, pred_u_s2 = preds_u_s.chunk(2)
            # feats_u_s = F.interpolate(feats_u_s, size=(H, W), mode="bilinear", align_corners=True)
            feat_u_s1, feat_u_s2 = feats_u_s.chunk(2)

            pred_u_w = pred_u_w.detach()
            feat_u_w = feat_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            feat_u_w = F.interpolate(feat_u_w, size=(H, W), mode="bilinear", align_corners=True)
            feat_u_w_mix = F.interpolate(feat_u_w_mix, size=(H, W), mode="bilinear", align_corners=True)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1, feat_u_w_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone(), feat_u_w.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2, feat_u_w_cutmixed2 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone(), feat_u_w.clone()

            # del feat_u_w
            # torch.cuda.empty_cache()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]
            # cutmix_box1 = F.interpolate(cutmix_box1.unsqueeze(1), size=(h, w), mode="nearest").repeat(1,256,1,1)
            # cutmix_box1 = cutmix_box1.unsqueeze(1).repeat(1,256,1,1)
            # feat_u_w_cutmixed1[cutmix_box1 == 1] = feat_u_w_mix[cutmix_box1 == 1]
            # feat_u_w_cutmixed1[cutmix_box1.unsqueeze(1).expand(feat_u_w.shape) == 1] = \
            #     feat_u_w_mix[cutmix_box1.unsqueeze(1).expand(feat_u_w.shape) == 1]
            cutmix_box1 = cutmix_box1.unsqueeze(1).expand(feat_u_w.shape)
            # feat_u_w_cutmixed1 = feat_u_w_cutmixed1 * (1 - cutmix_box1) + feat_u_w_mix * cutmix_box1
            feat_u_w_cutmixed1 = torch.where(cutmix_box1 == 1, feat_u_w_mix, feat_u_w_cutmixed1)
            feat_u_w_cutmixed1 = F.interpolate(feat_u_w_cutmixed1, size=(h, w), mode="bilinear", align_corners=True)

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]
            # cutmix_box2 = F.interpolate(cutmix_box2.unsqueeze(1), size=(h, w), mode="nearest").repeat(1,256,1,1)
            # cutmix_box2 = cutmix_box2.unsqueeze(1).repeat(1,256,1,1)
            # feat_u_w_cutmixed2[cutmix_box2 == 1] = feat_u_w_mix[cutmix_box2 == 1]
            # feat_u_w_cutmixed2[cutmix_box2.unsqueeze(1).expand(feat_u_w.shape) == 1] = \
            #     feat_u_w_mix[cutmix_box2.unsqueeze(1).expand(feat_u_w.shape) == 1]
            cutmix_box2 = cutmix_box2.unsqueeze(1).expand(feat_u_w.shape)
            # feat_u_w_cutmixed2 = feat_u_w_cutmixed2 * (1 - cutmix_box2) + feat_u_w_mix * cutmix_box2
            feat_u_w_cutmixed2 = torch.where(cutmix_box2 == 1, feat_u_w_mix, feat_u_w_cutmixed2)
            feat_u_w_cutmixed2 = F.interpolate(feat_u_w_cutmixed2, size=(h, w), mode="bilinear", align_corners=True)

            # del feat_u_w, feat_u_w_mix
            # torch.cuda.empty_cache()

            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255))
            loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()

            # pixel-reference correlation consistency
            # conf_u_w_cutmixed1 = F.interpolate(conf_u_w_cutmixed1.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=True).squeeze(1)
            # ignore_mask_cutmixed1 = F.interpolate(ignore_mask_cutmixed1.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1)
            # conf_u_w_cutmixed2 = F.interpolate(conf_u_w_cutmixed2.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=True).squeeze(1)
            # ignore_mask_cutmixed2 = F.interpolate(ignore_mask_cutmixed2.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1)

            # if iters > 10:
            #     loss_c_s1 = corr_loss(feat_u_w_cutmixed1, feat_u_s1, conf_u_w_cutmixed1, ignore_mask_cutmixed1, local_rank, cfg['conf_thresh'])
            #     loss_c_s2 = corr_loss(feat_u_w_cutmixed2, feat_u_s2, conf_u_w_cutmixed2, ignore_mask_cutmixed2, local_rank, cfg['conf_thresh'])
            # else:
            #      loss_c_s1 = loss_c_s2 = torch.tensor(0)

            loss_c_s1 = corr_loss(feat_u_w_cutmixed1, feat_u_s1, local_rank)
            loss_c_s2 = corr_loss(feat_u_w_cutmixed2, feat_u_s2, local_rank)

            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0 + 0.1 * (loss_c_s1 + loss_c_s2) / 2.0

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_c.update((loss_c_s1.item() + loss_c_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())
            
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_c', (loss_c_s1.item() + loss_c_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss c: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg, total_loss_c.avg, 
                                            total_loss_w_fp.avg, total_mask_ratio.avg))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
