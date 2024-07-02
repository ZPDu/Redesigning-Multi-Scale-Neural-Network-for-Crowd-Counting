import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

def relloss(pred, gt):
  gt = gt.unsqueeze(1)
  loss_map = F.mse_loss(pred, gt, reduction='none')
  loss_map1 = convert_to_small_scale(loss_map, patch_size=16)

  b, c, h, w = loss_map1.shape
  loss_map_tmp = loss_map1.detach()
  loss_map_tmp = loss_map_tmp.view(b, -1)

  loss_hard, hard_region_list = torch.topk(loss_map_tmp, k=9, dim=-1, largest=True, sorted=True)

  relativeloss = relativeLoss(pred, gt, hard_region_list)

  loss = torch.sum(loss_hard) + relativeloss

  return loss


def relativeLoss(pred, gt, hard_region_list):

  pred = convert_to_small_scale(pred, patch_size=16).view(pred.shape[0], -1)
  gt = convert_to_small_scale(gt, patch_size=16).view(pred.shape[0], -1)

  sorted_pred = torch.zeros(hard_region_list.shape).cuda()
  sorted_gt = torch.zeros(hard_region_list.shape).cuda()
  for idx in range(sorted_pred.shape[0]):
      sorted_pred[idx] = pred[idx, hard_region_list[idx]]
      sorted_gt[idx] = gt[idx, hard_region_list[idx]]
      _, sort_list = torch.sort(sorted_gt[idx], descending=True)
      sorted_pred[idx] = sorted_pred[idx, sort_list]

  loss_rel = 0.
  for start_idx in range(3):
      pred_tmp = sorted_pred[:, start_idx::3]

      d1 = pred_tmp[:, 0] - pred_tmp[:, 1]
      d2 = pred_tmp[:, 0] - pred_tmp[:, 2]

      loss_rel += torch.clamp((-1.0) * d1, 0.0)
      loss_rel += torch.clamp((-1.0) * d2, 0.0)
      loss_rel += torch.clamp((d1-d2), 0.0)

  loss_rel = torch.sum(loss_rel)
  return loss_rel

 
def convert_to_small_scale(den, patch_size=4):
    pool_filter = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size).cuda()
    target = pool_filter(den)

    return target * (patch_size**2)
