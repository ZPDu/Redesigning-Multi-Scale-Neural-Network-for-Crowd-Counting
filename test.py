import torch
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import misc.transforms as own_transforms
from datasets.SHHA.setting import cfg_data
from datasets.SHHA.SHHA import SHHA
import os
import numpy as np
from HMoDE import HMoDE

mean_std = cfg_data.MEAN_STD
val_main_transform = None
log_para = 100.
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])
gt_transform = standard_transforms.Compose([
    own_transforms.LabelNormalize(log_para)
])
restore_transform = standard_transforms.Compose([
    own_transforms.DeNormalize(*mean_std),
    standard_transforms.ToPILImage()
])

test_set = SHHA(cfg_data.DATA_PATH+'/test_data', main_transform=val_main_transform, img_transform=img_transform, gt_transform=gt_transform, data_augment=1)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False)

if __name__ == '__main__':
    net = HMoDE(False)
    net.load_state_dict(torch.load(cfg_data.RESUME_MODEL))
    net.cuda()
    net.eval()
    print('=' * 50)
    val_loss = []
    mae = 0.0
    mse = 0.0
    for vi, data in enumerate(test_loader, 0):
        img, gt_map = data
        # pdb.set_trace()
        with torch.no_grad():
            img = img.cuda()
            gt_map = gt_map.cuda()

            pred_map = net(img)[0][-1]

            pred_map = pred_map.data.cpu().numpy()
            gt_map = gt_map.data.cpu().numpy()

            gt_count = np.sum(gt_map) / log_para
            pred_cnt = np.sum(pred_map) / log_para

            mae += abs(gt_count - pred_cnt)
            mse += ((gt_count - pred_cnt) * (gt_count - pred_cnt))

    mae = mae / test_set.get_num_samples()
    mse = np.sqrt(mse / test_set.get_num_samples())
    print('mae:', mae, 'mse:', mse)
