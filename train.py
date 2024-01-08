import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
from HMoDE import *
# from func import relloss
import os
from datasets.SHHA.loading_data import loading_data
from datasets.SHHA.setting import cfg_data
from misc.timer import Timer

transform = standard_transforms.Compose([
    standard_transforms.ToTensor()])

exp_name = cfg_data.EXP_NAME
log_txt = cfg_data.EXP_PATH + '/' + exp_name + '.txt'

if not os.path.exists(cfg_data.EXP_PATH):
    os.mkdir(cfg_data.EXP_PATH)

pil_to_tensor = standard_transforms.ToTensor()

train_record = {'best_mae': 1e20, 'mse': 1e20, 'corr_loss': 0, 'corr_epoch': -1, 'best_model_name': ''}

_t = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}

rand_seed = cfg_data.SEED
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

train_set, train_loader, val_set, val_loader, restore_transform = loading_data()

def main():
    load = False
    begin = 0

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg_data.GPU_ID
    torch.backends.cudnn.benchmark = True

    net = HMoDE(True)
    net = nn.DataParallel(net)
    net = net.cuda()

    net.train()

    optimizer = optim.Adam(net.parameters(), lr=2e-5)
    stepLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    i_tb = 0
    if load:
        checkpoint = torch.load(os.path.join(cfg_data.EXP_PATH, 'latestmodel.pth'))
        net.load_state_dict(checkpoint['model'])
        begin = checkpoint['epoch'] + 1
        i_tb = checkpoint['i_tb']
        optimizer.load_state_dict(checkpoint['optimizer'])
        train_record['best_mae'] = checkpoint['record']['best_mae']
        train_record['mse'] = checkpoint['record']['mse']
        train_record['corr_epoch'] = checkpoint['record']['corr_epoch']
        train_record['corr_loss'] = checkpoint['record']['corr_loss']

    for epoch in range(begin, cfg_data.MAX_EPOCH):

        _t['train time'].tic()
        i_tb, model_path = train(train_loader, net, optimizer, epoch, i_tb)
        _t['train time'].toc(average=False)
        print('train time of one epoch: {:.2f}s'.format(_t['train time'].diff))
        if epoch + 1 >= 100:
            _t['val time'].tic()
            validate(val_loader, val_set, epoch)
            _t['val time'].toc(average=False)
            print('val time of one epoch: {:.2f}s'.format(_t['val time'].diff))

        if (epoch+1) == 100:
            stepLR.step()

def train(train_loader, net, optimizer, epoch, i_tb):
    net.train()
    mseloss = nn.MSELoss(reduction='sum').cuda()

    for i, data in enumerate(train_loader, 0):
        _t['iter time'].tic()
        img, gt_map = data
        img = Variable(img).cuda()
        gt_map = Variable(gt_map).cuda()
        amp_gt = (gt_map>(1e-5*cfg_data.LOG_PARA)).float().unsqueeze(1)

        # predicted maps, predicted attention map and expert importance loss
        pred_maps, amp, imp_loss = net(img)

        optimizer.zero_grad()

        loss = 0.
        rel_loss = 0.
        for i in range(len(pred_maps)):
            # density loss
            loss += (2**(int(i / 3))) * mseloss(pred_maps[i], gt_map)
            # relative loss
            # rel_loss += (2**(int(i / 3))) * relloss(pred_maps[i], gt_map)
        # attention loss
        amp = nn.functional.interpolate(amp, amp_gt.shape[2:], mode='nearest')
        cross_entropy_loss = (amp_gt * torch.log(amp+1e-10) + (1 - amp_gt) * torch.log(1 - amp+1e-10)) * -1

        # total objectives, loss weights can be adjusted
        loss = loss + rel_loss + torch.sum(imp_loss) + torch.sum(cross_entropy_loss)

        loss = loss / pred_maps[0].shape[0]
        loss.backward()
        optimizer.step()

        if (i + 1) % cfg_data.PRINT_FREQ == 0:
            loss = mseloss(pred_maps[0].squeeze(), gt_map)
            i_tb = i_tb + 1
            _t['iter time'].toc(average=False)
            print('[ep %d][it %d][loss %.8f][LR %.8f][%.2fs]' % \
                  (epoch + 1, i + 1, torch.sum(loss).item(), optimizer.state_dict()['param_groups'][0]['lr'], _t['iter time'].diff))
            print('        [gt: %.1f pred: %.6f]' % (
                gt_map[0].sum() / cfg_data.LOG_PARA, pred_maps[0][0].sum().item() / cfg_data.LOG_PARA))

    # save model
    to_saved_weight = []

    if len(cfg_data.GPU_ID) > 1:
        to_saved_weight = net.module.state_dict()
    else:
        to_saved_weight = net.state_dict()

    state = {'epoch': epoch, 'i_tb': i_tb, 'model': to_saved_weight, 'optimizer': optimizer.state_dict(),
             'record': train_record}
    model_path = os.path.join(cfg_data.EXP_PATH, 'latestmodel.pth')
    torch.save(state, model_path)

    return i_tb, model_path


def validate(val_loader, val_set, epoch):
    torch.cuda.empty_cache()
    mseloss = nn.MSELoss(reduction='sum').cuda()

    net = HMoDE(False)
    net.load_state_dict(torch.load(os.path.join(cfg_data.EXP_PATH, 'latestmodel.pth'))['model'])
    net.cuda()
    net.eval()
    print('=' * 50)
    val_loss = []
    mae = 0.0
    mse = 0.0

    for vi, data in enumerate(val_loader, 0):
        img, gt_map = data
        # pdb.set_trace()
        with torch.no_grad():
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()

            pred_map = net(img)[0]
            loss = mseloss(pred_map, gt_map)
            val_loss.append(loss.item())

            pred_map = pred_map.data.cpu().numpy() / cfg_data.LOG_PARA
            gt_map = gt_map.data.cpu().numpy() / cfg_data.LOG_PARA

            gt_count = np.sum(gt_map)
            pred_cnt = np.sum(pred_map)

            mae += abs(gt_count - pred_cnt)
            mse += ((gt_count - pred_cnt) * (gt_count - pred_cnt))


    mae = mae / val_set.get_num_samples()
    mse = np.sqrt(mse / val_set.get_num_samples())

    loss = np.mean(val_loss)

    if mae < train_record['best_mae']:
        train_record['best_mae'] = mae
        train_record['mse'] = mse
        train_record['corr_epoch'] = epoch + 1
        train_record['corr_loss'] = loss
        to_saved_weight = net.state_dict()
        state = {'model': to_saved_weight}
        model_path = os.path.join(cfg_data.EXP_PATH, 'best_model.pth')
        torch.save(state, model_path)

    print('=' * 50)
    print(exp_name)
    print('    ' + '-' * 20)
    print('    [mae %.1f mse %.1f], [val loss %.8f]' % (mae, mse, loss))
    print('    ' + '-' * 20)
    print('[best] [mae %.1f mse %.1f], [val loss %.8f], [epoch %d]' % (
        train_record['best_mae'], train_record['mse'], train_record['corr_loss'], train_record['corr_epoch']))
    print('=' * 50)


if __name__ == '__main__':
    main()
