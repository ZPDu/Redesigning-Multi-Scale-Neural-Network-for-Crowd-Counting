import torch
import torch.nn as nn
from torchvision import models

class HMoDE(nn.Module):
    def __init__(self, pretrained=False):
        super(HMoDE, self).__init__()

        # The decoder
        self.de_pred3 = nn.Sequential(
            Conv2d(512, 1024, 3, same_padding=True, NL='relu'),
            Conv2d(1024, 512, 3, same_padding=True, NL='relu'),
        )

        self.de_pred2 = nn.Sequential(
            Conv2d(512 + 512, 512, 3, same_padding=True, NL='relu'),
            Conv2d(512, 256, 3, same_padding=True, NL='relu'),
            
        )

        self.de_pred1 = nn.Sequential(
            Conv2d(256 + 256, 256, 3, same_padding=True, NL='relu'),
            Conv2d(256, 128, 3, same_padding=True, NL='relu'),
            
        )

        # density head definition
        self.head3 = nn.Sequential(
            Conv2d(512, 64, 1, same_padding=True, NL='relu'),
            Conv2d(64, 1, 1, same_padding=True, NL='relu')
        )
        self.head2 = nn.Sequential(
            Conv2d(256, 64, 1, same_padding=True, NL='relu'),
            Conv2d(64, 1, 1, same_padding=True, NL='relu')
        )
        self.head1 = nn.Sequential(
            Conv2d(128, 64, 1, same_padding=True, NL='relu'),
            Conv2d(64, 1, 1, same_padding=True, NL='relu')
        )

        # The gating networks in the two levels and the attention module
        self.gating1 = GatingBlock(512, 6)
        self.gating2 = GatingBlock2(512, 3)
        self.mask = MaskBlock(512)

        self._weight_init_()
        
        # Using VGG16 as backbone network
        vgg = models.vgg16_bn(pretrained=pretrained)
        features = list(vgg.features.children())
        # Partition VGG16 into five encoder blocks
        self.features1 = nn.Sequential(*features[0:6])
        self.features2 = nn.Sequential(*features[6:13])
        self.features3 = nn.Sequential(*features[13:23])
        self.features4 = nn.Sequential(*features[23:33])
        self.features5 = nn.Sequential(*features[33:43])

    def _weight_init_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_):
        size = input_.size()
        # encoding input images
        x1 = self.features1(input_)
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)
        # deconding with skip connections
        x = self.de_pred3(x5)
        x3_out = x
        x = nn.functional.interpolate(x, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x4, x], 1)
        x = self.de_pred2(x)
        x2_out = x
        x = nn.functional.interpolate(x, size=x3.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x3, x], 1)
        x = self.de_pred1(x)
        x1_out = x

        # density estimation using multi-scale features
        density_map3 = self.head3(x3_out)
        density_map2 = self.head2(x2_out)
        density_map1 = self.head1(x1_out)

        # upsample the estimated density maps to the same resolution
        density_map3 = nn.functional.interpolate(density_map3, size=density_map1.shape[2:], mode='nearest')
        density_map2 = nn.functional.interpolate(density_map2, size=density_map1.shape[2:], mode='nearest')
        del x2, x3, x3_out, x2_out, x1_out

        # generate attention map for the second gating net
        amp = self.mask(x5)

        # combining multi-scale density maps in the first level
        wmaps_ = self.gating1(x5, density_map1.shape)
        wmaps1 = wmaps_[0]
        wmaps2 = wmaps_[1]
        wmaps3 = wmaps_[2]
        # calculating expert importance loss
        imp_loss = cv_squared(wmaps1.sum(0)) + cv_squared(wmaps2.sum(0)) + cv_squared(wmaps3.sum(0))

        density1 = torch.cat([density_map1, density_map2], 1)
        density1 = density1 * wmaps1
        density1 = torch.sum(density1, 1, keepdim=True)
        density2 = torch.cat([density_map1, density_map3], 1)
        density2 = density2 * wmaps2
        density2 = torch.sum(density2, 1, keepdim=True)
        density3 = torch.cat([density_map2, density_map3], 1)
        density3 = density3 * wmaps3
        density3 = torch.sum(density3, 1, keepdim=True)

        # the second level
        x5 = amp * x5
        wmaps = self.gating2(x5, density_map1.shape)
        density = torch.cat([density1, density2, density3], 1)
        density = density * wmaps
        density = torch.sum(density, 1, keepdim=True)

        density = nn.functional.interpolate(density, size[2:], mode='nearest')

        return [density_map1, density_map2, density_map3, density1, density2, density3, density], amp, imp_loss

def cv_squared(x):
    eps = 1e-10
    x = x.sum(2).sum(1)
    return x.float().var() / (x.float().mean()**2 + eps)

class MaskBlock(nn.Module):
    # Attention module
    def __init__(self, num_features):
        super(MaskBlock, self).__init__()
        self.conv1 = Conv2d(num_features, 256, 3, same_padding=True)
        self.conv2 = Conv2d(256, 128, 3, same_padding=True)
        self.output = Conv2d(128, 1, 1, NL=None)
        self.sgm = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        amp = self.output(out)
        amp = self.sgm(amp)

        return amp


class GatingBlock(nn.Module):
    # Gating network in the 1st level
    def __init__(self, num_features, output_num):
        super(GatingBlock, self).__init__()
        self.output_num = output_num
        self.conv1 = Conv2d(num_features, 256, 3, same_padding=True)
        self.conv2 = Conv2d(256, output_num, 1, same_padding=True)

    def forward(self, x, size):
        out = self.conv1(x)
        out = self.conv2(out)

        wmap1 = out[:,0:2]
        wmap2 = out[:,2:4]
        wmap3 = out[:,4:6]
        wmap1 = nn.functional.softmax(wmap1, dim=1)
        wmap2 = nn.functional.softmax(wmap2, dim=1)
        wmap3 = nn.functional.softmax(wmap3, dim=1)

        wmap1 = nn.functional.interpolate(wmap1, size=size[2:], mode='nearest')
        wmap2 = nn.functional.interpolate(wmap2, size=size[2:], mode='nearest')
        wmap3 = nn.functional.interpolate(wmap3, size=size[2:], mode='nearest')

        wmaps = [wmap1, wmap2, wmap3]

        return wmaps


class GatingBlock2(nn.Module):
    # Gating network in the 2nd level
    def __init__(self, num_features, output_num):
        super(GatingBlock2, self).__init__()
        self.output_num = output_num
        self.conv1 = Conv2d(num_features, 1024, 3, same_padding=True)
        self.conv2 = Conv2d(1024, 2048, 3, same_padding=True)
        self.conv3 = Conv2d(2048, 1024, 3, same_padding=True)
        self.conv4 = Conv2d(1024, 512, 3, same_padding=True)
        self.conv5 = Conv2d(512, 256, 3, same_padding=True)
        self.conv6 = Conv2d(256, output_num, 1)

    def forward(self, x, size):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)

        wmaps = nn.functional.softmax(out,dim=1)
        wmaps = nn.functional.interpolate(wmaps, size=size[2:], mode='nearest')
        
        return wmaps

def cv_squared(x):
    # expert importance loss
    eps = 1e-10
    x = x.sum(2).sum(1)
    return x.float().var() / (x.float().mean()**2 + eps)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []
        if dilation==1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else nn.Identity()
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x