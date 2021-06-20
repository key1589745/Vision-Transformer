'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch import ViT


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CNN_BackBone(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(CNN_BackBone, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        
        layers = []
        for i in range(len(num_blocks)):
            layers.append(self._make_layer(block, self.in_planes*(2**i), num_blocks[i], stride=1 if i==0 else 2))
        self.layers = nn.ModuleList(layers)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        return out

class hybrid(nn.Module):
    def __init__(self, n_blocks=[2,2,1], patch_size=1, depth=6, head=16):
        super(hybrid, self).__init__()
        self.conv = CNN_BackBone(BasicBlock, n_blocks)
        fsize = 32//2**(len(n_blocks)-1)
        channels = 64*2**len(n_blocks)
        self.transformer = ViT(       
            image_size = fsize,
            patch_size = patch_size,
            channels = 512,
            num_classes = 10,
            dim = 1024,
            depth = depth,
            heads = head,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1)
    def forward(self, x):
        x = self.conv(x)
        x = self.transformer(x)
        return x