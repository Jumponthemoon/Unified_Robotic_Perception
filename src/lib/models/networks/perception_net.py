# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn.functional as F
import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class CoordAttention(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
         identity = x

         n,c,h,w = x.size()
         x_h = self.pool_h(x)
         x_w = self.pool_w(x).permute(0, 1, 3, 2)
         y = torch.cat([x_h, x_w], dim=2)
         y = self.conv1(y)
         y = self.bn1(y)
         y = self.act(y)

         x_h, x_w = torch.split(y, [h, w], dim=2)
         x_w = x_w.permute(0, 1, 3, 2)
         a_h = self.conv_h(x_h).sigmoid()
         a_w = self.conv_w(x_w).sigmoid()
         out = identity * a_w * a_h
         return out 

class EnvClassifier(nn.Module):
    def __init__(self, in_channels, num_class):
        super(EnvClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,2 * in_channels,
                      kernel_size=3,stride=2,bias=False)
        self.conv2 = nn.Conv2d(2 * in_channels,4 * in_channels,
                      kernel_size=3,stride=2,bias=False)
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(256*7*7,num_class)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = x.view(-1,256*7*7)
        x = self.fc(x)
        return x
        
class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding, inplace=True):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=inplace)
        #self.relu = Mish()
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class CSP_layer0(nn.Module):
    def __init__(self, in_c, out_c):
        super(CSP_layer0, self).__init__()

        outc_by2 = int(out_c / 2)

        self.conv1 = CBR(in_c, outc_by2, kernel=3, stride=1, padding=1)
        self.conv2 = CBR(outc_by2, out_c, kernel=3, stride=2, padding=1)

        self.conv3 = CBR(out_c, out_c, kernel=1, stride=1, padding=0)
        self.conv4 = CBR(out_c, outc_by2, kernel=1, stride=1, padding=0)
        self.conv5 = CBR(outc_by2, out_c, kernel=3, stride=1, padding=1)
        self.conv6 = CBR(out_c, out_c, kernel=1, stride=1, padding=0)

        # CSP Layer
        self.csp_dense = CBR(out_c, out_c, kernel=1, stride=1, padding=0)
        self.conv7 = CBR(out_c * 2, out_c, kernel=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x5 = x5 + x3  # Residual block
        x6 = self.conv6(x5)
        xd6 = self.csp_dense(x2)  # CSP
        x6 = torch.cat([x6, xd6], dim=1)
        x7 = self.conv7(x6)
        return x7


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        #self.relu = Mish()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
        #self.spatial_atten = SpatialAttention()
        #self.channel_atten = ChannelAttention(planes * self.expansion)
        #self.se = SE_Block(planes,reduction=16)
        self.ca = CoordAttention(planes * self.expansion, planes * self.expansion)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.ca(out)
        #out = self.channel_atten(out) * out
        #out = self.spatial_atten(out) * out
        #out = self.se(out)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.ca = CoordAttention(in_channels=planes * self.expansion, out_channels=planes * self.expansion)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.ca(out)
        out += residual
        out = self.relu(out)

        return out

class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
        self, in_channels_list, out_channels, conv_block, top_blocks=None
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            inner_top_down = torch.nn.functional.interpolate(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = getattr(self, inner_block)(feature)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))

        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)


class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [torch.nn.functional.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(torch.nn.functional.relu(p6))
        return [p6, p7]

class Channel2Spatial(nn.Module):
    """Channel to spatial is essentially the inverse of Spatial2Channel.
    It's now serve as a way to upsample feature map.
    """

    def __init__(
        self,
        in_c=1,
        in_h=1,
        in_w=1,
        ratio=2
    ):
        super(Channel2Spatial, self).__init__()
        self.ratio = ratio
        self.in_c = in_c
        self.in_h = in_h
        self.in_w = in_w

        assert self.in_c % (self.ratio ** 2) == 0, "in_channel: {0}, ratio:{1}".format(self.in_c, self.ratio)
        self.out_c = self.in_c // (self.ratio ** 2)
        self.out_h = self.in_h * self.ratio
        self.out_w = self.in_w * self.ratio

    def forward(self, x, is_onnx=False):
        if is_onnx:
            b = (
                x.shape[0] if not is_onnx else 1
            )
            out_c = int(self.in_c / (self.ratio**2))
            out_h = int(self.in_h * self.ratio)
            out_w = int(self.in_w * self.ratio)

            x = x.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
            x = x.view(b, self.in_h, self.in_w, out_c, self.ratio, self.ratio)
            x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, out_c, out_h, out_w)
        else:
            b = (
                x.shape[0] if not is_onnx else 1
            )
            in_c = x.shape[1]
            in_h = x.shape[2]
            in_w = x.shape[3]
            out_c = int(in_c / (self.ratio**2))
            out_h = int(in_h * self.ratio)
            out_w = int(in_w * self.ratio)

            x = x.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
            x = x.view(b, in_h, in_w, out_c, self.ratio, self.ratio)
            x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, out_c, out_h, out_w)
        return x

class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv, **kwargs):
        self.layer_id = [1, 2, 3, 4]
        self.fpn_layer_id = [2, 3, 4]
        c2s_in_shapes = [(128,64,64), (256,32,32), (512,16,16)]
       # c2s_in_shapes = [(128, 60, 108), (256, 30, 54), (512, 15, 27)]
        fpn_in_chls = [32, 64, 128]
        fpn_chls = 64
        self.inplanes = 64
        self.deconv_with_bias = False
        self.heads = heads
        opt = kwargs['opt']
        self.opt = opt
        print(self.opt)
        self.is_onnx = kwargs['is_onnx']
        self.branch_to_pred_vis = getattr(self.opt, 'branch_to_pred_kp_vis', 'hps') #branch_to_pred_kp_vis='hps'

        self.pred_kp_vis = True#len(opt.pred_vis_joint_names) > 0
        self.kp_vis_chls = 8# len(opt.pred_vis_joint_names) if not self.is_onnx else 4
        num_env_class = 2
        
        super(PoseResNet, self).__init__()

        self.csp_layer0 = CSP_layer0(12, 64)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # create c2s
        self.c2s_list = [Channel2Spatial(in_c, in_h, in_w, 2) for in_c, in_h, in_w in c2s_in_shapes]

        # create fpn
        self.fpn = FPN(fpn_in_chls, fpn_chls,
                       conv_block=conv_with_kaiming_uniform(use_gn=False, use_relu=False),
                       )
        self.classifier = EnvClassifier(fpn_chls,2)
        self._make_subnet_head(opt, fpn_chls)

       # uncertainty weight
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    def _make_subnet_head(self, opt, fpn_out_chl):
        cbr={'segmap':4,'hm':4,'hm_state':4,'hps_vis':4,'wh':2,'hps':8, 'id':2, 'reg':2, 'hm_hp':4, 'hp_offset':2}
        last_k={'segmap':3,'hm':3,'hm_state':3,'hps_vis':1,'wh':1,'hps':5, 'id':1, 'reg':1, 'hm_hp':1, 'hp_offset':1}
        if 'dep' in self.heads:
            cbr.update({'dep':2})
            last_k.update({'dep':1})
        if 'scene' in self.heads:
            cbr.update({'scene':4})
            last_k.update({'scene':3})
        if 'seg' in self.heads:
            cbr.update({'seg':4})
            last_k.update({'seg':3})
        
        up = []
        #from constants import CONSTANT
        for head in sorted(self.heads):
            num_output = self.heads[head]
            num_buffers=cbr[head]
            last_conv_k=last_k[head]
            subnet = []
            buffer_chl = 64
            if head == 'scene':
                subnet.append(self.classifier)
            else:
                for i in range(num_buffers):
                    if i == 0:
                        in_c = fpn_out_chl
                    else:
                        in_c = buffer_chl
                    subnet.append(CBR(in_c, buffer_chl, kernel=3, stride=1, padding=1))
                if head == 'seg':
                    subnet.append(nn.Conv2d(buffer_chl, buffer_chl, kernel_size=last_conv_k, stride=1, padding=last_conv_k//2))           
                    subnet.append(nn.Upsample(scale_factor=4, mode='bilinear',align_corners=True))
                    subnet.append(nn.Conv2d(buffer_chl, num_output,3,1,1 ))
                else:
                    subnet.append(nn.Conv2d(buffer_chl, num_output, kernel_size=last_conv_k, stride=1, padding=last_conv_k//2))           

            if head == 'segmap':
                subnet = subnet+[nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)
                                ]                               
            out = nn.Sequential(*subnet)
            self.__setattr__(head, out)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
          
        return nn.Sequential(*layers)

    def _make_classifier(self,in_channels,num_env_class):
        classifier = nn.Sequential(
            nn.Conv2d(in_channels,2*in_channels,
                      kernel_size=3,stride=2,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(2*in_channels,4*in_channels,
                       kernel_size=3,stride=2,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256*8*8,2))
        
        
        return classifier

    def forward(self, x):
        x = self.csp_layer0(x)

        stage_out = {}
        for id in self.layer_id:
            x = getattr(self, 'layer'+str(id))(x)
            stage_out[str(id)] = x

        fpn_in_feats = [stage_out[str(id)] for id in self.fpn_layer_id]
        fpn_in_feats = [c2s(x, self.is_onnx) for c2s, x in zip(self.c2s_list, fpn_in_feats)]

        fpn_out = self.fpn(fpn_in_feats)
        x = fpn_out[0]
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]

    def init_weights(self, num_layers, pretrained=True):
        if pretrained:

            for head in self.heads:
              if 'reid' in head:
                continue
              final_layer = self.__getattr__(head)
              for i, m in enumerate(final_layer.modules()):
                  if isinstance(m, nn.Conv2d):
                      if m.weight.shape[0] == self.heads[head]:
                          if 'hm' in head:
                              nn.init.constant_(m.bias, -2.19)
                          else:
                              nn.init.normal_(m.weight, std=0.001)
                              nn.init.constant_(m.bias, 0)
            #pretrained_state_dict = torch.load(pretrained)
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            print('=> imagenet pretrained model dose not exist')
            print('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

def conv_with_kaiming_uniform(use_gn=False, use_relu=False):
    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias=False if use_gn else True
        )
        nn.init.kaiming_uniform_(conv.weight, a=1)
        if not use_gn:
            nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if use_relu:
            module.append(nn.ReLU(inplace=True))
          #  module.append(Mish())
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv

def get_perception_net(num_layers, heads, head_conv, opt, is_onnx):
  block_class, layers = resnet_spec[num_layers]

  model = PoseResNet(block_class, layers, heads, head_conv=head_conv, opt=opt, is_onnx=is_onnx)
  model.init_weights(num_layers, pretrained=True)
  return model
