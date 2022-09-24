from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4) #避免取值取到0，在计算loss时log(0)为nan或者inf
  return y


def _extract_mid_point(x):
    mid_point = x[...,::1]
    return mid_point
def _extract_corner_point(x):
    corner_point = x[...,1::1]
    return corner_point

def _compute_mp_angular(a,b):
    
      
      
      
    a_norm = torch.norm(a,dim=3) #32,10,8
    b_norm = torch.norm(b,dim=3) #32,10,8
    a_dot_b = a * b 
    cos_theta = torch.acos(a_dot_b/(a_norm*b_norm),dim=2)
    return cos_theta
 
def _compute_cp_angular(a,b):
    
    
    pass
    # a=np.array([2,3])
    # b=np.array([3,2])
    # #计算a的范数（长度）
    # a_norm=np.linalg.norm(a)
    # #计算b的范数（长度）
    # b_norm=np.linalg.norm(b)
    # #计算a和b的点积（相应位置相乘再把结果相加）
    # a_dot_b=a.dot(b)
    # #使用余弦定理计算cos_there的弧度值
    # cos_theta=np.arccos(a_dot_b/(a_norm*b_norm))
    # #将弧度转化为度
    # print(np.rad2deg(cos_theta))
def _gather_feat(feat, ind, mask=None):
    """
      args: feat = batch,channel*k,1
            ind = batch,k
            
      return feat = batch,k,1
    """
    #feat 3, 2, 128, 128
    #ind 3, 10
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
   # print(feat.size())
    feat = feat.gather(1, ind)
   # print(ind.size())
   # print(feat.size())
    if mask is not None:
      #  print(mask.size())
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)

    return feat

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)