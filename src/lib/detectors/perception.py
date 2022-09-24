from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from optparse import Values
#import pandas as pd
import numpy as np
import copy
np.set_printoptions(threshold=np.inf)
from pytorch_grad_cam import GradCAM,EigenCAM
import cv2
#import numpy as np
from progress.bar import Bar
import time
import torch
torch.set_printoptions(profile='full')

import torch.nn as nn
import torch.nn.functional as F
try:
  from external.nms import soft_nms_39
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import multi_pose_decode
from models.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform,affine_transform,draw_overlay,CLASSES_cn
from utils.post_process import multi_pose_post_process
from utils.debugger import Debugger
#from torchvision.models._utils.IntermediateLayerGetter
from .base_detector import BaseDetector
# CLASSES_cn = [
#         "不确定", "背景", "井盖", "立柱", "灯杆", "凳子", "台阶",
#         "草地", "石子路", "机动车道", "人行道", '路牌', '建筑', '墙面', '栅栏', '单台', '垃圾桶', '垃圾站', '消防栓'
# ]

class MultiPoseDetector(BaseDetector):
  def __init__(self, opt):
    super(MultiPoseDetector, self).__init__(opt)
    self.flip_idx = opt.flip_idx
    self.num_joints=8
  def process(self, images, return_time=True):    
    with torch.no_grad():
      torch.cuda.synchronize()
      output = self.model(images)[-1]
      ######## Extract output for each head ##########
      if self.opt.scene:
        values, label_id = torch.max(F.log_softmax(output['scene']),dim=1)
      else:
        label_id = torch.Tensor([0]).to('cuda:0')
      output['hm'] = output['hm'].sigmoid_()
      if not self.opt.dep:
        output['dep'] = None
      else:
        output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
      if self.opt.hm_state:
        output['hm_state'] = output['hm_state'].sigmoid_()
      if self.opt.hm_hp and not self.opt.mse_loss:
        output['hm_hp'] = output['hm_hp'].sigmoid_()
      if self.opt.seg:
        output['seg'] = torch.softmax(output['seg'],1)
        output['seg'] = torch.argmax(output['seg'],1)        
      kps_vis = output['hps_vis'].sigmoid_()
      reg = output['reg'] if self.opt.reg_offset else None
      hm_hp = output['hm_hp'] if self.opt.hm_hp else None
      hp_offset = output['hp_offset']
      torch.cuda.synchronize()
      forward_time = time.time()
      dets = multi_pose_decode(
          output['hm'], output['hm_state'], output['wh'], output['hps'],output['dep'],label_id,
          reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K, kps_vis=kps_vis) #hp_offset -> hp_offset

    if return_time:
        return output, dets, forward_time


  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = multi_pose_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'],self.opt.num_classes)

    for j in range(1,self.num_classes + 1):
      if self.opt.hm_state:
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 32)        
      else:
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 21)

      # import pdb; pdb.set_trace()
      dets[0][j][:, :4] /= scale
      dets[0][j][:, 5:] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if len(self.scales) > 1 or self.opt.nms:
         soft_nms_39(results[j], Nt=0.5, method=2)
        
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    return results    

  def debug(self, debugger, images, dets, output, grayscale_cam, scale=1):
    dets = dets.detach().cpu().numpy().copy()
    dets[:, :, :4] *= self.opt.down_ratio
    dets[:, :, 5:39] *= self.opt.down_ratio
    img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = img[:,:,:3]
    img = np.clip(((
      img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
    pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
    debugger.add_blend_img(img, pred, 'pred_hm',0.8,'pred_hm')
    if self.opt.hm_hp:
      pred = debugger.gen_colormap_hp(
        output['hm_hp'][0].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hmhp',0.8,'pred_hmhp')  
    ######## attention #########
    target_layers = [self.model.layer4]
    model1 = nn.Sequential(
      self.model.csp_layer0,
      self.model.layer1,
      self.model.layer2,
      self.model.layer3,
      self.model.layer4, 
    )
    if self.opt.vis_cam:
      cam = EigenCAM(model1,target_layers,use_cuda=False)
      grayscale_cam = cam(images)[0,:,:]
      grayscale_cam = np.maximum(grayscale_cam, 0)
      grayscale_cam = grayscale_cam/np.max(grayscale_cam)
      grayscale_cam = np.uint8(255*grayscale_cam)
      grayscale_cam = cv2.applyColorMap(grayscale_cam,cv2.COLORMAP_JET)
      debugger.add_blend_img(img, grayscale_cam, 'pred_att',0.8,'pred_att')
      ######## seg process ########
    if self.opt.seg:
      seg = output['seg'][0].detach().cpu().numpy()
      seg_img = cv2.resize(img,(512,512))
      seg_label = draw_overlay(seg_img, seg, CLASSES_cn, textscale=0.8)
      debugger.add_img(seg_label, img_id='seg')

  def show_results(self, debugger, image, results,image_name,output,tracker=None):     
    debugger.add_img(image, img_id='multi_pose')
    cls_box = {0:0,1:0,2:0,3:0}
    try:
      scene_id = int(results[1][0][-2])
    except:
      scene_id = 1
    if self.opt.dep:
      bird_view = np.ones((500, 500, 3), dtype=np.uint8) * 230
      debugger.add_img(bird_view,img_id='bird')
    if self.opt.scene:
      debugger.add_coco_scene(scene_id, img_id='multi_pose')
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        if bbox[4] > self.opt.vis_thresh:
            debugger.add_coco_bbox(j,bbox[:5],img_id='multi_pose')
            debugger.add_coco_line(bbox[:21],j-1, bbox[4], img_id='multi_pose')
            debugger.add_coco_hp(j,bbox[5:21],bbox[21:29], img_id='multi_pose') 
            if self.opt.dep:
              debugger.add_coco_depth(bbox,img_id='multi_pose')
              debugger.add_coco_bev(j-1,bbox,img_id='bird')

    
    debugger.save_all_imgs(path=self.opt.save_folder,opt=self.opt,prefix='',name=image_name)
