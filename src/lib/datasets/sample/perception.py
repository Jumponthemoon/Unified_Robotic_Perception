from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import albumentations as A
import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug, normalization, _nms
from utils.image import get_affine_transform, affine_transform,spatial2channel
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math
import torch.nn as nn
from numpy import ones,vstack
from numpy.linalg import lstsq
from scipy.stats import linregress

class MultiPoseDataset(data.Dataset):
  def __init__(self):
      self.scenes = {'indoor':0,'outdoor':1}

  def _coco_box_to_bbox(self, box):
    # bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
    #                 dtype=np.float32)
    bbox = np.array([box[0], box[1], box[2], box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i
  
  def _scene_to_id(self,scene):
    
    return self.scenes[scene]


  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)
    bbox_list = []
    img = cv2.imread(img_path)
    label_img = cv2.imread(img_path.replace('train','label'),0)
    ###################### Aug ######################
    for i in range(num_objs):
        anns[i]['bbox'] = list(anns[i]['bbox'])
        anns[i]['bbox'].append(i)
        bbox_list.append(anns[i]['bbox'])
    points_list=[]
    for i in range(num_objs):
      for j in range(len(anns[i]['keypoints'][0])):
          if (j+1) % 3 != 0:
            points_list.append(anns[i]['keypoints'][0][j])
    points_list=np.reshape(points_list,(-1,2))
    points_list=[tuple(k) for k in points_list]
    transform = A.Compose(
      [ A.RandomBrightnessContrast (brightness_limit=0.3, 
                                    contrast_limit=0.3, 
                                    brightness_by_max=True, 
                                    always_apply=False, p=0.5),
        A.MotionBlur(blur_limit=(10,20),p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ISONoise(p=0.5),
        A.RandomGamma(),
        A.Rotate(limit=(-10,10),p=1)    
       ],
      bbox_params=A.BboxParams(format='pascal_voc',min_area=0,min_visibility=0.1), 
      keypoint_params=A.KeypointParams(format='xy',remove_invisible=False))  
    transformation = transform(image=img,keypoints=points_list,bboxes=bbox_list,mask=label_img)
   # transformation = transform(image=img,keypoints=points_list,bboxes=bbox_list)
    img=transformation['image']
    img_label=transformation['mask']
    trans_keypoints=transformation['keypoints']
    trans_bboxes = transformation['bboxes']
    no_box_id = []
    #提取aug后丢弃的框的id，并将丢掉的框置空
    for box in trans_bboxes:
          box_id = box[-1]
          anns[box_id]['bbox'] = box
          if box_id not in range(num_objs):
                anns[box_id]['bbox'] = []
               # no_box_id.append(box_id)
    #
    for i in no_box_id:
          anns[i]['bbox']=[]
    new_keypoints=[]
    for ele in trans_keypoints:
      x,y=ele
      new_keypoints.extend((x,y))
      new_keypoints.append(0)
    ###################################################
    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    s = max(img.shape[0], img.shape[1]) * 1.0
    rot = 0
    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])

        c[0] = int(img.shape[1]//2)
        c[1] = int(img.shape[0]//2)
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      if np.random.random() < self.opt.aug_rot:
        rf = self.opt.rotate
        rot = np.clip(np.random.randn()*rf, -rf*2, rf*2)

      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1
        

    trans_input = get_affine_transform(
      c, s, rot, [self.opt.input_w, self.opt.input_h])
    inp_ori = cv2.warpAffine(img, trans_input, 
                         (self.opt.input_w, self.opt.input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp_ori.astype(np.float32) / 255.)

    img_label = cv2.warpAffine(img_label, trans_input, 
                         (self.opt.input_w, self.opt.input_h),
                         flags=cv2.INTER_LINEAR)

    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = spatial2channel(inp)
    inp = inp.transpose(2, 0, 1)
    
    num_joints = self.num_joints
    output_w = self.opt.output_w
    output_h = self.opt.output_h

    trans_output_rot = get_affine_transform(c, s, rot, [output_w, output_h])
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

    hm = np.zeros((self.num_classes, output_w, output_h), dtype=np.float32)
    hm_state = np.zeros((self.num_states, output_w, output_h), dtype=np.float32)
    hm_hp = np.zeros((num_joints, output_w, output_h), dtype=np.float32)
    dense_kps = np.zeros((num_joints, 2, output_w, output_h), 
                          dtype=np.float32)
    dense_kps_mask = np.zeros((num_joints, output_w, output_h), 
                               dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)
    kps_vis = np.zeros((self.max_objs, num_joints), dtype=np.float32)  
    kps_vis[:, :] = -1
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)

    hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
    hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)
    hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)
    if self.opt.dep:
      dep = np.zeros((self.max_objs,1),dtype=np.float32)
    segmap = np.zeros((512,512),dtype=np.float32)

    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian
    img = cv2.copyMakeBorder(img,200,200,200,200,cv2.BORDER_CONSTANT,value=0)
    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      if self.opt.dep:
        dep[k]=ann['depth']
      if self.opt.scene:
        scene = self._scene_to_id(ann['scene'])
      bbox = self._coco_box_to_bbox(ann['bbox'])
      cls_id = int(ann['category_id']) - 1
      if cls_id == 5:
        cls_id = 3
      if cls_id == 4 or cls_id == 10:
        cls_id = 4
      state_id = int(ann['category_state_id']) - 1
      ############### Transform Keypoint #############
      for i in range (len(ann['keypoints'][0])):
            if (i+1)%3!=0:
              ann['keypoints'][0][i]=new_keypoints[k*24+i]
      pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3)
      ##################### Vis ######################
      #Point
      font = cv2.FONT_HERSHEY_SIMPLEX
      for i in range(len(pts)):
          cv2.circle(img,(int(pts[i][0]+200),int(pts[i][1])+200), 5, (0,0,255),-1)
          cv2.putText(img, str(k), (int(pts[i][0]+100),int(pts[i][1])+100), font, 1, (255, 0, 0), thickness=3, lineType=cv2.LINE_AA)
     # Line
      for i in range(2):
           x1=int(pts[i][0]+100)
           y1=int(pts[i][1]+100)
           x2=int(pts[i+1][0]+100)
           y2=int(pts[i+1][1]+100)
           cv2.line(img,(x1,y1),(x2,y2),(255,255,0),3)
     # Box
      cv2.rectangle(img, (int(bbox[0]+200), int(bbox[1]+200)),
                    (int(bbox[2]+200), int(bbox[3]+200)), (0,255,0), 4)
      cv2.circle(img,(int((bbox[0]+bbox[2]+400)/2), int((bbox[1]+bbox[3]+400)/2)),5,(255,255,0),-1)
      cv2.circle(img,(int((bbox[0]+bbox[2]+400)/2),int((bbox[1]+bbox[3]+400)/2)), 5, (255,0,255),-1)
      x = int((bbox[0]+bbox[2]+400)/2)
      y = int((bbox[1]+bbox[3]+400)/2)

      h_ori, w_ori = bbox[3]-bbox[1],bbox[2]-bbox[0]
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        pts[:, 0] = width - pts[:, 0] - 1
        for e in self.flip_idx:
          pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()
      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox = np.clip(bbox, 0, self.opt.output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if (h > 0 and w > 0) or (rot != 0):
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius)) 
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        wh[k] = 1. * w, 1. * h
        ind[k] = ct_int[1] * self.opt.output_h + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        num_kpts = pts[:, 2].sum()
        if num_kpts == 0:
          hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
          hm_state[state_id, ct_int[1], ct_int[0]] = 0.9999
          reg_mask[k] = 0
        
        hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        hp_radius = self.opt.hm_gauss \
                    if self.opt.mse_loss else max(0, int(hp_radius)) 
        ################ Make segmap by line ###############
        if self.opt.segmap:
          if cls_id != 2:
            pts_seg = pts.copy()
            for j in range(3):
              pts_seg[j, :2] = affine_transform(pts_seg[j, :2], trans_output_512)
            seg_radius = int(gaussian_radius((math.ceil(h_ori*512*0.8/1080),math.ceil(w_ori*512*0.8/1920))))        
            for j in range(2):   
                points = [(pts_seg[j][0],pts_seg[j][1]),(pts_seg[j+1][0],pts_seg[j+1][1])]
                x_coords, y_coords = zip(*points)
                b = vstack([x_coords,ones(len(x_coords))]).T
                m_line, c_line = lstsq(b, y_coords,rcond=None)[0]
                t,b=min(pts_seg[j][1],pts_seg[j+1][1]),max(pts_seg[j][1],pts_seg[j+1][1])
                
                for i in range(int(t),int(b)):
                    y = i
                    if pts_seg[j][0] != pts_seg[j+1][0]:
                      x = (y-c_line)/m_line
                    else:
                      x = pts_seg[j][0]                
                    draw_umich_gaussian(segmap,(float(x),float(y)),seg_radius)
                        
        for j in range(num_joints):
          if pts[j, 2] > 0:
            pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
            ########################### Vis #################################
            cv2.circle(inp_ori,(int(pts[j][0]),int(pts[j][1])), 1, (0,0,255),-1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(inp_ori, str(j), (int(pts[j][0]),int(pts[j][1])), font, 1, (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            if cls_id == 2:# and j > 0:
              kps_mask[k, j * 2: j * 2 + 2] = 0
            elif cls_id  in [0,1,4] and j > 2:
              kps_mask[k, j * 2: j * 2 + 2] = 0
            else:
              kps_mask[k, j * 2: j * 2 + 2] = 1
            kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
            #kps_mask[k, j * 2: j * 2 + 2] = 1
            pt_int = pts[j, :2].astype(np.int32)
            hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
            hp_ind[k * num_joints + j] = pt_int[1] * self.opt.output_h + pt_int[0]
            hp_mask[k * num_joints + j] = 1
            if self.opt.dense_hp:
              # must be before draw center hm gaussian
              draw_dense_reg(dense_kps[j], hm[cls_id], ct_int, 
                              pts[j, :2] - ct_int, radius, is_offset=True)
              draw_gaussian(dense_kps_mask[j], ct_int, radius)
            draw_gaussian(hm_hp[j], pt_int, hp_radius)
          kps_vis[k, j] = 1 if pts[j, 2] == 1 and pts[j,0] > 0 and pts[j,1] > 0 else 0  

        draw_gaussian(hm[cls_id], ct_int, radius)
        draw_gaussian(hm_state[state_id], ct_int, radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                       ct[0] + w / 2, ct[1] + h / 2, 1] + 
                       pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])

    if rot != 0:
      hm = hm * 0 + 0.9999
      reg_mask *= 0
      kps_mask *= 0

    if self.opt.scene:
      if scene == 0:
        hm = hm * np.zeros(hm.shape)
        reg_mask = reg_mask * np.zeros(reg_mask.shape)
        wh = wh * np.zeros(wh.shape)
        kps = kps * np.zeros(kps_mask.shape)
        kps_mask = kps_mask * np.zeros(kps_mask.shape)
        hm_state = hm_state * np.zeros(hm_state.shape)
        kps_vis = kps_vis * np.zeros(kps_vis.shape)
        reg = reg * np.zeros(reg.shape)
        hm_hp = hm_hp * np.zeros(hm_hp.shape)
        hp_offset = hp_offset * np.zeros(hp_offset.shape)
        hp_mask = hp_mask * np.zeros(hp_mask.shape)
        hp_ind = hp_ind * np.zeros(hp_ind.shape, dtype=np.int64)
        ind = ind * np.zeros(ind.shape, dtype=np.int64)
      else:
        hm = hm * np.ones(hm.shape)
        reg_mask = reg_mask * np.ones(reg_mask.shape)
        wh = wh * np.ones(wh.shape)
        kps = kps * np.ones(kps_mask.shape)
        kps_mask = kps_mask * np.ones(kps_mask.shape)
        hm_state = hm_state * np.ones(hm_state.shape)
        kps_vis = kps_vis * np.ones(kps_vis.shape)
        reg = reg * np.ones(reg.shape)
        hm_hp = hm_hp * np.ones(hm_hp.shape)
        hp_offset = hp_offset * np.ones(hp_offset.shape)
        hp_mask = hp_mask * np.ones(hp_mask.shape)
        hp_ind = hp_ind * np.ones(hp_ind.shape, dtype=np.int64)
        ind = ind * np.ones(ind.shape, dtype=np.int64)
        
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
            'hps': kps, 'hps_mask': kps_mask,'hm_state': hm_state, 'hps_vis':kps_vis}

    ########################################## update ret ##############################################
    if self.opt.seg:
      ret.update({'seg':img_label})
    if self.opt.dep:
      ret.update({'dep':dep})
    if self.opt.scene:
      ret.update({'scene':scene})
    if self.opt.segmap:
      ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
           'hps': kps, 'hps_mask': kps_mask,'hm_state': hm_state, 'hps_vis':kps_vis, 'segmap':segmap}
    if self.opt.dense_hp:
      dense_kps = dense_kps.reshape(num_joints * 2, output_res, output_res)
      dense_kps_mask = dense_kps_mask.reshape(
        num_joints, 1, output_res, output_res)
      dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=1)
      dense_kps_mask = dense_kps_mask.reshape(
        num_joints * 2, output_res, output_res)
      ret.update({'dense_hps': dense_kps, 'dense_hps_mask': dense_kps_mask})
      del ret['hps'], ret['hps_mask']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.hm_hp:
      ret.update({'hm_hp': hm_hp})
    if self.opt.reg_hp_offset:
      ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 40), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
    return ret
