from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import defaultdict
from cv2 import threshold

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

class COCOHP(data.Dataset):
  num_classes = 14
  num_joints = 8
  num_states = 4
  default_resolution = [512, 512]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
  flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
              [11, 12], [13, 14], [15, 16]]
  def __init__(self, opt, split):
    super(COCOHP, self).__init__()
    self.classes_for_all_imgs = []
    self.edges = [[0, 1], [0, 2], [1, 3], [2, 4], 
                  [4, 6], [3, 5], [5, 6], 
                  [5, 7], [7, 9], [6, 8], [8, 10], 
                  [6, 12], [5, 11], [11, 12], 
                  [12, 14], [14, 16], [11, 13], [13, 15]]
    
    self.acc_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    self.data_dir = os.path.join(opt.data_dir, '/mnt/data/qianch/dataset/self_data/mat_det/exp/09_20_static_seg')
    #self.data_dir = os.path.join(opt.data_dir, '/mnt/data/qianch/dataset/0601_exp3')

    #self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
    self.img_dir = self.data_dir
    if split == 'test':
      self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'image_info_test-dev2017.json').format(split)
    else:
      self.annot_path = os.path.join(
        self.data_dir, 'annotations', 
        '{}.json').format(split)
    self.max_objs = 10
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.split = split
    self.opt = opt

    print('==> initializing coco 2017 {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    image_ids = self.coco.getImgIds()

    if split == 'train':
      self.images = []
      for img_id in image_ids:
        labels = []
        idxs = self.coco.getAnnIds(imgIds=[img_id])
        if len(idxs) > 0:
          self.images.append(img_id)
          ann_ids = self.coco.getAnnIds(imgIds=[img_id])
          anns = self.coco.loadAnns(ids=ann_ids)
          for ann in anns:
              labels.append(ann['category_id'])
          if 3 in labels:
                label = 2
          elif 1 in labels:
                label = 0
          else:
                label = 1
          self.classes_for_all_imgs.append(label)
    else:
      self.images = image_ids
    self.num_samples = len(self.images)
    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))
  
  def get_classes_for_all_imgs(self):
        return self.classes_for_all_imgs
  
  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = 1
        for dets in all_bboxes[image_id][cls_ind]:
          bbox = dets[:4]
          score = dets[4]
          bbox_out  = list(map(self._to_float, bbox))
          keypoints = np.array(dets[5:21], dtype=np.float32).tolist()
          keypoints  = list(map(self._to_float, keypoints))
          scene = int(dets[30])
          detection = {
              "cls_id": int(cls_ind-1),
              "state_id": int(dets[-1]),
              "image_id": int(image_id),
            #  "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score)),
              "keypoints": keypoints,
              "scene":scene
          }
          if score > 0.1:
            detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
              open('{}/results.json'.format(save_dir), 'w'))


  def run_eval(self, results, save_dir):

    self.save_results(results, save_dir)
