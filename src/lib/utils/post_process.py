from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.lib import arraysetops
from torch import det, threshold
from .image import transform_preds
from .ddd_utils import ddd2locrot


def get_pred_depth(depth):
  return depth

def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)
  

def ddd_post_process_2d(dets, c, s, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  include_wh = dets.shape[2] > 16
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (opt.output_w, opt.output_h))
    classes = dets[i, :, -1]
    for j in range(opt.num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :3].astype(np.float32),
        get_alpha(dets[i, inds, 3:11])[:, np.newaxis].astype(np.float32),
        get_pred_depth(dets[i, inds, 11:12]).astype(np.float32),
        dets[i, inds, 12:15].astype(np.float32)], axis=1)
      if include_wh:
        top_preds[j + 1] = np.concatenate([
          top_preds[j + 1],
          transform_preds(
            dets[i, inds, 15:17], c[i], s[i], (opt.output_w, opt.output_h))
          .astype(np.float32)], axis=1)
    ret.append(top_preds)
  return ret

def ddd_post_process_3d(dets, calibs):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  for i in range(len(dets)):
    preds = {}
    for cls_ind in dets[i].keys():
      preds[cls_ind] = []
      for j in range(len(dets[i][cls_ind])):
        center = dets[i][cls_ind][j][:2]
        score = dets[i][cls_ind][j][2]
        alpha = dets[i][cls_ind][j][3]
        depth = dets[i][cls_ind][j][4]
        dimensions = dets[i][cls_ind][j][5:8]
        wh = dets[i][cls_ind][j][8:10]
        locations, rotation_y = ddd2locrot(
          center, alpha, dimensions, depth, calibs[0])
        bbox = [center[0] - wh[0] / 2, center[1] - wh[1] / 2,
                center[0] + wh[0] / 2, center[1] + wh[1] / 2]
        pred = [alpha] + bbox + dimensions.tolist() + \
               locations.tolist() + [rotation_y, score]
        preds[cls_ind].append(pred)
      preds[cls_ind] = np.array(preds[cls_ind], dtype=np.float32)
    ret.append(preds)
  return ret

def ddd_post_process(dets, c, s, calibs, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  dets = ddd_post_process_2d(dets, c, s, opt)
  dets = ddd_post_process_3d(dets, calibs)
  return dets


def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds( dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret



def multi_pose_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  #??????????????????bboxes, scores, kps, clses???concate??????????????????
  ret = []
  for i in range(dets.shape[0]):
    top_preds={}
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = transform_preds(dets[i, :, 5:21].reshape(-1, 2), c[i], s[i], (w, h)) 
    classes = dets[i,:,-4]
    classes_state = dets[i,:,-3]
    scene = dets[i,:,-2]
    pts_vis = dets[i,:,21:29]    
    for j in range (num_classes):
      thresh = (dets[i,:,4]>0)
      ind = (classes==j) 
      inds = np.logical_and(thresh,ind)
      # inds = (classes==j)
      # # scores = dets[i,inds,4:5]     
      top_preds[j+1]=np.concatenate([bbox.reshape(-1,4)[inds],dets[i,inds,4:5],pts.reshape(-1,16)[inds],pts_vis.reshape(-1,8)[inds],
                                      classes_state.reshape(-1,1)[inds],scene.reshape(-1,1)[inds],dets[i, inds, -1].reshape(-1,1).astype(np.float32)],axis=1).astype(np.float32).tolist()  
      if len(top_preds[j+1])!=0:
        top_preds[j+1]=py_cpu_nms(top_preds[j+1],0.1)    
      ret.append(top_preds)
  return ret

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    #x1???y1???x2???y2?????????score??????
    dets=np.array(dets)
    #print(dets[:,:4])
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    #???????????????????????????
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #??????score?????????????????????
    order = scores.argsort()[::-1]

    keep = [] #????????????????????????
    while order.size > 0:
        i = order[0]
        keep.append(i) #??????????????????box????????????????????????
        #??????????????????,???????????????
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        #?????????????????????,?????????????????????0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #??????IoU??????????????? /?????????1+??????2-???????????????
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #print(ovr)
        #??????IoU???????????????box
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1] #??????ovr??????????????????order???????????????,??????????????????????????????????????????
       
    return dets[keep]

def centernet_nms(cate_preds):
      cate_preds = np.array(cate_preds)
      x1 = cate_preds[:,0]
      y1 = cate_preds[:,1]
      x2 = cate_preds[:,2]
      y2 = cate_preds[:,3]
      score = cate_preds[4]
      area = ( x2 - x1 + 1 ) * ( y2 - y1 + 1 ) #????????????????????????0?????????+1
      order = np.argsort(score)[::-1]    
      #?????????????????????????????????????????????
      keep=[]      
      #??????top_left,bottom_right
      if order.size()>0:
        for i in order[1:]:
              x_tl = np.maximum(cate_preds[i][0],cate_preds[-1][0])
              y_tl = np.maximum(cate_preds[i][0],cate_preds[-1][0])
              x_br = np.maximum(cate_preds[i][0],cate_preds[-1][0])
              y_br = np.maximum(cate_preds[i][0],cate_preds[-1][0])      
              inter_area = (x_br-x_tl)*(y_br-y_tl)
              iou = inter_area / (area[i] + area[order[0]])
              if (iou > 0.7):
                order.pop(i)
        keep.append(cate_preds[order.pop(0)])
              
            
# def nms(top_preds):
#   top_preds=np.array(top_preds)
#   #???top_left???right_bottom
#   x1 = top_preds[:,0]
#   y1 = top_preds[:,1]
#   x2 = top_preds[:,2]
#   y2 = top_preds[:,3]
#   scores = top_preds[:4]
#   #????????????????????????
#   areas=(x2-x1+1)*(y2-y1+1)
#   order = scores.argsort()[::-1]
  
#   keep=[]
#   while order.size()>0:
#     i = order[0]
#     keep.append(i)#??????????????????box????????????????????????
#     #??????????????????
#     xx1=np.maximum(x1[i],x1[order[i:]])
#     xx2=np.minimum(x2[i],x2[order[i:]])
#     yy1=np.maximum(y1[i],y1[order[i:]])
#     yy2=np.minimum(y2[i],y2[order[i:]])

#     w = np.maximum(0.0,xx2-xx1+1)
#     h = np.maximum(0.0,yy2-yy1+1)
#     inter = w*h
#     #??????iou
#     iou = inter/(areas[i]+areas[order[i:]]-inter) 
#     #??????iou???????????????bbox
#     inds = np.where(iou<=0.7)
#     order = order[inds+1]
    
#   return keep




  

  # return dets