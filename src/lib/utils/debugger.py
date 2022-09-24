from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import cv2
from torch import int32
from .ddd_utils import compute_box_3d, project_to_image, draw_box_3d
# from skspatial.objects import Line
# from skspatial.objects import Point
import matplotlib.pyplot as plt
import pylab
import sys
sys.path.append('/home/dev/qianch/centernet_outdoor/src/lib/utils')
from .image import get_affine_transform, affine_transform,spatial2channel
#line = Line(point=[0, 0], direction=[1, 1])
#point = Point([1, 4])

# point_projected = line.project_point(point)
# line_projection = Line.from_points(point, point_projected)
class Debugger(object):
  # def __init__(self, ipynb=False, theme='black', 
  #              num_classes=-1, dataset=None, down_ratio=4, image_name=None):
  def __init__(self, ipynb=False, theme='black', 
               num_classes=-1, dataset=None, down_ratio=4):
    self.ipynb = ipynb
    if not self.ipynb:
      import matplotlib.pyplot as plt
      self.plt = plt
    self.imgs = {}
    self.theme = theme
    colors = [(color_list[_]).astype(np.uint8) \
            for _ in range(len(color_list))]
    self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
    if self.theme == 'white':
      self.colors = self.colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
      self.colors = np.clip(self.colors, 0., 0.6 * 255).astype(np.uint8)
    self.dim_scale = 1
    if dataset == 'coco_hp':
      self.scenes = ['Indoor','Outdoor','Unknown']
      self.names = ['pole',
                    'tree',
                    'manhole',
                    'gate',
                    'signpost',
                    'gate',
                    'static_chair',
                    'static_bin',
                    'building_entry',
                    'community_entry',
                    'flat_board',
                    'static_dump_station',
                    'fire_hydrant']
      self.names_state = ['half_open','open','close','uncertain']
      self.num_class = 2
      self.num_joints = 8
      self.edges = [[0, 1], [0, 2], [1, 3], [2, 4], 
                    [3, 5], [4, 6], [5, 6], 
                    [5, 7], [7, 9], [6, 8], [8, 10], 
                    [5, 11], [6, 12], [11, 12], 
                    [11, 13], [13, 15], [12, 14], [14, 16]]
      self.ec = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),  
                 (255, 0, 0), (0, 0, 255), (255, 0, 255),
                 (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255),
                 (255, 0, 0), (0, 0, 255), (255, 0, 255),
                 (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)]
      self.colors_hp = [(255, 0, 255), (255, 0, 0), (0, 0, 255), 
        (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
        (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
        (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
        (255, 0, 0), (0, 0, 255)]
    elif num_classes == 80 or dataset == 'coco':
      self.names = coco_class_name
    elif num_classes == 20 or dataset == 'pascal':
      self.names = pascal_class_name
    elif dataset == 'gta':
      self.names = gta_class_name
      self.focal_length = 935.3074360871937
      self.W = 1920
      self.H = 1080
      self.dim_scale = 3
    elif dataset == 'viper':
      self.names = gta_class_name
      self.focal_length = 1158
      self.W = 1920
      self.H = 1080
      self.dim_scale = 3
    elif num_classes == 3 or dataset == 'kitti':
      self.names = kitti_class_name
      self.focal_length = 721.5377
      self.W = 1242
      self.H = 375
    num_classes = len(self.names)
    self.down_ratio=down_ratio
    # for bird view
    self.world_size = 64
    self.out_size = 384
    #self.image_name = image_name

  def add_img(self, img, img_id='default', revert_color=False):
    if revert_color:
      img = 255 - img
    self.imgs[img_id] = img.copy()
  
  def add_mask(self, mask, bg, imgId = 'default', trans = 0.8):
    self.imgs[imgId] = (mask.reshape(
      mask.shape[0], mask.shape[1], 1) * 255 * trans + \
      bg * (1 - trans)).astype(np.uint8)
  
  def show_img(self, pause = False, imgId = 'default'):
    cv2.imshow('{}'.format(imgId), self.imgs[imgId])
    if pause:
      cv2.waitKey()
  
  def add_blend_img(self, back, fore, img_id='blend', trans=0.8, name=None):
    # if self.theme == 'white':
    #   fore = 255 - fore
    if name == 'pred_att':
      self.imgs[img_id] = back*0.7+fore*0.3

    if name == 'pred_hm' or name =='pred_hmhp':
      fore = 255 - fore  
      if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
        fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
      if len(fore.shape) == 2:
        fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
      self.imgs[img_id] = (back * (1. - trans) + fore * trans)
      self.imgs[img_id][self.imgs[img_id] > 255] = 255
      self.imgs[img_id][self.imgs[img_id] < 0] = 0
      self.imgs[img_id] = cv2.resize(self.imgs[img_id].astype(np.uint8).copy(),(800,800),interpolation=cv2.INTER_CUBIC)


      
  def gen_colormap(self, img, output_res=None):
    img = img.copy()
    c, h, w = img.shape[0], img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
    colors = np.array(
      self.colors, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
    if self.theme == 'white':
      colors = 255 - colors
    color_map = (img * colors).max(axis=2).astype(np.uint8)
    color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
    return color_map

  def gen_colormap_hp(self, img, output_res=None):
    c, h, w = img.shape[0], img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
    colors = np.array(
      self.colors_hp, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
    if self.theme == 'white':
      colors = 255 - colors
    color_map = (img * colors).max(axis=2).astype(np.uint8)
    color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
    return color_map


  def add_rect(self, rect1, rect2, c, conf=1, img_id='default'): 
    cv2.rectangle(
      self.imgs[img_id], (rect1[0], rect1[1]), (rect2[0], rect2[1]), c, 1)
    if conf < 1:
      cv2.circle(self.imgs[img_id], (rect1[0], rect1[1]), int(10 * conf), c, 1)
      cv2.circle(self.imgs[img_id], (rect2[0], rect2[1]), int(10 * conf), c, 1)
      cv2.circle(self.imgs[img_id], (rect1[0], rect2[1]), int(10 * conf), c, 1)
      cv2.circle(self.imgs[img_id], (rect2[0], rect1[1]), int(10 * conf), c, 1)

  def add_coco_bbox(self,j,bbox,img_id='default'):
     cv2.rectangle(self.imgs[img_id],
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])), (0,255,0), 2)
   
  def add_kp_vis(self, points, pts_vis, thresh, img_id='default'):
    points = np.array(points, dtype=np.int32).reshape(self.num_joints, 2)
    for i, idx in enumerate([i for i in range (8)]):
      color = self.colors[0] if pts_vis[i] > thresh else self.colors[1]
      cv2.circle(self.imgs[img_id], (points[idx,0], points[idx,1]), 10, color, -1)

  def add_txt(self, pts, cat, conf=1, show_txt=True, img_id='default'):
    pts = np.array(pts, dtype=np.int32)
    txt = '{}{:.1f}'.format(self.names_state[cat], conf)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 1, 2)[0]    
    c=(0,255,255)
    cv2.putText(self.imgs[img_id], txt, (pts[0], pts[1] + 20), 
                  font, 1, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)

  def add_state_tag(self, pts, cat, conf=1, show_txt=True, img_id='default'):
    x1,y1,x2,y2 = pts[:4]
    cx = int((x1+x2)//2)
    cy = int((y1+y2)//2)
    pts = np.array(pts[5:21], dtype=np.int32)
    cat = int(cat)
    c=(0,255,255)
    txt = '{}'.format(self.names_state[cat])
    font = cv2.FONT_HERSHEY_COMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.2, 1)[0]
    if show_txt:

      cv2.rectangle(self.imgs[img_id],
                   (cx-40, cy - cat_size[1] ),
                   (cx-40 + cat_size[0], cy ), (0,0,0), -1)
      cv2.putText(self.imgs[img_id], txt, (cx-40, cy), 
                  font, 0.2, (255,255,255), thickness=1, lineType=cv2.LINE_AA)    


  def project_point(self,p1_x,p1_y,p2_x,p2_y,p3_x,p3_y):
    a = int(p2_y - p1_y)
    b = int(p1_x - p2_x)
    c = int(p2_x * p1_y - p1_x * p2_y)
    x0, y0 = int(p3_x),int(p3_y)
    proj_x = (b*(-b*x0+a*y0)+a*c)/(-a**2-b**2)
    proj_y = (b*c-a*(-b*x0+a*y0))/(-a**2-b**2)

    return (int(proj_x), int(proj_y))
  

  def add_coco_depth(self,bbox,img_id='default'):
    box = bbox[:5]
    ct_x, ct_y = int((box[0]+box[2])//2),int((box[1]+box[3])//2)
    cv2.circle(self.imgs[img_id],(ct_x, ct_y), 4, (0,255,255), -1)
    ego_x, ego_y = (self.imgs[img_id].shape[1]//2,self.imgs[img_id].shape[0])
    num_pts = int((ct_y - ego_y)//20)
    cv2.line(self.imgs[img_id], (ego_x,ego_y), (ct_x,ct_y), [0,255,0], 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(self.imgs[img_id],str(int(bbox[-1])), (int(ct_x)+2, int(ct_y) - 2), 
                  font, 1, (0,255 ,255), thickness=3, lineType=cv2.LINE_AA)

  def add_coco_line(self, pts, cat, conf=1, show_txt=True, img_id='default'):
    x1,y1,x2,y2 = pts[:4]
    cx = int((x1+x2)//2)
    cy = int((y1+y2)//2)
    pts = np.array(pts[5:21], dtype=np.int32)
    cat = int(cat)
    c=(0,255,255)
    txt = '{}{:.1f}'.format(self.names[cat], conf)
    font = cv2.FONT_HERSHEY_COMPLEX
    cat_size = cv2.getTextSize(txt, font, 1, 1)[0]
    if show_txt:
      cv2.rectangle(self.imgs[img_id],
                   (cx-40, cy - cat_size[1]-20 ),
                   (cx-40 + cat_size[0], cy-20 ), (0,0,0), -1)
      cv2.putText(self.imgs[img_id], txt, (cx-40, cy - 20), 
                  font, 1, (255,255,255), thickness=1, lineType=cv2.LINE_AA)
                      
  def add_coco_hp(self, cat,points,vis, img_id='default'): 
    self.num_joints=8
    points = np.array(points, dtype=np.int32).reshape(self.num_joints, 2)
    vis = vis.reshape(self.num_joints, 1)
    if cat in [1,2]:
      for i in range(3):
        cv2.circle(self.imgs[img_id],
                  (points[i, 0], points[i, 1]), 4, self.colors_hp[2], -1)
    if cat == 3:
        cv2.circle(self.imgs[img_id],
                  (points[0, 0], points[0, 1]), 4, self.colors_hp[2], -1)

  def add_coco_scene(self, scene_id, img_id='default'): 
    txt = self.scenes[scene_id]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 2, 2)[0]
    cv2.rectangle(self.imgs[img_id],
                   (0, 0 ),
                   (10 + cat_size[0], 35+cat_size[1] ), (0,0,0), -1)
    cv2.putText(self.imgs[img_id], txt, (5,60), 
                   font, 2, (255, 255,255 ), thickness=3, lineType=cv2.LINE_AA)   
    

  def add_points(self, points, img_id='default'):
    num_classes = len(points)
    # assert num_classes == len(self.colors)
    for i in range(num_classes):
      for j in range(len(points[i])):
        c = self.colors[i, 0, 0]
        cv2.circle(self.imgs[img_id], (points[i][j][0] * self.down_ratio, 
                                       points[i][j][1] * self.down_ratio),
                   5, (255, 255, 255), -1)
        cv2.circle(self.imgs[img_id], (points[i][j][0] * self.down_ratio,
                                       points[i][j][1] * self.down_ratio),
                   3, (int(c[0]), int(c[1]), int(c[2])), -1)

  def show_all_imgs(self, pause=False, time=0):
    if not self.ipynb:
      for i, v in self.imgs.items():
        cv2.imshow('{}'.format(i), v)
      if cv2.waitKey(0 if pause else 1) == 27:
        import sys
        sys.exit(0)
    else:
      self.ax = None
      nImgs = len(self.imgs)
      fig=self.plt.figure(figsize=(nImgs * 10,10))
      nCols = nImgs
      nRows = nImgs // nCols
      for i, (k, v) in enumerate(self.imgs.items()):
        fig.add_subplot(1, nImgs, i + 1)
        if len(v.shape) == 3:
          self.plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
        else:
          self.plt.imshow(v)
      self.plt.show()

  def save_img(self, imgId='default', path='./cache/debug/'):
    cv2.imwrite(path + '{}.png'.format(imgId), self.imgs[imgId])
    
  def save_all_imgs(self, opt,path='/mnt/data/qianch/project/CenterNet/src/result_seg', prefix='', genID=False, name=None):
    if not os.path.exists(path):
        os.mkdir(path)
    if genID:
      try:
        idx = int(np.loadtxt(path + '/id.txt'))
      except:
        idx = 0
      prefix=idx
      np.savetxt(path + '/id.txt', np.ones(1) * (idx + 1), fmt='%d')
    count = 0

   # fig = plt.figure(figsize=(12,12),dpi=400)

    ################## 图片为val_gt+val_pret ###################
    #val_gt_path = '/home/dev/data_disk/qianch/dataset/0506_split_new/val_gt/'+name+'.jpg'
    if opt.vis_val:
        val_gt_path = opt.demo+'/'+name+'.png'
        val_gt = cv2.imread(val_gt_path)
        val_pred = self.imgs['multi_pose']
        pred_hm = self.imgs['pred_hm']
        pred_hmhp = self.imgs['pred_hmhp']

        val_gt = cv2.resize(val_gt,(1920,1080))
        val_pred = cv2.resize(val_pred,(1920,1080))
        pred_hm = cv2.resize(pred_hm,(1920,1920))
        pred_hmhp = cv2.resize(pred_hmhp,(1920,1920))
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.rectangle(val_gt,(0, 0),(400,60), (0,0,0), -1)
        cv2.rectangle(val_pred,(0, 0),(350,60), (0,0,0), -1)
        cv2.rectangle(pred_hm,(0, 0),(700,60), (0,0,0), -1)
        cv2.rectangle(pred_hmhp,(0, 0),(760,60), (0,0,0), -1)

        cv2.putText(val_gt, 'Original', (0, 50), font, 2, (255,255,255), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(val_pred, 'Prediction', (0, 50), font, 2, (255,255,255), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(pred_hm, 'Pred Center Heatmap', (0, 50), font, 2, (255,255,255), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(pred_hmhp, 'Pred Keypoint Heatmap', (0, 50), font, 2, (255,255,255), thickness=3, lineType=cv2.LINE_AA)
        h1 = np.hstack([val_pred,val_gt])
        h2 = np.hstack([pred_hmhp,pred_hm])
        img = np.vstack([h1,h2])
        name=prefix
        cv2.imwrite(path + '/{}.jpg'.format(name), img, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
    if opt.vis_map:
    ################ 写pred #############
    # for i, v in self.imgs.items():
    #   #  if i == 'multi_pose':
    #       cv2.imwrite(path + '/{}{}.jpg'.format(i,prefix), v, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
    ####### 拼接pred\hm\hmhp\segmap ####
        multi_pose = self.imgs['multi_pose']
        multi_pose = cv2.copyMakeBorder(multi_pose, 420, 420, 0, 0, cv2.BORDER_CONSTANT)
        multi_pose = cv2.resize(multi_pose,(512,512))
        pred_hm = self.imgs['pred_hm']
        pred_hmhp = self.imgs['pred_hmhp']
        segmap = self.imgs['segmap']
        multi_pose = cv2.resize(multi_pose,(800,800))
        h1 = np.hstack([multi_pose,segmap])
        h2 = np.hstack([pred_hm,pred_hmhp])
        img = np.vstack([h1,h2])      
        print(prefix)  
        cv2.imwrite(path + '/{}.jpg'.format(prefix), img, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
    else:
      for i, v in self.imgs.items():
        if i == 'bird':
          v = v[140:500,100:400]
          cv2.imwrite(path + '/{}{}.jpg'.format(prefix, i), v, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        else:
          cv2.imwrite(path + '/{}{}.jpg'.format(name, i), v, [int(cv2.IMWRITE_JPEG_QUALITY), 30])

  def remove_side(self, img_id, img):
    if not (img_id in self.imgs):
      return
    ws = img.sum(axis=2).sum(axis=0)
    l = 0
    while ws[l] == 0 and l < len(ws):
      l+= 1
    r = ws.shape[0] - 1
    while ws[r] == 0 and r > 0:
      r -= 1
    hs = img.sum(axis=2).sum(axis=1)
    t = 0
    while hs[t] == 0 and t < len(hs):
      t += 1
    b = hs.shape[0] - 1
    while hs[b] == 0 and b > 0:
      b -= 1
    self.imgs[img_id] = self.imgs[img_id][t:b+1, l:r+1].copy()

  def project_3d_to_bird(self, pt):
    pt[0] += self.world_size / 2
    pt[1] = self.world_size - pt[1]
    pt[0] = pt[0] * 500 / self.world_size
    pt[1] = pt[1] * 500 / self.world_size
    return pt.astype(np.int32)

  def add_ct_detection(
    self, img, dets, show_box=False, show_txt=True, 
    center_thresh=0.5, img_id='det'):
    # dets: max_preds x 5
    self.imgs[img_id] = img.copy()
    if type(dets) == type({}):
      for cat in dets:
        for i in range(len(dets[cat])):
          if dets[cat][i, 2] > center_thresh:
            cl = (self.colors[cat, 0, 0]).tolist()
            ct = dets[cat][i, :2].astype(np.int32)
            if show_box:
              w, h = dets[cat][i, -2], dets[cat][i, -1]
              x, y = dets[cat][i, 0], dets[cat][i, 1]
              bbox = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2],
                              dtype=np.float32)
              self.add_coco_bbox(
                bbox, cat - 1, dets[cat][i, 2], 
                show_txt=show_txt, img_id=img_id)
    else:
      for i in range(len(dets)):
        if dets[i, 2] > center_thresh:
          cat = int(dets[i, -1])
          cl = (self.colors[cat, 0, 0] if self.theme == 'black' else \
                                       255 - self.colors[cat, 0, 0]).tolist()
          ct = dets[i, :2].astype(np.int32) * self.down_ratio
          cv2.circle(self.imgs[img_id], (ct[0], ct[1]), 3, cl, -1)
          if show_box:
            w, h = dets[i, -3] * self.down_ratio, dets[i, -2] * self.down_ratio
            x, y = dets[i, 0] * self.down_ratio, dets[i, 1] * self.down_ratio
            bbox = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2],
                            dtype=np.float32)
            self.add_coco_bbox(bbox, dets[i, -1], dets[i, 2], img_id=img_id)


  def add_3d_detection(
    self, image_or_path, dets, calib, show_txt=False, 
    center_thresh=0.5, img_id='det'):
    if isinstance(image_or_path, np.ndarray):
      self.imgs[img_id] = image_or_path
    else: 
      self.imgs[img_id] = cv2.imread(image_or_path)
    for cat in dets:
      for i in range(len(dets[cat])):
        cl = (self.colors[cat - 1, 0, 0]).tolist()
        if dets[cat][i, -1] > center_thresh:
          dim = dets[cat][i, 5:8]
          loc  = dets[cat][i, 8:11]
          rot_y = dets[cat][i, 11]
          # loc[1] = loc[1] - dim[0] / 2 + dim[0] / 2 / self.dim_scale
          # dim = dim / self.dim_scale
          if loc[2] > 1:
            box_3d = compute_box_3d(dim, loc, rot_y)
            box_2d = project_to_image(box_3d, calib)
            self.imgs[img_id] = draw_box_3d(self.imgs[img_id], box_2d, cl)

  def compose_vis_add(
    self, img_path, dets, calib,
    center_thresh, pred, bev, img_id='out'):
    self.imgs[img_id] = cv2.imread(img_path)

    h, w = pred.shape[:2]
    hs, ws = self.imgs[img_id].shape[0] / h, self.imgs[img_id].shape[1] / w
    self.imgs[img_id] = cv2.resize(self.imgs[img_id], (w, h))
    self.add_blend_img(self.imgs[img_id], pred, img_id)
    for cat in dets:
      for i in range(len(dets[cat])):
        cl = (self.colors[cat - 1, 0, 0]).tolist()
        if dets[cat][i, -1] > center_thresh:
          dim = dets[cat][i, 5:8]
          loc  = dets[cat][i, 8:11]
          rot_y = dets[cat][i, 11]
          if loc[2] > 1:
            box_3d = compute_box_3d(dim, loc, rot_y)
            box_2d = project_to_image(box_3d, calib)
            box_2d[:, 0] /= hs
            box_2d[:, 1] /= ws
            self.imgs[img_id] = draw_box_3d(self.imgs[img_id], box_2d, cl)
    self.imgs[img_id] = np.concatenate(
      [self.imgs[img_id], self.imgs[bev]], axis=1)

  def add_2d_detection(
    self, img, dets, show_box=False, show_txt=True, 
    center_thresh=0.5, img_id='det'):
    self.imgs[img_id] = img
    for cat in dets:
      for i in range(len(dets[cat])):
        cl = (self.colors[cat - 1, 0, 0]).tolist()
        if dets[cat][i, -1] > center_thresh:
          bbox = dets[cat][i, 1:5]
          self.add_coco_bbox(
            bbox, cat - 1, dets[cat][i, -1], 
            show_txt=show_txt, img_id=img_id)

  def add_coco_bev(self, cat,bbox, img_id='bird'):    
    box = bbox[:5]
    # box中心点
    ct_x, ct_y = int((box[0]+box[2])//2),int((box[1]+box[3])//2)
    # box相对ego的位置
    ego_box_x = (-960+ct_x)//60
    distance = bbox[-1]
    cv2.circle(self.imgs[img_id],(250,500),2,(255,0,0),-1)
    # 物体bev位置
    bird_loc = self.project_3d_to_bird(np.array([ego_box_x,distance]))    
    cv2.circle(self.imgs[img_id], (bird_loc[0], bird_loc[1]), 2, (0,0,255), -1)
    # 物体label
    cat = int(cat)
    txt = '{}'.format(self.names[cat])
    txt2 = '{}m'.format(str(int(bbox[-1])))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(self.imgs[img_id],txt, (bird_loc[0], bird_loc[1] - 2), 
                  font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(self.imgs[img_id],txt2, (bird_loc[0], bird_loc[1] -12), 
                  font, 0.4, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    cv2.line(self.imgs[img_id], (250,500), (bird_loc[0], bird_loc[1]), (0,255,0), 1,lineType=cv2.LINE_AA)

    


  def add_bird_view(self, dets, center_thresh=0.3, img_id='bird'):
    bird_view = np.ones((self.out_size, self.out_size, 3), dtype=np.uint8) * 230
    for cat in dets:
      cl = (self.colors[cat - 1, 0, 0]).tolist()
      lc = (250, 152, 12)
      for i in range(len(dets[cat])):
        if dets[cat][i, -1] > center_thresh:
          dim = dets[cat][i, 5:8]
          loc  = dets[cat][i, 8:11]
          rot_y = dets[cat][i, 11]
          rect = compute_box_3d(dim, loc, rot_y)[:4, [0, 2]]
          for k in range(4):
            rect[k] = self.project_3d_to_bird(rect[k])
            # cv2.circle(bird_view, (rect[k][0], rect[k][1]), 2, lc, -1)
          cv2.polylines(
              bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)],
              True,lc,2,lineType=cv2.LINE_AA)
          for e in [[0, 1]]:
            t = 4 if e == [0, 1] else 1
            cv2.line(bird_view, (rect[e[0]][0], rect[e[0]][1]),
                    (rect[e[1]][0], rect[e[1]][1]), lc, t,
                    lineType=cv2.LINE_AA)
    self.imgs[img_id] = bird_view

  def add_bird_views(self, dets_dt, dets_gt, center_thresh=0.3, img_id='bird'):
    alpha = 0.5
    bird_view = np.ones((self.out_size, self.out_size, 3), dtype=np.uint8) * 230
    for ii, (dets, lc, cc) in enumerate(
      [(dets_gt, (12, 49, 250), (0, 0, 255)), 
       (dets_dt, (250, 152, 12), (255, 0, 0))]):
      # cc = np.array(lc, dtype=np.uint8).reshape(1, 1, 3)
      for cat in dets:
        cl = (self.colors[cat - 1, 0, 0]).tolist()
        for i in range(len(dets[cat])):
          if dets[cat][i, -1] > center_thresh:
            dim = dets[cat][i, 5:8]
            loc  = dets[cat][i, 8:11]
            rot_y = dets[cat][i, 11]
            rect = compute_box_3d(dim, loc, rot_y)[:4, [0, 2]]
            for k in range(4):
              rect[k] = self.project_3d_to_bird(rect[k])
            if ii == 0:
              cv2.fillPoly(
                bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)],
                lc,lineType=cv2.LINE_AA)
            else:
              cv2.polylines(
                bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)],
                True,lc,2,lineType=cv2.LINE_AA)
            # for e in [[0, 1], [1, 2], [2, 3], [3, 0]]:
            for e in [[0, 1]]:
              t = 4 if e == [0, 1] else 1
              cv2.line(bird_view, (rect[e[0]][0], rect[e[0]][1]),
                      (rect[e[1]][0], rect[e[1]][1]), lc, t,
                      lineType=cv2.LINE_AA)
    self.imgs[img_id] = bird_view


kitti_class_name = [
  'p', 'v', 'b'
]

gta_class_name = [
  'p', 'v'
]

pascal_class_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
  "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
  "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

coco_class_name = [
     'person', 'bicycle', 'car', 'motorcycle', 'airplane',
     'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
     'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
     'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
     'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
     'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
     'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

color_list = np.array(
        [
            1.000, 1.000, 1.000,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255
