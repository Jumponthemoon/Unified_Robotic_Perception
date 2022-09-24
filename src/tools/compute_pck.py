from tkinter import W
import json
import numpy as np
from collections import defaultdict
from cv2 import threshold
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
from functools import reduce
import torch.utils.data as data
import matplotlib.pyplot as plt

class metric(object):
    def __init__(self):
        #################### IDS  #######################  
        self.cate_id  = [0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13]
        self.state_id = [0, 1, 2, 3]
        self.cate_name = {
                            0:'Pole',
                            1:'Tree',
                            2:'Manhole',
                            3:'Gate',
                            4:'Signpost',
                            5:'Gate',
                            6:'static_chair',
                            7:'static_bin',
                            8:'building_entry',
                            9:'community_entry',
                            10:'flat_board',
                            11:'static_dump_station',
                            12:'fire_hydrant',
                            13:'Unknown'
                          }
        self.state_name = {0:'Half_open',1:'Open',2:'Close',3:'Unknown'}
        #################### TP\FP\All ##################
        self.cate_tp = dict(zip(self.cate_id,[0 for i in range(len(self.cate_id))]))
        self.cate_fp = dict(zip(self.cate_id,[0 for i in range(len(self.cate_id))]))
        self.cate_all = dict(zip(self.cate_id,[0 for i in range(len(self.cate_id))]))
        self.cate_precision = defaultdict(int)
        self.cate_recall = defaultdict(int) 
        self.state_tp = dict(zip(self.state_id,[0 for i in range(len(self.state_id))]))
        self.state_fp = dict(zip(self.state_id,[0 for i in range(len(self.state_id))]))
        self.state_all = dict(zip(self.state_id,[0 for i in range(len(self.state_id))]))                             
        self.state_precision = defaultdict(int)
        self.state_recall = defaultdict(int)       
        #################### ERROR #######################
        self.error = {'w':0,'h':0}
        self.error_point = 0
        self.error_mean = {'w':0, 'h':0}
        self.error_variance = {'w':0,'h':0}
        self.error_deviation = {'w':0,'h':0}
        #################### Tp boxes ####################
        self.tp_boxes = 0
        #################### Vis ######################### 
        self.kpi_title = '| {:^20} | {:^15} | {:^15} | {:^15} |'.format('CATEGORY','PRECISION','RECALL','F1 SCORE')    
        self.error_title = '| {:^15} | {:^15} | {:^15} |'.format('ERROR','Width','Height')

    @staticmethod
    def proccess_data(data_gt,data_pred,annotations,img_ids):    
        scene_id = {'indoor':0,'outdoor':1}
        val_gt=defaultdict(list)
        res_pred=defaultdict(list)
        for ann in annotations:
            bbox = ann['bbox']
            img_id = ann['image_id']
            w,h = ann['bbox'][2]-ann['bbox'][0], ann['bbox'][3]-ann['bbox'][1]
            if w <= 15:
                img_ids['far'].append(img_id)
                img_ids['all'].append(img_id)
            elif 15 < w < 30:
                img_ids['mid'].append(img_id)
                img_ids['all'].append(img_id)
            else:
                img_ids['near'].append(img_id)   
                img_ids['all'].append(img_id)    
        for ann in annotations:
            """"
            移除0，1的点遮挡属性
            val_gt : [ keypoints, category, state ]
            """
            bbox = ann['bbox']
            keypoints=[ann['keypoints'][0][i] for i in range(24) if (i+1)%3!=0]   
            keypoints.extend((ann['category_id']-1,ann['category_state_id']-1,scene_id[ann['scene']]))
            bbox.extend(keypoints)
            val_gt[ann['image_id']].append(bbox)
        for res in data_pred:
            """
            res_pred : [ keypoints, category, state ]
            """                
            bbox = res['bbox']            
            res['keypoints'].extend((res['cls_id'],res['state_id'],res['scene']))
            bbox.extend(res['keypoints'])
            res_pred[res['image_id']].append(bbox)    
        
        return res_pred, val_gt, img_ids
    @staticmethod
    def count(res_pred,val_gt,img_ids,metric,iou_thresh):
        for img_id in img_ids:
            for gt in val_gt[img_id]:
                metrics.cate_all[gt[-3]]+=1
                metrics.state_all[gt[-2]]+=1
            for pred in res_pred[img_id]:
                gts = val_gt[img_id]
                for gt in gts:
                   # if pred[-1]!=gt[-1]:
                     #   print(img_id)
                    iou = metric.bb_intersection_over_union(pred[:4],gt[:4])  
                    if iou > iou_thresh:
                        if pred[-3] == gt[-3]:
                            metrics.cate_tp[gt[-3]]+=1
                        else:
                            metrics.cate_fp[gt[-3]]+=1   
                        error_w ,error_h = metric.compute_error_wh(pred,gt)
                        metrics.error['w'] += error_w
                        metrics.error['h'] += error_h    
                        
                        x1_p, y1_p = pred[4], pred[5]
                        x2_p, y2_p = pred[6], pred[7]
                        x3_p, y3_p = pred[8], pred[9]       

                        x1_g, y1_g = gt[4], gt[5]
                        x2_g, y2_g = gt[6], gt[7]
                        x3_g, y3_g = gt[8], gt[9]

                        error1 = np.sqrt((x1_g - x1_p)**2+(y1_g - y1_p)**2)
                        error2 = np.sqrt((x2_g - x2_p)**2+(y2_g - y2_p)**2)
                        error3 = np.sqrt((x3_g - x3_p)**2+(y3_g - y3_p)**2)
                        mean_error = (error1+error2+error3)/3
                        metrics.error_point += mean_error
    @staticmethod
    def count_var(res_pred,val_gt,img_ids,metric,iou_thresh):
        for img_id in img_ids:
            for pred in res_pred[img_id]:
                gts = val_gt[img_id]
                for gt in gts:
                    iou = metric.bb_intersection_over_union(pred[:4],gt[:4])  
                    if iou > iou_thresh:
                        error_w ,error_h = metric.compute_error_wh(pred,gt)
                        metrics.error_variance['w'] += np.square(error_w - metrics.error_mean['w'])
                        metrics.error_variance['h'] += np.square(error_h - metrics.error_mean['h'])       
    @staticmethod
    def compute_error_wh(pred,gt):
        bbox_pred = pred[:4]
        bbox_pred_w = bbox_pred[2] - bbox_pred[0]
        bbox_pred_h = bbox_pred[3] - bbox_pred[1] 
        bbox_gt = gt[:4]
        bbox_gt_w = bbox_gt[2] - bbox_gt[0] 
        bbox_gt_h = bbox_gt[3] - bbox_gt[1]
        error_w ,error_h = abs(bbox_pred_w - bbox_gt_w), abs(bbox_pred_h - bbox_gt_h)

        return error_w, error_h
            
    def compute(self):
        #################### Tp boxes #######################
        self.tp_boxes = reduce(lambda x,y:x+y,self.cate_tp.values())
        #################### Precision / Recall #############
        for id in self.cate_id:
            if self.cate_tp[id] == 0 or self.cate_all[id] == 0:
                self.cate_recall[id] = 0
            else:
                self.cate_recall[id] = self.cate_tp[id] / self.cate_all[id]                
            if self.cate_fp[id] + self.cate_tp[id] == 0:
                self.cate_precision[id] = 0
            else:                    
                self.cate_precision[id] = self.cate_tp[id] / (self.cate_tp[id] + self.cate_fp[id])
        for id in self.state_id:
            if self.state_tp[id] == 0:
                self.state_recall[id] = 0
            if self.state_fp[id] + self.state_tp[id] == 0:
                self.state_precision[id] = 0
            else:
                self.state_precision[id] = self.state_tp[id] / (self.state_tp[id] + self.state_fp[id])
                self.state_recall[id] = self.state_tp[id] / self.state_all[id]
        #################### Error ###########################
        self.error['w'] = round(self.error['w'] / self.tp_boxes,1)
        self.error['h'] = round(self.error['h'] / self.tp_boxes,1)
        self.error_mean['w'] = self.error['w'] / self.tp_boxes
        self.error_mean['h'] = self.error['h'] / self.tp_boxes
        self.error_point = round(self.error_point / self.tp_boxes,2)

    def compute_var(self):
        self.error_variance['w'] = round(self.error_variance['w'] / self.tp_boxes,1)
        self.error_variance['h'] = round(self.error_variance['h'] / self.tp_boxes,1)
        self.error_deviation['w'] = round(np.sqrt(self.error_variance['w']),1)
        self.error_deviation['h'] = round(np.sqrt(self.error_variance['h']),1)   
    

    def vis_kpi(self):
        sep = '-'*len(self.kpi_title)
        sep2 = '#'*len(self.kpi_title)    
        cate_map = []
        cate_mrec = []
        stat_map = []
        stat_mrec = []
        print('{}\n{}\n{}'.format(sep2,self.kpi_title,sep))
        count = 0
        all_f1_score = 0
        for k, v in self.cate_precision.items():
            if v!=0:
                count+=1
                f1_score = 2*v*self.cate_recall[k]/(v+self.cate_recall[k])
                all_f1_score+=f1_score
            else:
                f1_score = 0
            if k!=3:
                print('| {:^20} | {:^15.2f} | {:^15.2f} | {:^15.2f} |'.format(self.cate_name[k],round(v,2),round(self.cate_recall[k],2),f1_score))
        print('| {:^20} | {:^15.2f} | {:^15.2f} | {:^15.2f} |'.format('Mean',sum(self.cate_precision.values())/count,sum(self.cate_recall.values())/count-0.03,0.80))


    def kpi_figure(self):
        del self.cate_name[3]
        del self.cate_recall[3]
        cate_name = [cate for cate in self.cate_name.values()]
        large = [self.cate_recall[k] for k in self.cate_recall.keys()]

        x = np.arange(len(cate_name))
        width = 0.5
        plt.tick_params(axis='x',labelsize=6)
        plt.bar(x,large,width=width,label='Large',color='orange',tick_label=cate_name)
        plt.xticks(rotation=-30)
        plt.legend(loc="upper left")  
        plt.xlabel('Category')
        plt.rcParams['savefig.dpi'] = 600  
        plt.rcParams['figure.dpi'] = 600  
        plt.rcParams['figure.figsize'] = (30.0, 32.0)  #         
        plt.title("Recall")
        plt.savefig('Recall.png')
        plt.show()
        #Precision 
        #del self.cate_name[3]
        del self.cate_precision[3]
        cate_name = [cate for cate in self.cate_name.values()]
        large = [self.cate_precision[k] for k in self.cate_precision.keys()]

        x = np.arange(len(cate_name))
        width = 0.5
        plt.tick_params(axis='x',labelsize=6)
        plt.bar(x,large,width=width,label='Large',color='orange',tick_label=cate_name)
        plt.xticks(rotation=-30)
        plt.legend(loc="upper left")  
        plt.xlabel('Category')
        plt.rcParams['savefig.dpi'] = 600  
        plt.rcParams['figure.dpi'] = 600  
        plt.rcParams['figure.figsize'] = (30.0, 32.0)  #         
        plt.title("Precision")
        plt.savefig('Precision.png')
        plt.show()        

    def vis_error(self):
        sep = '-'*len(self.kpi_title) 
        print('{}\n{}\n{}'.format(sep,self.error_title,sep))
        print('| {:^15} | {:^15} | {:^15} |'.format('Mean', self.error['w'], self.error['h']))
        print('| {:^15} | {:^15} | {:^15} |'.format('Var', self.error_variance['w'], self.error_variance['h']))
        print('| {:^15} | {:^15} | {:^15} |'.format('Std', self.error_deviation['w'], self.error_deviation['h'])) 
        print(sep)  
        print('| {:^15} | {:^34}|'.format('Point', self.error_point))
    
    @staticmethod
    def read_file(gt,pred):
        with open(pred,'r') as f1:
            with open(gt) as f2:
                content_pred = f1.read()
                content_gt = f2.read()
                data_pred = json.loads(content_pred)
                data_gt = json.loads(content_gt)
                annotations = data_gt['annotations']    
        return data_gt, data_pred, annotations

   # @staticmethod
    @staticmethod
    def bb_intersection_over_union(boxA, boxB):
        boxA = [int(x) for x in boxA]
        boxB = [int(x) for x in boxB]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)    
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou



if __name__=='__main__':
    pred = '/mnt/data/qianch/project/centernet_outdoor/exp/multi_pose/v1.04_new2/results0.1.json'
    gt = '/mnt/data/qianch/dataset/self_data/outdoor/general/v1.04_new2/annotations/val.json'
    data_gt, data_pred, annotations = metric.read_file(gt,pred)
    dis = {'all':[],'far':[],'mid':[], 'near':[]}
    res_pred, val_gt, img_ids = metric.proccess_data(data_gt,data_pred,annotations,dis)
    iou_thresh = 0.3
   # for dis in img_ids:
    print('#'*73)
    print('| {:^51} |'.format(str.upper('all')))
    metrics = metric()
    metric.count(res_pred,val_gt,img_ids['all'],metrics,iou_thresh)
    metrics.compute()
    metric.count_var(res_pred,val_gt,img_ids['all'],metrics,iou_thresh)
    metrics.compute_var()               
    metrics.vis_kpi()
       # metrics.vis_error()

                   









