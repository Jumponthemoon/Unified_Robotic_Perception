import os
import yaml
import logging

class CONSTANT(object):
    def __init__(self, special_keys=("Image", "Global", "Camera", "Train")):
        self.special_keys = special_keys
        self.registered_tasks = {}
        self.tasks_config = {}

    def init_from_scratch(self, config, args=None):
        self._init_from_scratch(config, args)

    def init_from_path(self, args=None):
        if args.config:
            assert (
                args.save
            ), "If you override the config, that indicate you want to just load weight as init,\
                            please then indicate the new save path by passing -s [PATH]"
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)
        else:
            logging.warning(
                "Config will get overloaded by the config in {0}/config.yaml".format(args.init)
            )
            
            #with open(args.init + "20210721_pred_kp_vis.yaml", "r") as f:
            with open(args.init + "/config.yaml", "r") as f:    
                config = yaml.safe_load(f)

        # Setup the CONSTANT() singleton
        self._init_from_scratch(config, args)
        return config

    def _init_default(self):
        """This function can be used to store default values for backward compatibility.
        it will be called at the beginning of _init_from_scratch.
        similar to reset(), self.registered_tasks and self.tasks_config will be cleaned once this function is called.
        Examples:
            self.default_a = a
        """
        self.registered_tasks = {}
        self.tasks_config = {}
        self.max_data_fetch_iteration = 200

    def _init_from_scratch(self, config, args=None):
        self._init_default()
        for key, val in config.items():
            if key in self.special_keys:
                for sub_key, sub_val in val.items():
                    setattr(self, sub_key, sub_val)
            else:
                setattr(self, key, val)

        self.gpus_str = self.gpus
        self.gpus = [int(gpu) for gpu in self.gpus.split(',')]
        self.gpus = [i for i in range(len(self.gpus))] if self.gpus[0] >= 0 else [-1]
        self.lr = float(self.lr)
        self.lr_step = [int(i) for i in self.lr_step.split(',')]
        self.ds_sample_freq = [float(i) for i in self.ds_sample_freq.split(',')]
        self.test_scales = [float(i) for i in self.test_scales.split(',')]

        self.fix_res = not self.keep_res
        print('Fix size testing.' if self.fix_res else 'Keep resolution testing.')
        self.reg_offset = not self.not_reg_offset
        self.reg_bbox = not self.not_reg_bbox
        self.hm_hp = not self.not_hm_hp
        self.reg_hp_offset = (not self.not_reg_hp_offset) and self.hm_hp

        if self.head_conv == -1:  # init default head_conv
            self.head_conv = 256 if 'dla' in self.arch else 64
        self.pad = 127 if 'hourglass' in self.arch else 31
        print(self.padding)
        self.padding = [self.padding for i in range(3)]
        self.num_stacks = 2 if self.arch == 'hourglass' else 1
        self.cbr_config = {name:(cbr_num, last_k) for name, cbr_num, last_k in zip(self.head_names, self.cbr_num, self.last_k)}

        if self.trainval:
            self.val_intervals = 100000000

        if self.debug > 0:
            self.num_workers = 0
            self.batch_size = 1
            self.gpus = [self.gpus[0]]
            self.master_batch_size = -1

        if self.master_batch_size == -1:
            self.master_batch_size = self.batch_size // len(self.gpus)
        rest_batch_size = (self.batch_size - self.master_batch_size)
        self.chunk_sizes = [self.master_batch_size]
        for i in range(len(self.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(self.gpus) - 1)
            if i < rest_batch_size % (len(self.gpus) - 1):
                slave_chunk_size += 1
            self.chunk_sizes.append(slave_chunk_size)
        print('training chunk_sizes:', self.chunk_sizes)

        self.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        if self.data_dir:
            pass
        elif self.coco_data_dir:
            self.data_dir = self.coco_data_dir
        else:
            self.data_dir = os.path.join(self.root_dir, 'data')
        self.exp_dir = os.path.join(self.root_dir, 'exp', self.task)

        # Model save directory if args.save is specified, default is args.init
        self.save_dir = args.save
        if not self.save_dir:
            self.save_dir = args.init
        self.resume = True if (args.init and not args.config) else False
        if self.resume:
            model_path = self.save_dir[:-4] if self.save_dir.endswith('TEST') \
                else self.save_dir
            self.load_model = os.path.join(model_path, 'model_last.pth')
        # load from a pretrained model from -i if -c, -s and -i are specified
        if args.init and args.config and args.save:
            self.load_model = os.path.join(args.init, 'model_last.pth')

        self.debug_dir = os.path.join(self.save_dir, 'debug')
        print('The output will be saved to ', self.save_dir)

        os.makedirs(self.save_dir, exist_ok=True)
        # Model initial directory
        self.init_path = args.init

        self.accu_grad = config.get('accu_grad', False)
        self.optimizer = config.get('optimizer', 'Adam')
        self.grad_norm = config.get('grad_norm', False)
        self.grad_bounds = config.get('grad_bounds', {})
        if len(self.grad_bounds) == 0:
            self.grad_bounds.update({'pose': 2.0, 'reid': 1.0})

        self.loss_weight = {'hm_loss': self.hm_weight,
                            'hp_loss': self.hp_weight,
                            'hm_hp_loss': self.hm_hp_weight,
                            'hp_offset_loss': self.off_weight,
                            'wh_loss': self.wh_weight,
                            'off_loss': self.off_weight,
                            'id_loss': 1.
                            }

        self.joint_idx ={'nose': 0, 'left_eye':1, 'right_eye':2,
                         'left_ear':3, 'right_ear':4, 'left_shoulder':5,
                         'right_shoulder':6, 'left_elbow':7, 'right_elbow':8,
                         'left_wrist':9, 'right_wrist':10, 'left_hip':11,
                         'right_hip':12, 'left_knee':13, 'right_knee':14,
                         'left_ankle':15, 'right_ankle':16}
        self.pred_vis_joint_names = config.get('pred_kp_vis', [])
        self.pred_vis_joint_idx = [self.joint_idx[i] for i in self.pred_vis_joint_names]
        self.kp_vis_weight = config.get('kp_vis_weight', 1.)

        # for inter-tasks process purppose
        self.config = config.copy()

    def update_dataset_info_and_set_heads(self, opt, dataset):
        input_h, input_w = dataset.default_resolution
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes

        # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w
        opt.output_h = opt.input_h // opt.down_ratio
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)

        if opt.task == 'exdet':
            # assert opt.dataset in ['coco']
            num_hm = 1 if opt.agnostic_ex else opt.num_classes
            opt.heads = {'hm_t': num_hm, 'hm_l': num_hm,
                         'hm_b': num_hm, 'hm_r': num_hm,
                         'hm_c': opt.num_classes}
            if opt.reg_offset:
                opt.heads.update({'reg_t': 2, 'reg_l': 2, 'reg_b': 2, 'reg_r': 2})
        elif opt.task == 'ddd':
            # assert opt.dataset in ['gta', 'kitti', 'viper']
            opt.heads = {'hm': opt.num_classes, 'dep': 1, 'rot': 8, 'dim': 3}
            if opt.reg_bbox:
                opt.heads.update(
                    {'wh': 2})
            if opt.reg_offset:
                opt.heads.update({'reg': 2})
        elif opt.task == 'ctdet':
            # assert opt.dataset in ['pascal', 'coco']
            opt.heads = {'hm': opt.num_classes,
                         'wh': 2 if not opt.cat_spec_wh else 2 * opt.num_classes}
            if opt.reg_offset:
                opt.heads.update({'reg': 2})
        elif opt.task == 'multi_pose':
            # assert opt.dataset in ['coco_hp']
            opt.flip_idx = dataset.flip_idx
            opt.heads = {'hm': opt.num_classes, 'wh': 2, 'hps': 34}
            if opt.reg_offset:
                opt.heads.update({'reg': 2})
            if opt.hm_hp:
                opt.heads.update({'hm_hp': 17})
            if opt.reg_hp_offset:
                opt.heads.update({'hp_offset': 2})
        elif opt.task == 'mot':
            opt.heads = {'hm': opt.num_classes,
                         'wh': 4 if opt.ltrb else 2,
                         'id': opt.reid_dim}
            if opt.reg_offset:
                opt.heads.update({'reg': 2})
            opt.nID = dataset.nID
            opt.heads.update({'reid': (opt.reid_dim, opt.nID)})
        elif opt.task == 'multi_pose_mot':
            # assert opt.dataset in ['coco_hp']
            opt.flip_idx = dataset.flip_idx
            opt.heads = {'hm': opt.num_classes, 'wh': 4, 'hps': 34}
            opt.heads.update({'wh': 4}) if opt.ltrb else opt.heads.update({'wh': 2})
            if opt.reg_offset:
                opt.heads.update({'reg': 2})
            if opt.hm_hp:
                opt.heads.update({'hm_hp': 17})
            if opt.reg_hp_offset:
                opt.heads.update({'hp_offset': 2})
            # opt.heads.update({'id': opt.reid_dim})
            if getattr(opt, 'combine_reid_to_wh', False):
                opt.heads['wh'] = opt.heads['wh'] + opt.reid_dim
            else:
                opt.heads.update({'id': opt.reid_dim})
        else:
            assert 0, 'task not defined!'
        print('heads', opt.heads)
        return opt

    def init_head_config(self):
        default_dataset_info = {
            'ctdet': {'default_resolution': [512, 512], 'num_classes': 80,
                      'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                      'dataset': 'coco'},
            'exdet': {'default_resolution': [512, 512], 'num_classes': 80,
                      'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                      'dataset': 'coco'},
            'multi_pose': {
                'default_resolution': [512, 512], 'num_classes': 1,
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                # 'mean': [0.0, 0.0, 0.0], 'std': [1., 1., 1.],
                'dataset': 'coco_hp', 'num_joints': 17,
                'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                             [11, 12], [13, 14], [15, 16]]},
            'ddd': {'default_resolution': [384, 1280], 'num_classes': 3,
                    'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                    'dataset': 'kitti'},
            'mot': {'default_resolution': [608, 1088], 'num_classes': 1,
                    'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                    'dataset': 'jde', 'nID': 14455},
            'multi_pose_mot': {
                'default_resolution': [512, 512], 'num_classes': 1,
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'coco_hp', 'num_joints': 17,
                'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                             [11, 12], [13, 14], [15, 16]]},
        }

        class Struct:
            def __init__(self, entries):
                for k, v in entries.items():
                    self.__setattr__(k, v)

        dataset = Struct(default_dataset_info[self.task])
        self.dataset = dataset.dataset
        opt = self.update_dataset_info_and_set_heads(self, dataset)
        return opt

    def __str__(self):
        return "\n".join(self.__dict__.keys())
