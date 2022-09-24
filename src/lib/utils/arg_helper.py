import os
import shutil
import logging
import yaml
import argparse

def argument_parser():
    """This is the argument parser function used for train/validate/inference function"""
    parser = argparse.ArgumentParser(description='Commands for train/val/inference methods of XPModel')
    parser.add_argument('-c', '--config', help='Yaml file training config defined in configs/')
    parser.add_argument('-s', '--save', help='Path to save the model')
    parser.add_argument('-i', '--init', help='Initial from path...')
    parser.add_argument('-d', '--dir', help='Path containing videos or images for inference')
    parser.add_argument('--crop_box', action='store_true', help='to save cropped box')
    parser.add_argument('--load_off_reid', action='store_true', help='to load offline reid feature')
    parser.add_argument('--estimate_reid_dist', action='store_true', help='estimate reid distance distribution')
    parser.add_argument('--infer_type', default='video', choices=['video', 'image'], help='Path containing videos or images for inference')
    parser.add_argument('--undist', action='store_true', help='to undistort fisheye region')
    args = parser.parse_args()

    return args

def copy_config_and_args(global_config, config, args):
    # Move and Rename config
    try:
        shutil.copy(config, global_config.save_dir)
        os.rename(global_config.save_dir + '/' + config.split('/')[-1], global_config.save_dir + '/config.yaml')
    except Exception as e:
        logging.warning('Did not replace the config due to %s' % e)
    # Save argument to config as .yaml.
    args_dict = {arg: getattr(args, arg) for arg in dir(args) if '_' not in arg}
    with open(global_config.save_dir + '/' + 'args.yaml', 'w') as f:
        yaml.dump(args_dict, f)

    return


def init_constant(args):
    from constants import CONSTANT
    global_config = CONSTANT()
    # Load the config.
    if args.init:
        # Read from previous trained model
        _ = global_config.init_from_path(args=args)
        # Move and rename config to save path.
        config_path = args.config if args.config else args.init + '/config.yaml'
    else:
        # Read from scratch
        assert 'config' in dir(args), 'Please define the config path in command line'
        assert 'save' in dir(args), 'Please define the savinit_processe path in command line'
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        #print(config)
        #print(args)
        #print(global_config)
        _ = global_config.init_from_scratch(config, args)
        # Move and rename config to save path.
        config_path = args.config if args.config else args.init
    copy_config_and_args(global_config, config_path, args)

    return global_config
