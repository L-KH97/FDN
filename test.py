import argparse, os

import torch

from src.util.config_parse import ConfigParser
from src.trainer import get_trainer_class


def main():
    # parsing configuration
    args = argparse.ArgumentParser()
    args.add_argument('-s', '--session_name', default=None,  type=str)
    args.add_argument('-c', '--conf',       default='',  type=str)
    args.add_argument('-e', '--ckpt_epoch',   default=20,     type=int)
    args.add_argument('-g', '--gpu',          default='0',  type=str)
    args.add_argument(      '--pretrained',   default='',  type=str)
    args.add_argument(      '--thread',       default=4,     type=int)
    args.add_argument(      '--self_en',      action='store_true')
    args.add_argument(      '--test_img',     default=None,  type=str)
    args.add_argument(      '--test_dir',     default=None,  type=str)

    args = args.parse_args()

    assert args.conf is not None, 'config file path is needed'
    if args.session_name is None:
        args.session_name = args.conf # set session name to config file name

    cfg = ConfigParser(args)

    # device setting
    if cfg['gpu'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']

    # intialize trainer
    trainer = get_trainer_class(cfg['trainer'])(cfg)

    # test
    trainer._before_test(dataset_load=False)
    trainer.test_img(args.test_img)


if __name__ == '__main__':
    main()
