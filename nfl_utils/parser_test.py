
import os
import yaml

import argparse

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

group = parser.add_argument_group('Model parameters')
group.add_argument("--model", help="timm model_type", default="tf_efficientnet_b0")
group.add_argument("--inp_channels", help="use channels if RGB, set 3", default=3)
group.add_argument("--num_img_feature", help="timm model output feature num from back bone model", default=5)
group.add_argument("--pretrained", help="if use timm pretrained model, set True", default=True)

group = group.add_argument_group('Training parameters')
group.add_argument("--n_epoch", help="training epoch num", default=10)
group.add_argument("--n_folds", help="cross validation fold num", default=3)
group.add_argument("--train_folds", help="use folds list", default=[0, 1, 2])
group.add_argument("--lr", default=1e-4)
group.add_argument("--T_max", default=10)
group.add_argument("--min_lr", default=1e-8)
group.add_argument("--weight_decay", default=1e-6)
group.add_argument("--print_freq", help="training situation print freq", default=1000)

group = group.add_argument_group('Dataset parameters')
group.add_argument("--batch_size", help="train/valid batchsize", default=128)
group.add_argument("--num_workers", default=2)

group = group.add_argument_group('Experiment Directory')
parser.add_argument("--EXP_NAME", help="experiment name", default="DEBUG")
parser.add_argument("--kaggle", help="if kaggle set True, else False", default="False")

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


if __name__=="__main__":
    args, args_text = _parse_args()
    CFG = vars(args)# convert to dict
    print(CFG)