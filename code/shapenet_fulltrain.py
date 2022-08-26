import argparse
from experiments.transformations.transf_exp import cat_exp
from experiments.transformations.utils.misc import get_fulltrain_strings_shapenet
import os
import torch
from sty import fg, bg, ef, rs

name_exp_set = 'fulltrain'

# subfolders = os.path.dirname(globals().get("__file__")).split('code/experiments/')[1]
# model_folder = lambda: f'./models/{subfolders}/ShapeNet/{network_name}/{name_exp_set}'
# test_output_folder = lambda: f'./results/{subfolders}/ShapeNet/{network_name}/{name_exp_set}/'

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("-transform", "--transforms",
                    help='Choose between any combination of vrst e.g. [vf], [rs], [s]. Leave empty for no transformation (only cat)',
                    default='', type=str)
parser.add_argument("-exp_max_obj", "--exp_max_obj",
                    default=10, type=int)
parser.add_argument("-network_name", "--network_name",
                    default='vgg11bn', type=str)
parser.add_argument("-seed", "--seed",
                            default=1, type=int)
    

PARAMS = vars(parser.parse_known_args()[0])
exp_transf = PARAMS['transforms']
max_objs = PARAMS['exp_max_obj']
seed = PARAMS['seed']
network_name = PARAMS['network_name']


shared_args = lambda: dict(seed=seed, project_name='All-Transformations', use_weblog=0 if torch.cuda.is_available() else 0, batch_size=64, max_epochs=-1 if torch.cuda.is_available() else 3, patience_stagnation=-1 if '0' in exp_transf else 1000, freeze_backbone=freeze_backbone,  pretraining_backbone=ptbkbn, num_viewpoints_train=num_v, use_mat=mat, max_objs_per_class_train=max_objs)


def call_cat_exp():
    other_info, output_model, output_test, _ = get_fulltrain_strings_shapenet(exp_transf, max_objs, num_v, classes_set, 0, network_name, seed)
    exp = cat_exp(**shared_args(),
                  additional_tags=other_info,
                  model_output_filename=output_model,
                  output_filename=output_test,
                  classes_set=classes_set,
                  diff_classes_set='set2',
                  network_name=network_name,
                  transformations=exp_transf,
                  pretraining=pt)
    return exp, output_model, output_test


pt = 'vanilla'
ptbkbn = 'vanilla'
mat = 0
freeze_backbone = 0

num_v = 50 if 'v' in exp_transf else 1

classes_set = 'set1'
exp, output_model, _ = call_cat_exp()
print(ef.inverse + "Finished" + rs.inverse)
