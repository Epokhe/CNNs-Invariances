from callbacks import TriggerActionWithPatience
from torchvision.transforms.transforms import RandomAffine
import collections
from experiment import create_backbone_exp, SupervisedLearningExperiment
import torch
import torchvision
import numpy as np
import re
import torch.nn as nn

from experiments.transformations.utils.datasets import TransformationInfoDataFrameSaver
import pickle
import os
import pathlib
import framework_utils
from experiments.transformations.utils.net import self_supervised_step, SameDifferentNetwork
from train_net import standard_net_step
from sty import fg, rs

project_path = './code/experiments/transformations/'

n = './data/ShapeNet2D/id_to_name.pickle'
if not os.path.exists(n):
    try:
        from nltk.corpus import wordnet
        syns = list(wordnet.all_synsets())

    except LookupError:
        import nltk
        nltk.download('wordnet')
        from nltk.corpus import wordnet
        syns = list(wordnet.all_synsets())

    offsets_list = [(s.offset(), s) for s in syns]
    offsets_dict = dict(offsets_list)
    import nltk
    nltk.download('wordnet')
    d = {f'{k:08}': v.name() for k, v in offsets_dict.items()}
    pathlib.Path(os.path.dirname('./data/ShapeNet2D/')).mkdir(parents=True, exist_ok=True)
    pickle.dump(d, open('./data/ShapeNet2D/id_to_name.pickle', 'wb'))


def get_fulltrain_strings_shapenet(transform, objs, num_v, class_set, mat, network_name, seed):
    project_name = project_path.split("code/experiments/")[1]
    name_net = f'FULLTRAIN_T{transform}_objs{objs}_vp{num_v}_mat{mat}_tr{class_set}_S{seed}'

    output_model = f'./models/{project_name}/ShapeNet/{network_name}/fulltrain/{name_net}.pt'
    output_test = f'./results/{project_name}/ShapeNet/{network_name}/fulltrain/{name_net}_test.pickle'
    info_str = name_net + f'_{network_name}'
    return info_str, output_model, output_test, name_net


class TransformExperiment(create_backbone_exp(SupervisedLearningExperiment)):
    def __init__(self, **kwargs):
        self.freeze_all_but_last = None
        self.scramble_fc = None
        super().__init__(**kwargs)

    def parse_arguments(self, parser):
        super().parse_arguments(parser)
        parser.add_argument("-freeze_all_but_last", "--freeze_all_but_last",
                            help="",
                            type=lambda x: bool(int(x)),
                            default=False)
        parser.add_argument("-scramble_fc", "--scramble_fc",
                            help="0: nope. 1: scramble all the FC layers. 2: only scramble the very last one. ",
                            type=int,
                            default=0)

    def finalize_init(self, PARAMS, list_tags):
        self.freeze_all_but_last = PARAMS['freeze_all_but_last']
        self.scramble_fc = PARAMS['scramble_fc']
        list_tags.append(f'freeze_all-1') if self.freeze_all_but_last else None
        list_tags.append(f'scramble_fc{self.scramble_fc}') if self.scramble_fc != 0 else None

        super().finalize_init(PARAMS, list_tags)

    def get_net(self, new_num_classes=None):
        if self.network_name == 'vgg16':
            self.net = torchvision.models.vgg16(pretrained=False, progress=True, num_classes=new_num_classes)
        elif self.network_name == 'vgg16bn':
            self.net = torchvision.models.vgg16_bn(pretrained=False, progress=True, num_classes=new_num_classes)
        elif self.network_name == 'vgg11bn':
            self.net = torchvision.models.vgg11_bn(pretrained=False, progress=True, num_classes=new_num_classes)
        elif self.network_name == 'allconv_vgg11bn':
            self.net = torchvision.models.vgg11_bn(pretrained=False, progress=True, num_classes=new_num_classes)
            self.net.classifier = nn.Linear(25088, new_num_classes)
        elif self.network_name == 'allconvC':
            p = 0
            self.net = nn.Sequential(collections.OrderedDict([
                ('features', nn.Sequential(nn.Conv2d(3, 96, (3, 3), padding=p, bias=True), nn.ReLU(),
                                           nn.Conv2d(96, 96, (3, 3), padding=p, bias=True), nn.ReLU(),
                                           nn.Conv2d(96, 96, (3, 3), stride=2, padding=2, bias=True), nn.ReLU(),                                                                nn.Conv2d(96, 192, (3, 3), padding=p, bias=True), nn.ReLU(),
                                           nn.Conv2d(192, 192, (3, 3), padding=p, bias=True), nn.ReLU(),
                                           nn.Conv2d(192, 192, (3, 3), stride=2, padding=3, bias=True), nn.ReLU(),
                                           nn.Conv2d(192, 192, (3, 3), padding=p, bias=True), nn.ReLU(),
                                           nn.Conv2d(192, 192, (1, 1), padding=p, bias=True), nn.ReLU(),
                                           nn.Conv2d(192, 10, (1, 1), padding=p, bias=True), nn.ReLU())),
                ('avgpool',  nn.AvgPool2d(kernel_size=6)),
                ('classifier', nn.Sequential(nn.Flatten(),
                                             nn.Linear(10, 10)))]))
        elif self.network_name == 'vgg19bn':
            self.net = torchvision.models.vgg19_bn(pretrained=False, progress=True, num_classes=new_num_classes)
        elif self.network_name == 'resnet18':
            self.net = torchvision.models.resnet18(pretrained=False, progress=True, num_classes=new_num_classes)
        elif self.network_name == 'resnet50':
            self.net = torchvision.models.resnet50(pretrained=False, progress=True, num_classes=new_num_classes)
        elif self.network_name == 'alexnet':
            self.net = torchvision.models.alexnet(pretrained=False, progress=True, num_classes=new_num_classes)
        elif self.network_name == 'inception_v3':  # nope
            self.net = torchvision.models.inception_v3(pretrained=False, progress=True, num_classes=new_num_classes)
        elif self.network_name == 'densenet201':
            self.net = torchvision.models.densenet201(pretrained=False, progress=True, num_classes=new_num_classes)

        elif self.network_name == 'googlenet':
            self.net = torchvision.models.googlenet(pretrained=False, progress=True, num_classes=new_num_classes)
        elif self.network_name == 'samediffnet':  # when this is selected, we run the Same/Different task
            net = torchvision.models.vgg11_bn(pretrained=False, progress=True, num_classes=1)
            for m in net.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = 0
            net.avgpool = nn.Identity()
            net.classifier[0] = nn.Linear(8192, net.classifier[0].out_features, bias=True)
            net.classifier = nn.Sequential(*list(net.classifier.children())[:-3])
            net.classifier[3] = nn.Sequential(nn.Linear(4096, 512, bias=True), nn.Sigmoid())
            self.net = SameDifferentNetwork(net)
        elif self.network_name == 'samediffnetLineval':
            net = torchvision.models.vgg11_bn(pretrained=False, progress=True, num_classes=1)
            net.avgpool = nn.Identity()
            net.classifier[0] = nn.Linear(8192, net.classifier[0].out_features, bias=True)
            net.classifier = nn.Sequential(*list(net.classifier.children())[:-3])
            net.classifier[3] = nn.Linear(4096, 512, bias=True)
            self.net = nn.Sequential(collections.OrderedDict([
                ('backbone', net),
                ('classifier', nn.Sequential(nn.Linear(512, 512),
                                             nn.ReLU(),
                                             nn.Linear(512, 256),
                                             nn.ReLU(),
                                             nn.Linear(256, new_num_classes)))]))
            self.step = standard_net_step
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            assert False, f"Network name {self.network_name} not recognized"

        if self.network_name == 'samediffnet':
            self.step = self_supervised_step
            self.loss_fn = torch.nn.MSELoss()
        else:
            self.step = standard_net_step
            self.loss_fn = torch.nn.CrossEntropyLoss()

        if self.freeze_all_but_last:
            for param in self.net.parameters():
                param.requires_grad = False
            self.net.classifier[-1].requires_grad_(True)

        self.net, _ = self.finish_get_net()


        if self.scramble_fc == 1:
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)

        return self.net

    def backbone(self):
        if self.network_name == 'samediffnet' or self.network_name == 'samediffnetLineval':
            return self.net.backbone
        elif 'vgg' in self.network_name:
            return self.net.features
        else:
            return None

    def prepare_test_callbacks(self, *args, **kwargs):
        all_cb = super().prepare_test_callbacks(*args, **kwargs,
                                                dataframe_saver=TransformationInfoDataFrameSaver)
        return all_cb

    def prepare_train_callbacks(self, log_text, train_loader, test_loaders=None):
        all_cb = super().prepare_train_callbacks(log_text, train_loader, test_loaders=test_loaders)
        # This modality never ended up in the paper. It's a gradual transformation training, and it seems to improve
        # generalization in case of difficult training setups (e.g. vtrsbc) but we didn't use that condition anymore
        if '0' in self.transformations:
            class CheckStoppingLevel:
                there_is_another_level = True
                max_steps = 10
                current_step = 0

                def __init__(self, dataloader, transformation, weblogger):
                    self.dataloader = dataloader
                    self.dataset = dataloader.dataset
                    self.weblogger = weblogger
                    self.transformation = transformation

                    self.scale_steps = np.array([np.linspace(1, 0.2, self.max_steps), np.linspace(1, 1.5, self.max_steps)]).T
                    self.translate_steps = np.array([np.linspace(0, 0.3, self.max_steps), np.linspace(0, 0.3, self.max_steps)]).T
                    self.degree_steps = np.linspace(0, 180, self.max_steps)

                    self.dataset.transform.transforms.insert(0, RandomAffine(degrees=0, scale=None, translate=None))


                def go_next_level(self, logs, *args, **kwargs):
                    if self.current_step == self.max_steps - 1:
                        logs['stop'] = True
                    else:
                        self.current_step += 1
                        scale = None
                        translate = None
                        degree = 0
                        if 't' in self.transformation:
                            translate = list(self.translate_steps[self.current_step])
                        if 'r' in self.transformation:
                            degree = self.degree_steps[self.current_step]
                        if 's' in self.transformation:
                            scale = list(self.scale_steps[self.current_step])
                        self.dataset.transform.transforms[[idx for idx, i in enumerate(self.dataset.transform.transforms) if isinstance(i, RandomAffine)][0]] = RandomAffine(degrees=degree, scale=scale, translate=translate)
                        print(fg.magenta + f"Updated transformation. Step {self.current_step}/{self.max_steps}. Translate {translate}, Rot {degree}, Scale {scale}." + rs.fg)
                        for idx, data in enumerate(self.dataloader):
                            images, labels, more = data
                            framework_utils.plot_images_on_weblogger(self.dataset, self.dataset.name_generator, self.dataset.stats, images, labels, more, log_text=f'STEP {self.current_step}', weblogger=self.weblogger)
                            break

            ck = CheckStoppingLevel(train_loader, self.transformations, weblogger=self.weblogger)
            all_cb += [TriggerActionWithPatience(min_delta=1, patience=100, min_delta_is_percentage=True, mode='max',
                                                 reaching_goal=85,
                                                 metric_name='webl/mean_acc' if self.weblogger else 'cnsl/mean_acc',
                                                 check_every=self.weblog_check_every if self.weblogger else self.console_check_every,
                                                 triggered_action=ck.go_next_level)]

        return all_cb


def create_image_viewpoints_exp(obj_class):
    class ShapeNetExp(obj_class):
        def __init__(self, **kwargs):
            self.diff_classes_set = None
            self.classes_set = None
            self.shapenet_folder = None
            self.num_viewpoints_train = None
            self.num_viewpoints_test = None
            self.max_objs_per_class_train = None
            self.max_objs_per_class_test = None
            self.transformations = None
            self.transformations_test = None
            self.do_train = None
            super().__init__(**kwargs)

        def parse_arguments(self, parser):
            super().parse_arguments(parser)
            parser.add_argument("-classes_set", "--classes_set",
                                help="Specify the object set for training [set1], [set2]",
                                type=str,
                                default=None)
            parser.add_argument("-diff_classes_set", "--diff_classes_set",
                                help="Specify the object set for testing [set1], [set2]",
                                type=str,
                                default=None)
            parser.add_argument("-shapenet_folder", "--shapenet_folder",
                                help="Specify the folder to use in ./data/ShapeNet2D/",
                                type=str,
                                default='ShapeNet2DFullNomat')
            parser.add_argument("-max_objs_per_class_train", "--max_objs_per_class_train",
                                help="Determines the maximum number of objects per each class from the ShapeNet dataset for training",
                                type=int,
                                default=None)
            parser.add_argument("-max_objs_per_class_test", "--max_objs_per_class_test",
                                help="Determines the maximum number of objects per each class from the ShapeNet dataset for testing",
                                type=int,
                                default=None)
            parser.add_argument("-num_vtr", "--num_viewpoints_train",
                                help="Num viewpoints to train on. -1 to use all",
                                type=int,
                                default=None)
            parser.add_argument("-num_vte", "--num_viewpoints_test",
                                help="Num viewpoints to train on. -1 to use all",
                                type=int,
                                default=None)
            parser.add_argument("-transform", "--transformations",
                                help="any combination of s, t, r e.g. [rst] [rs] [t]",
                                type=str,
                                default='')
            parser.add_argument("-transform_test", "--transformations_test",
                                help="any combination of s, t, r e.g. [rst] [rs] [t]",
                                type=str,
                                default=None)
            parser.add_argument("-do_train", "--do_train",
                                help="If False, only do test",
                                type=lambda x: bool(int(x)),
                                default=True)
            return parser

        def finalize_init(self, PARAMS, list_tags):
            self.classes_set = PARAMS['classes_set']
            self.diff_classes_set = PARAMS['diff_classes_set']
            self.shapenet_folder = PARAMS['shapenet_folder']
            self.num_viewpoints_train = PARAMS['num_viewpoints_train']
            self.num_viewpoints_test = PARAMS['num_viewpoints_test']
            self.max_objs_per_class_train = PARAMS['max_objs_per_class_train']
            self.max_objs_per_class_test = PARAMS['max_objs_per_class_test']
            self.transformations = PARAMS['transformations']
            self.transformations_test = PARAMS['transformations_test']
            self.do_train = PARAMS['do_train']
            if self.num_viewpoints_test is None:
                self.num_viewpoints_test = self.num_viewpoints_train
            if self.max_objs_per_class_test is None:
                self.max_objs_per_class_test = self.max_objs_per_class_train
            if self.transformations_test is None:
                self.transformations_test = self.transformations

            # list_tags.append(self.classes_set)
            super().finalize_init(PARAMS, list_tags)
    return ShapeNetExp

exp_categorization_task = create_image_viewpoints_exp(TransformExperiment)

all_classes = {'airplane.n.01': 3182, 'ashcan.n.01': 274, 'bag.n.06': 66, 'basket.n.01': 90, 'bathtub.n.01': 685, 'bed.n.01': 186, 'bench.n.01': 1450, 'birdhouse.n.01': 58, 'bookshelf.n.01': 359, 'bottle.n.01': 398, 'bowl.n.03': 149, 'bus.n.01': 736, 'cabinet.n.01': 1254, 'camera.n.01': 90, 'can.n.01': 86, 'cap.n.01': 44, 'car.n.01': 2187, 'cellular_telephone.n.01': 665, 'chair.n.01': 5404, 'clock.n.01': 518, 'computer_keyboard.n.01': 52, 'dishwasher.n.01': 74, 'display.n.06': 873, 'earphone.n.01': 58, 'faucet.n.01': 595, 'file.n.03': 238, 'guitar.n.01': 638, 'helmet.n.02': 130, 'jar.n.01': 475, 'knife.n.02': 339, 'lamp.n.02': 1844, 'laptop.n.01': 368, 'loudspeaker.n.01': 1276, 'mailbox.n.01': 75, 'microphone.n.01': 54, 'microwave.n.02': 122, 'motorcycle.n.01': 269, 'mug.n.04': 171, 'piano.n.01': 188, 'pillow.n.01': 77, 'pistol.n.01': 246, 'pot.n.04': 454, 'printer.n.03': 133, 'remote_control.n.01': 53, 'rifle.n.01': 1894, 'rocket.n.01': 66, 'skateboard.n.01': 121, 'sofa.n.01': 2532, 'stove.n.02': 174, 'table.n.02': 6744, 'telephone.n.01': 871, 'tower.n.01': 104, 'train.n.01': 298, 'vessel.n.02': 1513, 'washer.n.03': 135}


ss = np.argsort([v for k, v in all_classes.items()])
class_name = [k for k, v in all_classes.items()]
ordered_name = [class_name[i] for i in ss]
all_clean_name = [re.sub('.n.0\d+', '', i) for i in class_name]

small_set = {v:k for v, k in all_classes.items() if k < 200}
len(small_set)
all_minus_small = {v:k for v,k in all_classes.items() if v not in small_set}
len(all_minus_small)

selected = ['airplane', 'bench', 'bus', 'car', 'clock', 'guitar', 'knife', 'lamp', 'rifle', 'sofa', 'table', 'vessel', 'bathtub', 'bottle', 'cellular_telephone', 'chair', 'pot', 'jar', 'faucet', 'laptop']
rest = list(set(all_clean_name) - set(selected))
## THE GLOBAL SEED CAN CHANGE, BUT WE WANT TO ALWAYS DRAW THE SAME CLASSES
initial_seed = np.random.get_state()
np.random.seed(0)
all_idx = np.random.choice(range(len(selected)), len(selected), replace=False)
# set original seed
np.random.set_state(initial_seed)

new_order = [selected[i] for i in all_idx]
set1 = {}
set2 = {}
set_rest = {}
[{set1.update({k:v}) for  k,v in all_classes.items()  if re.search(f'{a}.n.', k)} for a in new_order[:10]]
[{set2.update({k:v}) for  k,v in all_classes.items()  if re.search(f'{a}.n.', k)} for a in new_order[10:]]
[{set_rest.update({k:v}) for  k,v in all_classes.items()  if re.search(f'{a}.n.', k)} for a in rest]

classes_sets = {}
classes_sets['set1'] = list(set1.keys())
classes_sets['set2'] = list(set2.keys())
classes_sets['rest'] = list(set_rest.keys())

id_to_name = pickle.load(open(n, 'rb'))
name_to_id = {v:k for k,v in id_to_name.items()}
classes_sets['id_set1'] = {name_to_id[k]: v for k, v in set1.items()}
classes_sets['id_set2'] = {name_to_id[k]: v for k, v in set2.items()}
classes_sets['id_rest'] = {name_to_id[k]: v for k, v in set_rest.items()}
