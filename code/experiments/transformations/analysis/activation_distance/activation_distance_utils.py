from distance_activation import DistanceActivation
from distance_activation import CompareWith
from PIL import Image
from torchvision.transforms import RandomAffine
import re
from experiments.transformations.utils.misc import classes_sets, exp_categorization_task
from experiments.transformations.utils.datasets import SubclassImageFolder
from datasets import add_compute_stats, SubsetImageFolder
import numpy as np
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
from pathlib import Path


def map_string_to_label(str):
    if str == 'vanilla':
        return 'Untrained'
    if str == 'ImageNet':
        return 'ImageNet'
    if str == '':
        return 'No Transformation'
    if str == 'r':
        return 'Rotation'
    if str == 's':
        return 'Scale'
    if str == 't':
        return 'Translate'
    if str == 'vtrsbc':
        return 'All'
    if str == 'v':
        return 'Viewpoint'
    else:
        return str


class DistanceActivationScale(DistanceActivation):
    def get_base_and_other_canvasses(self, class_num, name_class):
        selected_class = [i for i in self.dataset.samples if i[1] == class_num]
        num = np.random.choice(len(selected_class))
        image_name_base = selected_class[num][0]

        default_transform = self.dataset.transform  # this is generally to tensor, normalize, and sometime resize
        scale_values = np.linspace(0.2, 1.8, 50)
        image_base = Image.open(image_name_base)
        other_canvasses = []
        base_canvas = default_transform(image_base)

        for s in scale_values:
            if self.compare_with == CompareWith.ANY_OBJECT:
                num = np.random.choice(len(self.dataset.samples))
                image_name = self.dataset.samples[num][0]
                image_comp = Image.open(image_name)
            if self.compare_with == CompareWith.SAME_CLASS_DIFF_OBJ:
                selected_class = [idx for idx, i in enumerate(self.dataset.samples) if i[1] == class_num and i[0] != image_name_base]
                num = np.random.choice(len(selected_class))
                idx_selected = selected_class[num]
                image_name = self.dataset.samples[idx_selected][0]
                image_comp = Image.open(image_name)
            if self.compare_with == CompareWith.SAME_OBJECT:
                image_comp = image_base

            ra = RandomAffine(degrees=0, scale=[s, s])
            other_canvasses.append(default_transform(ra(image_comp)))

        return base_canvas, other_canvasses, scale_values


class DistanceActivationRotate(DistanceActivation):
    def get_base_and_other_canvasses(self, class_num, name_class):
        selected_class = [i for i in self.dataset.samples if i[1] == class_num]
        num = np.random.choice(len(selected_class))
        image_name_base = selected_class[num][0]

        normalize = self.dataset.transform
        degrees_values = np.linspace(-180, 180, 50)
        image_base = Image.open(image_name_base)
        other_canvasses = []
        base_canvas = normalize(image_base)

        for d in degrees_values:
            if self.compare_with == CompareWith.ANY_OBJECT:
                num = np.random.choice(len(self.dataset.samples))
                image_name = self.dataset.samples[num][0]
                image_comp = Image.open(image_name)
            if self.compare_with == CompareWith.SAME_CLASS_DIFF_OBJ:
                selected_class = [idx for idx, i in enumerate(self.dataset.samples) if i[1] == class_num and i[0] != image_name_base]
                num = np.random.choice(len(selected_class))
                idx_selected = selected_class[num]
                image_name = self.dataset.samples[idx_selected][0]
                image_comp = Image.open(image_name)
            if self.compare_with == CompareWith.SAME_OBJECT:
                image_comp = image_base

            ra = RandomAffine(degrees=(d, d+0.01))
            transf_image = ra(image_comp)
            other_canvasses.append(normalize(transf_image))

        return base_canvas, other_canvasses, degrees_values


class DistanceActivationContrast(DistanceActivation):
    def get_base_and_other_canvasses(self, class_num, name_class):
        selected_class = [i for i in self.dataset.samples if i[1] == class_num]
        num = np.random.choice(len(selected_class))
        image_name_base = selected_class[num][0]

        normalize = self.dataset.transform
        values = np.linspace(0.2, 1.8, 20)
        image_base = Image.open(image_name_base)
        other_canvasses = []
        base_canvas = normalize(image_base)
        for d in values:
            if self.compare_with == CompareWith.ANY_OBJECT:
                num = np.random.choice(len(self.dataset.samples))
                image_name = self.dataset.samples[num][0]
                image_comp = Image.open(image_name)
            if self.compare_with == CompareWith.SAME_CLASS_DIFF_OBJ:
                selected_class = [idx for idx, i in enumerate(self.dataset.samples) if i[1] == class_num and i[0] != image_name_base]
                num = np.random.choice(len(selected_class))
                idx_selected = selected_class[num]
                image_name = self.dataset.samples[idx_selected][0]
                image_comp = Image.open(image_name)
            if self.compare_with == CompareWith.SAME_OBJECT:
                image_comp = image_base

            transf_image = F.adjust_contrast(image_comp, d)
            other_canvasses.append(normalize(transf_image))

        return base_canvas, other_canvasses, values


class DistanceActivationBrightness(DistanceActivation):
    def get_base_and_other_canvasses(self, class_num, name_class):
        selected_class = [i for i in self.dataset.samples if i[1] == class_num]
        num = np.random.choice(len(selected_class))
        image_name_base = selected_class[num][0]

        normalize = self.dataset.transform
        values = np.linspace(0, 1.8, 20)
        image_base = Image.open(image_name_base)
        other_canvasses = []
        base_canvas = normalize(image_base)
        for d in values:
            if self.compare_with == CompareWith.ANY_OBJECT:
                num = np.random.choice(len(self.dataset.samples))
                image_name = self.dataset.samples[num][0]
                image_comp = Image.open(image_name)

            if self.compare_with == CompareWith.SAME_CLASS_DIFF_OBJ:
                selected_class = [idx for idx, i in enumerate(self.dataset.samples) if i[1] == class_num and i[0] != image_name_base]
                num = np.random.choice(len(selected_class))
                idx_selected = selected_class[num]
                image_name = self.dataset.samples[idx_selected][0]
                image_comp = Image.open(image_name)
            if self.compare_with == CompareWith.SAME_OBJECT:
                image_comp = image_base

            transf_image = F.adjust_brightness(image_comp, d)
            other_canvasses.append(normalize(transf_image))

        return base_canvas, other_canvasses, values


class DistanceActivationTranslate(DistanceActivation):
    def get_base_and_other_canvasses(self, class_num, name_class):
        selected_class = [i for i in self.dataset.samples if i[1] == class_num]
        num = np.random.choice(len(selected_class))
        image_name_base = selected_class[num][0]

        default_transform = self.dataset.transform
        tx = np.linspace(-0.3, 0.3, 25)
        ty = np.linspace(-0.3, 0.3, 25)
        x, y = np.meshgrid(tx, ty)
        x = x.flatten()
        y = y.flatten()
        image_base = Image.open(image_name_base)
        other_canvasses = []
        base_canvas = default_transform(image_base)

        for idx, _ in enumerate(x):
            if self.compare_with == CompareWith.ANY_OBJECT:
                num = np.random.choice(len(self.dataset.samples))
                image_name = self.dataset.samples[num][0]
                image_comp = Image.open(image_name)

            if self.compare_with == CompareWith.SAME_CLASS_DIFF_OBJ:
                selected_class = [idx for idx, i in enumerate(self.dataset.samples) if i[1] == class_num and i[0] != image_name_base]
                num = np.random.choice(len(selected_class))
                idx_selected = selected_class[num]
                image_name = self.dataset.samples[idx_selected][0]
                image_comp = Image.open(image_name)
            if self.compare_with == CompareWith.SAME_OBJECT:
                image_comp = image_base

            dx = float(x[idx] * image_comp.size[0])
            dy = float(y[idx] * image_comp.size[1])
            new_img = F.affine(image_comp, 0, (dx, dy), 1, 0, interpolation=InterpolationMode.NEAREST, fill=0)
            other_canvasses.append(default_transform(new_img))

        return base_canvas, other_canvasses, (x, y)


class DistanceActivationViewpoint(DistanceActivation):
    def get_base_and_other_canvasses(self, class_num, name_class):
        objects = self.dataset.classes_to_subclasses[name_class]
        num = np.random.choice(len(objects))
        obj = objects[num]
        base_incl_azi = (75, 36)
        all_paths = [self.dataset.samples[i][0] for i in self.dataset.subclasses[obj]]
        default_transform = self.dataset.transform

        incl_azi = [re.search(rf'O\d+_I(\d+)_A(\d+)',i) for i in all_paths]
        incl_azi = [(int(i.groups()[0]), int(i.groups()[1])) for i in incl_azi]

        base_path = [i for i in all_paths if re.search(rf'O\d+_I{base_incl_azi[0]}_A{base_incl_azi[1]}', i)]

        assert len(base_path) == 1
        image_base = Image.open(base_path[0])
        other_canvasses = []
        base_canvas = default_transform(image_base)

        incl = []
        azi = []
        for idx, p in enumerate(all_paths):
            incl.append(incl_azi[idx][0])
            azi.append(incl_azi[idx][1])

            if self.compare_with == CompareWith.ANY_OBJECT:
                num = np.random.choice(len(self.dataset.idx_to_subclass))
                rnd_objs = [self.dataset.samples[i] for i in self.dataset.subclasses[self.dataset.idx_to_subclass[num]]]
                image_name = [i for i in rnd_objs if re.search(f'I{incl[-1]}_A{azi[-1]}', i[0])][0][0]
                image_comp = Image.open(image_name)

            if self.compare_with == CompareWith.SAME_CLASS_DIFF_OBJ:
                num = np.random.choice(len([i for i in self.dataset.classes_to_subclasses[name_class] if i!= obj]))
                rnd_objs = [self.dataset.samples[i] for i in self.dataset.subclasses[self.dataset.classes_to_subclasses[name_class][num]]]
                image_name = [i for i in rnd_objs if re.search(f'I{incl[-1]}_A{azi[-1]}', i[0])][0][0]
                image_comp = Image.open(image_name)

            if self.compare_with == CompareWith.SAME_OBJECT:
                image_comp = Image.open(p)

            other_canvasses.append(default_transform(image_comp))

        return base_canvas, other_canvasses, (incl, azi)


def get_distance_fun(distance_transform):
    if distance_transform == 'VP':
        distAct = DistanceActivationViewpoint
    elif distance_transform == 'Rotate':
        distAct = DistanceActivationRotate
    elif distance_transform == 'Scale':
        distAct = DistanceActivationScale
    elif distance_transform == 'Translate':
        distAct = DistanceActivationTranslate
    elif distance_transform == 'Contrast':
        distAct = DistanceActivationContrast
    elif distance_transform == 'Brightness':
        distAct = DistanceActivationBrightness
    return distAct


def get_experiment_shapeNet(type_dataset='same_classes_diff_objs'):
    this_file_path = str(Path(globals().get("__file__", "./_")).absolute().parent)

    if type_dataset == 'same_classes_diff_objs':
        classes_set = 'set1'
        name_dataset = 'ShapeNet2DFullNomat/testDiffObjs'
    elif type_dataset == 'different_classes':
        classes_set = 'id_set2'
        name_dataset = 'whole_dataset_mv_nomat'
    elif type_dataset == 'rest':
        classes_set = 'id_rest'
        name_dataset = 'whole_dataset_mv_nomat'
    else:
        assert False, 'type_dataset not recognised'
    network_name = 'vgg11bn'
    pt = 'vanilla'
    max_objs_per_class_train = 3
    exp = exp_categorization_task(seed=1,
                                  classes_set=classes_set,
                                  network_name=network_name,
                                  max_objs_per_class_train=max_objs_per_class_train,
                                  transformations='',
                                  transformations_test='',
                                  pretraining=pt,
                                  use_weblog=False,
                                  use_device_num=3,
                                  verbose=False)


    stats = {'mean': [0.06229676, 0.0607271, 0.05646703], 'std': [0.14454809, 0.14061172, 0.12994126]}

    dataset_args = dict(stats=stats, name_classes=classes_sets[exp.classes_set], name_generator=f"train {exp.classes_set}", root=f"./data/ShapeNet2D/{name_dataset}", save_load_samples_filename=this_file_path + f'/samples/whole_dataset_mv{max_objs_per_class_train}.s', num_objects_per_class=exp.max_objs_per_class_train)

    dataset_no_transf = add_compute_stats(SubclassImageFolder)(**dataset_args)

    return exp, dataset_no_transf


# This also worked but I didn't have time to prepare it properly for the paper!
def get_cifar10_exp(network_name):
    cifar10_set2 = ['bird', 'dog', 'ship']
    exp = exp_categorization_task(seed=1,
                                  network_name=network_name,
                                  transformations='',
                                  use_weblog=False,
                                  use_device_num=3,
                                  verbose=False)

    stats = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247, 0.243, 0.261]}
    add_PIL_transforms = None
    dataset = add_compute_stats(SubsetImageFolder)(root='./data/CIFAR10/train',
                                                   add_PIL_transforms=add_PIL_transforms,
                                                   name_generator="train cifar10", 
                                                   save_stats_file=None,
                                                   name_classes=cifar10_set2,
                                                   stats=stats)

    return exp, dataset

## Some code useful for debugging
# type_dataset = 'different_classes'
# exp, dataset_no_transf = get_experiment_shapeNet(type_dataset)
#
# # exp, dataset_no_transf = get_cifar10_exp('vgg11bn')
# distance_metric = 'cossim'
# exp.pretraining = 'vanilla'
# net = exp.get_net(1000)
# distAct = DistanceActivationTranslate
# dist = distAct(net, dataset_no_transf, use_cuda=exp.use_cuda, distance=distance_metric, compare_with=CompareWith.SAME_CLASS_DIFF_OBJ)
# distance_net, distance_img, x_values = dist.calculate_distance_dataloader()