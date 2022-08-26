import os
import pathlib
import pickle
import re
from pathlib import Path

import numpy as np
import torchvision
from sty import fg, rs, ef
from torch import Tensor
from torch.utils.data import Sampler, WeightedRandomSampler, RandomSampler
from torchvision.datasets import DatasetFolder
from torchvision.transforms import RandomAffine, functional as F

from callbacks import GenericDataFrameSaver
from datasets import MyImageFolder


class TransformInfo:
    def __call__(self, tensor):
        return tensor, {}


class RandomAffineInfo(RandomAffine, TransformInfo):
    def forward(self, img):
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        img_size = F._get_image_size(img)

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)

        return F.affine(img, *ret, interpolation=self.interpolation, fill=fill), \
               {'angle': ret[0], 'translation': ret[1], 'scale': ret[2], 'shear': ret[3]}


class TransformationInfoDataFrameSaver(GenericDataFrameSaver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.additional_logs_names = ['angle', 'translX', 'translY', 'scale', 'azi', 'incl']
        self.column_names.extend(self.additional_logs_names)

    def _get_additional_logs(self, logs, c):
        angle = logs['angle'][c].item() if 'angle' in logs else None
        trX = logs['translation'][0][c].item() if 'translation' in logs else None
        trY = logs['translation'][1][c].item() if 'translation' in logs else None
        scale = logs['scale'][c].item() if 'scale' in logs else None
        azi = logs['azi'][c].item() if 'azi' in logs else None
        incl = logs['incl'][c].item() if 'incl' in logs else None
        return [angle, trX, trY, scale, azi, incl]

    def _compute_and_log_metrics(self, data_frame):
        return data_frame


class SelectObjects(MyImageFolder):
    """
    This is used for the objects in ShapeNet
    """
    def __init__(self, name_classes=None, num_objects_per_class=None, selected_objects=None, num_viewpoints_per_object=None, take_specific_azi_incl=True, save_load_samples_filename=None, **kwargs):
        self.name_classes = name_classes
        # take_specific_azi_incl = False  take random viewpoints
        #                           True  take one specific viewpoint: 75, 36
        #                           (x, y) take one specific viewpoint: (x,y)
        # only work if num_viewpoints_per_object is 1, otherwise changed to False.
        if num_viewpoints_per_object == 1 and take_specific_azi_incl is True:
            take_specific_azi_incl = (75, 36)
        super().__init__(name_classes, **kwargs)
        self.selected_objects = selected_objects
        get_obj_num = lambda name: int(re.search(r"O(\d+)_", name).groups()[0])
        original_sample_size = len(self.samples)
        loaded = False
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        if save_load_samples_filename is not None:
            if os.path.isfile(save_load_samples_filename):
                print(fg.red + f"LOADING SAMPLES FROM /samples/{Path(save_load_samples_filename).name}" + rs.fg)
                if num_objects_per_class is not None or selected_objects is not None or num_viewpoints_per_object is not None:
                    print("Max num objects, Num Viewpoints and specific object to select will be ignored")
                self.samples = pickle.load(open(save_load_samples_filename, 'rb'))
                self.selected_objects = np.hstack([np.unique([get_subclass_from_sample(self.idx_to_class, i) for i in self.samples if i[1] == c]) for c in range(len(self.classes))])
                loaded = True
            else:
                print(fg.yellow + f"Path {save_load_samples_filename} not found, will compute samples" + rs.fg)

        if not loaded:
            from progress.bar import Bar
            if num_objects_per_class is not None or self.selected_objects is not None:
                print("SELECTING OBJECTS")

                all_objs_count = ([np.unique([get_subclass_from_sample(self.idx_to_class, i) for i in self.samples if i[1] == c]) for c in range(len(self.classes))])
                if self.selected_objects is None:
                    self.selected_objects = np.hstack([np.random.choice(o, np.min((num_objects_per_class, len(o))), replace=False) for o in all_objs_count])
                self.samples = [i for i in self.samples if get_subclass_from_sample(self.idx_to_class, i) in self.selected_objects]

            if num_viewpoints_per_object is not None:
                print("SELECTING VIEWPOINTS")
                selected_vp_samples = []
                index_next_object = 0
                bar = Bar(f"Selecting {num_viewpoints_per_object} viewpoints", max=len(self.samples))

                if take_specific_azi_incl and num_viewpoints_per_object > 1:
                    print(fg.yellow + "Num_viewpoints_per_object > 1 and take_specific_azi_incl is True. Take_specific_azi_incl changed to false (random vp)" + rs.fg)

                while True:
                    # find next index:
                    obj_num = get_obj_num(self.samples[index_next_object][0])
                    j = index_next_object
                    while j < len(self.samples) and get_obj_num(self.samples[j][0]) == obj_num:
                        j += 1

                    if num_viewpoints_per_object == 1 and take_specific_azi_incl:
                        selected_sample = [i for i in self.samples[index_next_object:j] if re.search(rf'O\d+_I{take_specific_azi_incl[0]}_A{take_specific_azi_incl[1]}', i[0])]
                        assert len(selected_sample) <= 1
                        if not selected_sample:
                            #if not found
                            incl_azi = [re.search(rf'O\d+_I(\d+)_A(\d+)', i[0]) for i in self.samples[index_next_object:j]]
                            incl_azi = [(int(i.groups()[0]), int(i.groups()[1])) for i in incl_azi]
                            selected_index = np.argmin(np.mean(np.abs(np.array(incl_azi)-take_specific_azi_incl), 1))
                            selected_incl_azi = incl_azi[selected_index]
                            print(fg.red + f"Viewpoint {take_specific_azi_incl} not found, using {selected_incl_azi}" + rs.fg)
                            selected_sample = self.samples[index_next_object:j][selected_index]
                        else:
                            selected_sample = selected_sample[0]

                        selected_vp_samples.append(selected_sample)

                    else:
                        itmp = np.random.choice(j - index_next_object, np.min((num_viewpoints_per_object, j - index_next_object)), replace=False)
                        selected_vp_samples.extend([self.samples[index_next_object:j][i] for i in itmp])

                    bar.next(n=j-index_next_object)
                    index_next_object = j

                    if j >= len(self.samples):
                        break

                self.samples = selected_vp_samples

        # all_objects = np.unique([get_obj_num(i[0]) for i in self.samples])
        print(f"\nNum ALL samples: {original_sample_size}, Num Objects: {len(self.selected_objects) if self.selected_objects is not None else 'all'},  Num selected vp: {num_viewpoints_per_object}, \nFinal Num Samples = num objs x num vp: {len(self.samples)}")
        all_objs_count = {self.idx_to_class[j]: len(np.unique([get_subclass_from_sample(self.idx_to_class, i) for i in self.samples if i[1] == j])) for j in range(len(self.classes))}
        [print('{:25}: {:4}'.format(k, i)) for k, i in all_objs_count.items()]
        tmplst = [np.min((i, len(self.samples)-1)) for i in [0, 1, 3, 5, 100, 2000, -1]]
        print(fg.cyan + f"Samples in indexes {tmplst}:\t" + rs.fg, end="")
        [print(rf'{i[1]}: {get_subclass_with_azi_incl(self.idx_to_class, i[0])}', end="\t") for i in [(self.samples[s], s) for s in tmplst]]
        if self.selected_objects is not None:
            tmplst = [np.min((i, len(self.selected_objects)-1)) for i in [0, 1, 3, 5, 100, 2000, -1]]
            print(fg.cyan + f"\nObjects in indexes {tmplst}:\t" + rs.fg, end="")
            [print(f'{i[1]}: {i[0]}', end="\t") for i in [(self.selected_objects[s], s) for s in tmplst]]

        print()

        if save_load_samples_filename is not None and loaded is False:
            print(fg.yellow + f"SAVING SAMPLES IN /samples/{Path(save_load_samples_filename).name}" + rs.fg)
            pathlib.Path(Path(save_load_samples_filename).parent).mkdir(parents=True, exist_ok=True)
            pickle.dump(self.samples, open(save_load_samples_filename, 'wb'))

    def _find_classes(self, dir: str):
        classes = [d.name for d in os.scandir(dir) if d.is_dir() and (d.name in self.name_classes if self.name_classes is not None else True)]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class SubclassImageFolder(SelectObjects):
    def __init__(self, sampler=None, **kwargs):
        super().__init__(**kwargs)
        self.subclasses = {}
        self.subclass_to_idx = {}
        self.classes_to_subclasses = {}
        self.subclasses_names = []
        self.subclasses_to_classes = {}
        subclass_idx = 0
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        for idx, i in enumerate(self.samples):
            subclass_name = get_subclass_from_sample(self.idx_to_class, i)

            if subclass_name in self.subclasses:
                self.subclasses[subclass_name].append(idx)
            else:
                self.subclasses_names.append(subclass_name)
                self.subclasses[subclass_name] = [idx]
                self.subclasses_to_classes[subclass_name] = i[1]
                if self.idx_to_class[i[1]] not in self.classes_to_subclasses:
                    self.classes_to_subclasses[self.idx_to_class[i[1]]] = [subclass_name]
                else:
                    self.classes_to_subclasses[self.idx_to_class[i[1]]].append(subclass_name)

                subclass_idx += 1
        print('Length classes:')
        # [print('{:20}: {:4}'.format(k, len(i))) for k, i in self.classes_to_subclasses.items()]
        self.subclass_to_idx = {k: idx for idx, k in enumerate(self.subclasses_names)}
        self.idx_to_subclass = {v: k for k, v in self.subclass_to_idx.items()}
        self.samples_sb = [(a[0], a[1], self.subclass_to_idx[get_subclass_from_sample(self.idx_to_class, a)]) for a in self.samples]

        print(f"Subclasses folder dataset. Num classes: {len(self.classes)}, num subclasses: {len(self)}")

    def __getitem__(self, index):
        path, class_idx, object_idx = self.samples_sb[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        sample, class_idx, info = self.finalize_getitem(path=path, sample=sample, labels=class_idx, info={'label_object': object_idx})
        return sample, class_idx, info


def get_subclass_from_sample(idx_to_class, sample):
    get_obj_num = lambda name: int(re.search(r"O(\d+)_", name).groups()[0])
    name_class = idx_to_class[sample[1]]
    return name_class + '_' + str(get_obj_num(sample[0]))


def get_subclass_with_azi_incl(idx_to_class, sample):
    get_obj_num = lambda name: int(re.search(r"O(\d+)_", name).groups()[0])
    name_class = idx_to_class[sample[1]]
    incl_azi = re.search('O\d+_I(\d+)_A(\d+).png', sample[0]).groups()
    return name_class + '_' + str(get_obj_num(sample[0])) + f'_I{incl_azi[0]}_A{incl_azi[1]}'


class SameDifferentSampler(Sampler):
    def get_subclasses_weighted_sampler(self, dataset):
        weights_class = 1 / np.array([np.sum([True for k, v in dataset.subclasses_to_classes.items() if v == c]) for c in range(len(dataset.classes))])
        weights_dataset = [weights_class[dataset.subclasses_to_classes[i]] for i in dataset.subclasses_names]
        return WeightedRandomSampler(weights=weights_dataset, num_samples=len(dataset.subclasses_names), replacement=True)

    def __init__(self, batch_size, dataset, subclasses, subclasses_names, prob_same=0.5, rebalance_classes=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.subclasses = subclasses
        self.subclasses_names = subclasses_names
        self.prob_same = prob_same
        # self.subclasses_list = copy.deepcopy(self.subclasses_names)
        if rebalance_classes:
            self.sampler_training = self.get_subclasses_weighted_sampler(dataset)
            self.sampler_candidate = self.get_subclasses_weighted_sampler(dataset)
        else:
            self.sampler_training = RandomSampler(self.subclasses_names)
            self.sampler_candidate = RandomSampler(self.subclasses_names)
        # else:
        #     self.sampler = SequentialSampler(self.subclasses_names)

    def __iter__(self):
        training_idx = []
        candidate_idx = []
        i = iter(self.sampler_candidate)
        for idx in self.sampler_training:
            sel_obj = self.subclasses_names[idx]
            training_idx.append(np.random.choice(self.subclasses[sel_obj], 1)[0])
            if np.random.rand() < self.prob_same:
                candidate_idx.append(np.random.choice(self.subclasses[sel_obj], 1)[0])
            else:
                candidate_idx.append(np.random.choice(self.subclasses[self.subclasses_names[next(i)]], 1)[0])
                # candidate_idx.append(np.random.choice(self.subclasses[np.random.choice(list(set(self.subclasses_names) - set([sel_obj])), 1)[0]], 1)[0])
            if len(candidate_idx) == self.batch_size:
                yield np.hstack((candidate_idx, training_idx))
                candidate_idx = []
                training_idx = []
        yield np.hstack((candidate_idx, training_idx))


def add_unpack_transf_info(obj: DatasetFolder):
    class MyCompose(torchvision.transforms.Compose):
        def __call__(self, img):
            info = {}
            for t in self.transforms:
                if isinstance(t, TransformInfo):
                    img, i = t(img)
                    info.update(i)
                else:
                    img = t(img)
            return img, info

    class UnpackTransformInfo(obj):
        def finalize_getitem(self, path, sample, labels, info):
            img, i = sample
            incl_azi = re.search("I(\d+)_A(\d+)", os.path.basename(path))
            vp_info = {'azi': int(incl_azi.groups()[1]), 'incl':  int(incl_azi.groups()[0])}
            info.update(i)
            info.update(vp_info)
            return img, labels, info

        def finalize_init(self):
            self.transform = MyCompose(self.transform.transforms)
    return UnpackTransformInfo


def get_transforms(code, info=False):
    import torchvision.transforms as transforms
    add_PIL_transforms = []
    add_T_transforms = []
    scale = None
    translate = None
    contrast_value = 0
    brightness_value = 0
    degree = 0
    if 's' in code:
        scale = [0.2, 1.5]
    if 't' in code:
        translate = [0.3, 0.3]
    if 'r' in code:
        degree = 180
    if 'c' in code:
        contrast_value = 0.8
    if 'b' in code:
        brightness_value = 0.8
    if 's' in code or 'r' in code or 't' in code:
        add_PIL_transforms += [RandomAffineInfo(degrees=degree, scale=scale, translate=translate)] if info else\
                              [RandomAffine(degrees=degree, scale=scale, translate=translate)]

    if 'b' in code or 'c' in code:
        add_PIL_transforms += [transforms.ColorJitter(brightness=brightness_value, contrast=contrast_value)]
    if '' in code:
        pass
    if 'g' in code:
        add_PIL_transforms += [transforms.Grayscale(num_output_channels=3)]

    return add_PIL_transforms, add_T_transforms