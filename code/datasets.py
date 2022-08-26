import os
import pathlib
import pickle
from pathlib import Path
from time import time

import cloudpickle
import numpy as np
import torch
import torchvision
from PIL import ImageStat
from sty import fg, rs, ef
from torchvision import transforms as tf
from torchvision.datasets import ImageFolder


class MyImageFolder(ImageFolder):
    def finalize_getitem(self, path, sample, labels, info=None):
        if info is None:
            info = {}
        return sample, labels, info

    def __init__(self, name_classes=None, *args, **kwargs):
        print(fg.red + ef.inverse + "ROOT:  " + kwargs['root'] + rs.inverse + rs.fg)
        self.name_classes = name_classes
        super().__init__(*args, **kwargs)

    def _find_classes(self, dir: str):
        if self.name_classes is None:
            return super()._find_classes(dir)
        else:
            classes = [d.name for d in os.scandir(dir) if d.is_dir() and (d.name in self.name_classes)]
            classes.sort()
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        output = self.finalize_getitem(path=path, sample=sample, labels=target)
        return output


def add_return_path(class_obj):
    class ImageFolderWithPath(class_obj):
        def __getitem__(self, index):
            img, target = super().__getitem__(index)
            path = self.samples[index][0]
            return img, target, {'path': path}
    return ImageFolderWithPath


def add_compute_stats(obj_class):
    class ComputeStatsUpdateTransform(obj_class):
        ## This class basically is used for normalize Dataset Objects such as ImageFolder in order to be used in our more general framework
        def __init__(self, name_generator, add_PIL_transforms=None, add_tensor_transforms=None, num_image_calculate_mean_std=70, stats=None, save_stats_file=None, **kwargs):
            """

            @param add_tensor_transforms:
            @param stats: this can be a dict (previous stats, which will contain 'mean': [x, y, z] and 'std': [w, v, u], a path to a pickle file, or None
            @param save_stats_file:
            @param kwargs:
            """
            print(fg.yellow + f"\n**Creating Dataset [" + fg.cyan + f"{name_generator}" + fg.yellow + "]**" + rs.fg)

            super().__init__(**kwargs)
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

            if add_PIL_transforms is None:
                add_PIL_transforms = []
            if add_tensor_transforms is None:
                add_tensor_transforms = []

            self.transform = torchvision.transforms.Compose([*add_PIL_transforms, torchvision.transforms.ToTensor(), *add_tensor_transforms])

            self.name_generator = name_generator
            self.additional_transform = add_PIL_transforms
            self.num_image_calculate_mean_std = num_image_calculate_mean_std
            self.num_classes = len(self.classes)

            compute_stats = False

            if isinstance(stats, dict):
                self.stats = stats
                print(fg.red + f"Using precomputed stats: " + fg.cyan + f"mean = {self.stats['mean']}, std = {self.stats['std']}" + rs.fg)

            elif isinstance(stats, str):
                if os.path.isfile(stats):
                    self.stats = pickle.load(open(stats, 'rb'))
                    print(fg.red + f"Using stats from file [{Path(stats).name}]: " + fg.cyan + f"mean = {self.stats['mean']}, std = {self.stats['std']}" + rs.fg)
                    if stats == save_stats_file:
                        save_stats_file = None
                else:
                    print(fg.red + f"File [{Path(stats).name}] not found, stats will be computed." + rs.fg)
                    compute_stats = True

            if stats is None or compute_stats is True:
                self.stats = self.call_compute_stats()

            if save_stats_file is not None:
                print(f"Stats saved in {save_stats_file}")
                pathlib.Path(os.path.dirname(save_stats_file)).mkdir(parents=True, exist_ok=True)
                pickle.dump(self.stats, open(save_stats_file, 'wb'))

            normalize = torchvision.transforms.Normalize(mean=self.stats['mean'],
                                                         std=self.stats['std'])
            # self.stats = {}
            # self.stats['mean'] = [0.491, 0.482, 0.44]
            # self.stats['std'] = [0.247, 0.243, 0.262]
            # normalize = torchvision.transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

            self.transform.transforms += [normalize]

            print(f'Map class_name -> labels: \n {self.class_to_idx}\n{len(self)} samples.')
            self.finalize_init()

        def finalize_init(self):
            pass

        def call_compute_stats(self):
            return compute_mean_and_std_from_dataset(self, None, max_iteration=self.num_image_calculate_mean_std)

        def __getitem__(self, idx, class_name=None):
            image, *rest = super().__getitem__(idx)
            label = rest[0]
            return image, label, rest[1] if len(rest) > 1 else {}

    return ComputeStatsUpdateTransform


class SubsetImageFolder(MyImageFolder):
    def __init__(self, num_images_per_class=None, save_load_samples_filename=None, **kwargs):
        super().__init__(**kwargs)
        loaded = False
        if num_images_per_class:
            if save_load_samples_filename is not None:
                if os.path.isfile(save_load_samples_filename):
                    print(fg.red + f"LOADING SAMPLES FROM /samples/{Path(save_load_samples_filename).name}" + rs.fg)
                    self.samples = pickle.load(open(save_load_samples_filename, 'rb'))
                    loaded = True
                else:
                    print(fg.yellow + f"Path {save_load_samples_filename} not found, will compute samples" + rs.fg)

            if not loaded:
                subset_samples = []
                for c in range(len(self.classes)):
                    objs_class = [i for i in self.samples if i[1] == c]
                    subset_samples.extend([objs_class[i] for i in np.random.choice(len(objs_class), num_images_per_class, replace=False)])
                self.samples = subset_samples

            if save_load_samples_filename is not None and loaded is False:
                print(fg.yellow + f"SAVING SAMPLES IN /samples/{Path(save_load_samples_filename).name}" + rs.fg)
                pathlib.Path(Path(save_load_samples_filename).parent).mkdir(parents=True, exist_ok=True)
                pickle.dump(self.samples, open(save_load_samples_filename, 'wb'))


class Stats(ImageStat.Stat):
    def __add__(self, other):
        return Stats(list(map(np.add, self.h, other.h)))


def compute_mean_and_std_from_dataset(dataset, dataset_path=None, max_iteration=100, data_loader=None, verbose=True):
    if max_iteration < 30:
        print(f'Max Iteration in Compute Mean and Std for dataset is lower than 30! This could create unrepresentative stats!') if verbose else None
    start = time()
    stats = {}
    transform_save = dataset.transform
    if data_loader is None:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    statistics = None
    c = 0
    stop = False
    for data, _, _ in data_loader:
        for b in range(data.shape[0]):
            if c % 10 == 9:
                print(f'{c}/{max_iteration}, m: {np.array(statistics.mean)/255}, std: {np.array(statistics.stddev)/255}')
            c += 1
            if statistics is None:
                statistics = Stats(tf.ToPILImage()(data[b]))
            else:
                statistics += Stats(tf.ToPILImage()(data[b]))
            if c > max_iteration:
                stop = True
                break
        if stop:
            break
    print()
    print(np.array(statistics.mean)/255)
    print(np.array(statistics.stddev) / 255)
    stats['time_one_iter'] = (time() - start)/max_iteration
    stats['mean'] = np.array(statistics.mean)/255
    stats['std'] = np.array(statistics.stddev) / 255
    stats['iter'] = max_iteration
    print((fg.cyan + 'mean: {}; std: {}, time: {}' + rs.fg).format(stats['mean'], stats['std'], stats['time_one_iter'])) if verbose else None
    if dataset_path is not None:
        print('Saving in {}'.format(dataset_path))
        with open(dataset_path, 'wb') as f:
            cloudpickle.dump(stats, f)

    dataset.transform = transform_save
    return stats