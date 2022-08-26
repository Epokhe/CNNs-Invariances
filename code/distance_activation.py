import torch
from framework_utils import make_cuda
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum

class CompareWith(Enum):
    ANY_OBJECT = 0
    SAME_OBJECT = 1
    SAME_CLASS_DIFF_OBJ = 2


class DistanceActivation(ABC):
    def __init__(self, net, dataset=None, distance='cossim', use_cuda=None, compare_with=CompareWith.SAME_OBJECT):
        self.cuda = False
        self.distance = distance
        if use_cuda is None:
            if torch.cuda.is_available():
                self.cuda = True
            else:
                self.cuda = False
        else:
            self.cuda = use_cuda
        self.net = net
        self.dataset = dataset
        self.only_save = ['Conv2d', 'Linear']
        self.detach_this_step = True
        self.activation = {}
        self.last_linear_layer = ''
        self.all_layers_name = []
        self.compare_with = compare_with
        self.setup_network()

    @abstractmethod
    def get_base_and_other_canvasses(self, class_num, name_class):
        raise NotImplementedError

    def finalize_each_class(self, name_class, cossim, cossim_imgs,  x_values):
        """
        Use for plotting, analysis, etc.
        """
        pass;

    def get_activation(self, name):
        def hook(model, input, output):
            if self.detach_this_step:
                self.activation[name] = output.detach()

        return hook

    def group_all_layers(self):
        all_layers = []

        def recursive_group(net):
            for layer in net.children():
                if not list(layer.children()):  # if leaf node, add it to list
                    all_layers.append(layer)
                else:
                    recursive_group(layer)

        recursive_group(self.net)
        return all_layers

    def compute_distance(self, a, b):
        if self.distance == 'cossim':
            return torch.nn.CosineSimilarity(dim=0)(a.flatten(), b.flatten()).item()
        if self.distance == 'euclidean':
            diff =a.flatten() - b.flatten()
            return torch.sqrt(torch.dot(diff, diff)).item()

    def get_cosine_similarity_from_images(self, base_canvas, other_canvasses):
        prediction_base = torch.argmax(self.net(make_cuda(base_canvas.unsqueeze(0), self.cuda))).item()
        base_activation = {}
        # base_activation = activation['one_to_last']
        # i= 0
        all_used_layers = [] # this may be useful in case of splitting networks
        for name, features in self.activation.items():
            if not np.any([i in name for i in self.only_save]):
                continue
            base_activation[name] = features
            # all_used_layers.append(name)
            # if not self.all_layers_name[i] == name:
            #     stop=1
            # print(self.all_layers_name[i] == name)
            # i+=1
        # print(i)
        # cos_fun = torch.nn.CosineSimilarity(dim=1)
        distance_net = {}
        predictions = []
        for canvas_comparison in other_canvasses:
            canvas_comparison_activation = {}
            predictions.append(torch.argmax(self.net(make_cuda(canvas_comparison.unsqueeze(0), self.cuda))).item())
            for name, features in self.activation.items():
                if not np.any([i in name for i in self.only_save]):
                    continue
                canvas_comparison_activation[name] = features
                if name not in distance_net:
                    distance_net[name] = []
                distance_net[name].append(self.compute_distance(base_activation[name], canvas_comparison_activation[name]))

        distance_image = [self.compute_distance(base_canvas, c) for c in other_canvasses]

        return distance_net, distance_image, (prediction_base, predictions)

    @staticmethod
    def get_average_cossim_across_classes_and_values(cossim_net):
        all_layers = list(cossim_net[0].keys())

        # global cossim across all values for each layer
        mean = [np.mean(np.array([v[l] for k, v in cossim_net.items()])) for l in all_layers]
        std = [np.std(np.array([v[l] for k, v in cossim_net.items()])) for l in all_layers]
        return mean, std, all_layers

    @staticmethod
    def get_average_cossim_across_classes(cossim_net):
        if isinstance(cossim_net[0], dict):

            all_layers = list(cossim_net[0].keys())
            mean = [np.mean(np.array([v[l] for k, v in cossim_net.items()]), axis=0) for l in all_layers]
            std = [np.std(np.array([v[l] for k, v in cossim_net.items()]), axis=0) for l in all_layers]
            return mean, std, all_layers
        else:
            mean = np.mean(np.array([v for k, v in cossim_net.items()]), axis=0)
            std = np.std(np.array([v for k, v in cossim_net.items()]), axis=0)
            return mean, std, None

    def get_cosine_similarity_one_class_random_img(self, class_num, dataset):
        """
        cossim: [group class idx][name feature][list of cossim for each step]
        """

        name_class = dataset.idx_to_class[class_num]

        base_canvas, other_canvasses, x_values = self.get_base_and_other_canvasses(class_num, name_class)
        cossim_net, cossim_images,  p = self.get_cosine_similarity_from_images(base_canvas, other_canvasses)
        self.finalize_each_class(name_class, cossim_net, cossim_images, x_values)
        return cossim_net, cossim_images, x_values

    def setup_network(self):
        self.was_train = self.net.training
        self.net.eval()
        all_layers = self.group_all_layers()
        self.hook_lists = []
        for idx, i in enumerate(all_layers):
            name = '{}: {}'.format(idx, str.split(str(i), '(')[0])
            self.hook_lists.append(i.register_forward_hook(self.get_activation(name)))
            if np.any([i in name for i in self.only_save]):
                self.all_layers_name.append(name)
        self.last_linear_layer = self.all_layers_name[-1]

    def remove_hooks(self):
        for h in self.hook_lists:
            h.remove()
        if self.was_train:
            self.net.train()

    def calculate_distance_dataloader(self, dataset=None):
        # self.setup_network()
        if dataset is None:
            dataset = self.dataset
        else:
            self.dataset = dataset
        cossim_net = {}
        cossim_img = {}
        for c in range(dataset.num_classes):
            cossim_net[c], cossim_img[c], x_values = self.get_cosine_similarity_one_class_random_img(c, dataset)

        return cossim_net, cossim_img, x_values
