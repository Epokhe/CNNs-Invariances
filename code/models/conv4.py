import torch
import torch.nn as nn
import collections
import math
from models.model_utils import conv_block
class Conv4(torch.nn.Module):
    def __init__(self, flatten=True, track_running_stats=True):
        super(Conv4, self).__init__()
        self.feature_size = 64
        self.name = "conv4"

        self.conv4 = nn.Sequential(conv_block(3, 8, max_pool=False, batch_norm=True, act_fun='ReLU', bias=False, avg_pool=True, track_running_stats=track_running_stats),
                                   conv_block(8, 16, max_pool=False, batch_norm=True, act_fun='ReLU', bias=False, avg_pool=True, track_running_stats=track_running_stats),
                                   conv_block(16, 32, max_pool=False, batch_norm=True, act_fun='ReLU', bias=False, avg_pool=True, track_running_stats=track_running_stats),
                                   conv_block(32, 64, max_pool=False, batch_norm=True, act_fun='ReLU', bias=False, avg_pool=True, track_running_stats=track_running_stats),
                                   conv_block(64, 128, max_pool=False, batch_norm=True, act_fun='ReLU', bias=False, avg_pool=False, track_running_stats=track_running_stats))
                                   # conv_block(128, 128, max_pool=False, batch_norm=True, act_fun='ReLU', bias=False, avg_pool=False, track_running_stats=track_running_stats),

        self.is_flatten = flatten
        self.flatten = nn.Flatten()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        h = self.conv4(x)
        if(self.is_flatten): h = self.flatten(h)
        return h