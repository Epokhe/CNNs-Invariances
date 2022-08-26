
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def conv_block(in_channels: int, out_channels: int, kernel_size=3, max_pool=True,
               batch_norm=True, act_fun='ReLU', bias=True, avg_pool=False, track_running_stats=True, bn_momentum=0.1) -> nn.Module:
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.

    # Arguments
        in_channels:
        out_channels:
    size -> size - (kernel_size + 1) / 2 (if padding is 0)
    size -> size / 2 (because padding is 1)
    """

    module = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=bias)]
    module.append(nn.BatchNorm2d(out_channels, track_running_stats=track_running_stats, momentum=bn_momentum)) if batch_norm else None
    if act_fun == 'ReLU':
        module.append(nn.ReLU())
    elif act_fun == 'Sigmoid':
        module.append(nn.Sigmoid())
    elif act_fun == 'Tanh':
        module.append(nn.Tanh())
    else:
        assert False, f"Activation Function {act_fun} not recognized"
    module.append(nn.MaxPool2d(kernel_size=2, stride=2)) if max_pool else None
    module.append(nn.AvgPool2d(kernel_size=2, stride=2)) if avg_pool else None
    return nn.Sequential(*module)


