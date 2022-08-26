from torchvision.models.vgg import VGG, make_layers
import torch.nn as nn
import torch

def smallCNNnp(**kwargs):
    # cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    cfg = [64, 64, 128, 64, 8]  # notice how we change the last filter num to fit model in memory!
    kwargs['init_weights'] = True
    model = smallCNN_NoPooling(make_layers(cfg, batch_norm=False), **kwargs)
    return model

def smallCNNp(**kwargs):
    # cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    cfg = [64, 'M', 64, 'M', 128, 'M', 64, 'M', 8]  # notice how we change the last filter num to fit model in memory!
    kwargs['init_weights'] = True
    model = smallCNN_NoPooling(make_layers(cfg, batch_norm=False), **kwargs)
    return model


class smallCNN_pooling(VGG):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(8 * 16 * 16, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class smallCNN_NoPooling(VGG):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(8 * 224 * 224, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
