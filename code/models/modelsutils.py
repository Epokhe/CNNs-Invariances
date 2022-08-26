class TypeNet(Enum):
    VGG = 0
    FC = 1
    RESNET = 2
    SMALL_CNN = 3
    OTHER = 4


def get_net(self, new_num_classes=None):
    net, params_to_update = self.prepare_network(network=self.network_name,
                                                 new_num_classes=new_num_classes,
                                                 is_server=self.use_cuda,
                                                 grayscale=self.grayscale,
                                                 use_gap=self.use_gap,
                                                 feature_extraction=self.feature_extraction,
                                                 pretraining=self.pretraining,
                                                 big_canvas=self.big_canvas,
                                                 shallow_FC=self.shallow_FC,
                                                 freeze_fc=self.freeze_fc,
                                                 scramble_fc=self.scramble_fc,
                                                 scramble_conv=self.scramble_conv)
    return net, params_to_update
import torchvision
@staticmethod
def get_network_structure(network, pretrain_ImageNet=False):
    if network == 'vgg11_bn':
        network = torchvision.models.vgg11_bn(pretrained=pretrain_ImageNet, progress=True)
    elif network == 'vgg11':
        network = torchvision.models.vgg11(pretrained=pretrain_ImageNet, progress=True)
    elif network == 'vgg16_bn':
        network = torchvision.models.vgg16_bn(pretrained=pretrain_ImageNet, progress=True)
    elif network == 'vgg16':
        network = torchvision.models.vgg16(pretrained=pretrain_ImageNet, progress=True)
    elif network == 'smallCNNnopool':
        if pretrain_ImageNet:
            assert False, f"No pretraining ImageNet for {network}"
        network = smallCNNnp()
    elif network == 'smallCNNpool':
        if pretrain_ImageNet:
            assert False, f"No pretraining ImageNet for {network}"

        network = smallCNNp()
    elif network == 'resnet18':
        network = torchvision.models.resnet18(pretrained=pretrain_ImageNet, progress=True)
    elif network == 'FC4':  # 2500, 2000, 1500, 1000, 500, 10
        if pretrain_ImageNet:
            assert False, f"No pretraining ImageNet for {network}"

        network = FC4()
    elif not isinstance(network, torch.nn.Module):
        assert False, 'network is neither a recognised neural network name, nor a nn.Module'

    return network


@classmethod
def change_net_structure(cls, network, type_net, change_to_grayscale=False, new_num_classes=None, shallow_FC=False, use_gap=False, use_big_canvas=False):
    changed_str = ''
    if change_to_grayscale:
        network.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
        changed_str += "\tInput is now in grayscale (1 channel)\n"

    resize_output = True
    if new_num_classes is None:
        new_num_classes == network.classifier[-1].out_features
        resize_output = False
    elif new_num_classes == network.classifier[-1].out_features:
        resize_output = False

    if resize_output is False and (shallow_FC or use_gap or use_big_canvas):
        changed_str += "\tResize Output is False (same number of output units), however (shallow_FC or use_gap or use_big_canvas) is True. The output weights will be lost!\n"
        resize_output = network.classifier[-1].out_features

    if type_net == TypeNet.VGG:

        if use_gap:
            changed_str += '\tPretraing model has a GAP!\n'
            network.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            if shallow_FC:
                changed_str += '\tPretraing model has a shallow FC!\n'
                changed_str += f"\tChanged output to {new_num_classes}\n"
                network.classifier = torch.nn.Linear(512, new_num_classes)
            else:
                network.classifier[0] = torch.nn.Linear(512, 4096)
        else:
            if shallow_FC:
                changed_str += '\tPretraing model has a shallow FC!\n'
                changed_str += f"\tChanged output to {new_num_classes}\n"
                network.classifier = torch.nn.Linear(512 * 7 * 7, new_num_classes)

            elif resize_output:
                changed_str += f"\tChanged output to {new_num_classes}\n"
                network.classifier[-1] = torch.nn.Linear(network.classifier[-1].in_features, new_num_classes)

        if use_big_canvas and not use_gap:
            # if big_canvas, input is 400x400, and to obtain an AdaptiveAvgPool that does not
            # do anything, we need a size of 12, 12
            # ToDo: why not using a bigcanvas of 448 so everything is just double?!
            network.avgpool = torch.nn.AdaptiveAvgPool2d((12, 12))
            changed_str += '\tPretraing model is using big canvas!\n'
            if shallow_FC:
                changed_str += '\tPretraing model has a shallow FC!\n'
                changed_str += f"\tChanged output to {new_num_classes}\n"
                network.classifier = torch.nn.Linear(512 * 12 * 12, new_num_classes)
            else:
                network.classifier[0] = torch.nn.Linear(512 * 12 * 12, 4096)

    if type_net == TypeNet.SMALL_CNN:
        if resize_output:
            changed_str += f"\tChanged output to {new_num_classes}\n"
            network.classifier[-1] = torch.nn.Linear(network.classifier[-1].in_features, new_num_classes)
            if shallow_FC or use_gap or use_big_canvas:
                assert False, "Parameters shallow_FC, use_gap, or use_big_canvas not implemented for SMALL_CNN net"

    if type_net == TypeNet.RESNET:
        if resize_output:
            network.fc = torch.nn.Linear(network.fc.in_features, new_num_classes)
            changed_str += f"\tChanged output to {new_num_classes}\n"
            if shallow_FC or use_gap or use_big_canvas:
                assert False, "Parameters shallow_FC, use_gap, or use_big_canvas not implemented for RESENT net"

    if type_net == TypeNet.FC:
        if resize_output:
            changed_str += f"\tChanged output to {new_num_classes}\n"
            network.classifier[-1] = torch.nn.Linear(network.classifier[-1].in_features, new_num_classes)
            if shallow_FC or use_gap or use_big_canvas:
                assert False, "Parameters shallow_FC, use_gap, or use_big_canvas not implemented for FC net"

    return network, changed_str


@classmethod
def prepare_load(cls, network, pretrain_path, type_net):
    try:
        find_output = int(re.findall('\d+', re.findall(r'o\d+', pretrain_path)[0])[0])
    except:
        print(f'Some problems when checking the output class num for path: {pretrain_path}')
        assert False
    find_gray = 'gray' in pretrain_path
    find_gap = 'gap1' in pretrain_path
    find_sFC = 'sFC1' in pretrain_path
    find_bC = 'bC1' in pretrain_path
    isnp = re.findall(r'nopool', pretrain_path)
    find_NP = bool(isnp[0]) if isnp else False
    network, change_str = cls.change_net_structure(network, type_net, find_gray, find_output, find_sFC, find_gap, find_bC)
    print("Preparing network before loading...")
    if change_str == '':
        print("Nothing has changed")
    else:
        print(change_str)
    return network


@staticmethod
def freeze_fully_connected(network):
    for param in network.classifier.parameters():
        param.requires_grad = False
    params_to_update = []
    for param in network.parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    return params_to_update


@staticmethod
def feature_extraction_method(network, feature_extraction):
    for param in network.features.parameters():
        param.requires_grad = False
    params_to_update = []
    for param in network.parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    return params_to_update


@classmethod
def prepare_network(cls, network, new_num_classes=None, is_server=False, grayscale=False, use_gap=False, feature_extraction=False, pretraining='vanilla', big_canvas=False, shallow_FC=False, verbose=True, freeze_fc=False, scramble_fc=False, scramble_conv=False):
    """
    @param network: this can be a string such as 'vgg16' or a torch.nn.Module
    @param pretraining: can be [vanilla], [ImageNet] or the path
    @return:
    """

    if isinstance(network, str):
        if 'vgg' in network:
            type_net = TypeNet.VGG
        elif 'FC' in network:
            type_net = TypeNet.FC
        elif 'resnet' in network:
            type_net = TypeNet.RESNET
        elif 'smallCNN' in network:
            type_net = TypeNet.SMALL_CNN
    else:
        type_net = TypeNet.OTHER

    if feature_extraction and pretraining == 'vanilla':
        assert False, 'You selected feature extraction, but you have a vanilla net! Someone gotta train those convolutions man!'

    if pretraining != 'vanilla' and not isinstance(network, str):
        assert False, "With a pretrained network, specify the network structure as a string in 'network'"

    if pretraining != 'vanilla' and big_canvas:
        Warning('If you use a big canvas, you will lose the pretraining on the first FC layers!')

    if grayscale and pretraining != 'vanilla':
        assert False, 'Cannot use pretrained network and grayscale image - you would lose the pretraining on the first conv layer'

    pretrain_path = None
    pretrain_ImageNet = False
    if pretraining != 'vanilla':
        if pretraining == 'ImageNet':
            pretrain_ImageNet = True
        else:
            pretrain_path = pretraining

    network = cls.get_network_structure(network, pretrain_ImageNet=pretrain_ImageNet)

    # Remember that the network structure just created may have a different num_classes than the pretrained state dict we are loading here. If this is the case, call prepare_load and adjust the structure accordingly to match pretrained state dict. Then in model surgery we'll put the num_classes back.
    if pretrain_path is not None:
        network = cls.prepare_load(network, pretrain_path, type_net)
        if verbose:
            print('Loaded model: {}'.format(pretrain_path))
        loaded_state_dict = torch.load(pretrain_path, map_location=torch.device('cuda' if is_server else 'cpu'))
        print('**Loading these parameters from pretrained network:') if verbose else None
        if scramble_fc:
            own_state = network.state_dict()
            if type_net == TypeNet.VGG:
                for name, param in loaded_state_dict.items():
                    if 'classifier' not in name:
                        print(name) if verbose else None
                        own_state[name].copy_(param)
                print('Scramble FC is ON: Fully Connected layer NOT copied when loading pretraining params') if verbose else None

            else:
                assert False, f"Scramble_fc not implemented for [{type_net}]"
        elif scramble_conv:
            own_state = network.state_dict()
            if type_net == TypeNet.VGG:
                for name, param in loaded_state_dict.items():
                    if 'features' not in name:
                        print(name) if verbose else None
                        own_state[name].copy_(param)
                print('Scramble conv is ON: Conv. layer NOT copied when loading pretraining params') if verbose else None
            else:
                assert False, f"Scramble_fc not implemented for [{type_net}]"
        else:
            print('ALL PARAMETERS') if verbose else None
            network.load_state_dict(loaded_state_dict)
    print('***')

    network, change_str = cls.change_net_structure(network, type_net, grayscale, new_num_classes, shallow_FC, use_gap, big_canvas)

    network.train_step = standard_net_step

    if feature_extraction and freeze_fc:
        assert False, 'Both feature extraction and freeze_fc are on - the network won''t learn anything!'
    if feature_extraction and (type_net == TypeNet.VGG or type_net == TypeNet.FC or type_net == TypeNet.RESNET or type_net == TypeNet.SMALL_CNN):
        params_to_update = cls.feature_extraction_method(network, feature_extraction)
    elif freeze_fc and type_net == TypeNet.VGG:
        params_to_update = cls.freeze_fully_connected(network)
    else:
        params_to_update = network.parameters()

    for m in network.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.track_running_stats = False
    # m.momentum = 0.4



    print(f"Network structure {'after loading state_dict' if pretraining != 'vanilla' else ''}: ", end="")
    if change_str == '':
        print('Nothing has changed')
    else:
        print("Changed. List of changes:")
        print(change_str)

    return network, params_to_update
