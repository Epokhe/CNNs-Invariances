from typing import Any

import torch.nn as nn
import torch
from models.model_utils import Flatten, conv_block
import torch.nn.functional as F
import numpy as np
from framework_utils import make_cuda
from models.conv4 import Conv4


def get_few_shot_encoder_basic(num_input_channels=1) -> nn.Module:
    """Creates a few shot encoder. This is the simple version used in the paper.
    """
    return nn.Sequential(
        conv_block(num_input_channels, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        Flatten(),
    )

def get_few_shot_evaluator(input_channels, output):
    return nn.Sequential(
        # conv_block(input_channels, 64),
        # Flatten(),
        nn.Linear(input_channels, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(256, output),
    )


def get_few_shot_encoder(num_input_channels=1, output=64, flatten=True) -> nn.Module:
    """Creates a few shot encoder as used in Matching and Prototypical Networks

    # Arguments:
        num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
            miniImageNet = 3
    For Omniglot (input 28x28) the output of the max pooling is 64x1x1, so for 224 images should be 64x8x8
    """

    modules = [conv_block(num_input_channels, 64),
               conv_block(64, 64),
               conv_block(64, 128)]
    modules.extend([conv_block(128, 256),
                    conv_block(256, 512),
                    conv_block(512, 512),
                    Flatten()]
                   if flatten else
                   [conv_block(128, output)])
    return nn.Sequential(*modules)


class RelationNetSung(nn.Module):
    def __init__(self, backbone, output_encoder=64, size_canvas=(224, 224), n_shots=1, grayscale=False, flatten=True):
        super().__init__()

        self.backbone = backbone

        if flatten:
            self.backbone.add_module('flatten', Flatten());

        self.relation_module = nn.Sequential(nn.Linear(128 * 2, 256),  # (imagesize=128/ 2^4) * 64 = 4096
                                             nn.BatchNorm1d(256),
                                             nn.LeakyReLU(),
                                             nn.Linear(256, 1),
                                             nn.Sigmoid())

    def forward(self, input):
        ## ToDo: Make it more general, or at least for training frames/sequence > 1!
        x, k, nSt, nFt, use_cuda = input
        emb_all = self.backbone(x)
        emb_all = nn.AdaptiveAvgPool2d(1)(emb_all)

        emb_candidates = emb_all[:k]  # this only works for nSt, nFt, nSc and nFc = 1!
        emb_trainings = emb_all[k:]
        return self.relation_module(torch.cat((emb_candidates, emb_trainings), dim=1))


class MatchingNetwork(nn.Module):
    def __init__(self, n: int, k: int, q: int, fce: bool, num_input_channels: int,
                 lstm_layers: int, lstm_input_size: int, unrolling_steps: int, device: torch.device, encoder=get_few_shot_encoder_basic):
        """Creates a Matching Network as described in Vinyals et al.

        # Arguments:
            n: Number of examples per class in the support set
            k: Number of classes in the few shot classification task
            q: Number of examples per class in the query set
            fce: Whether or not to us fully conditional embeddings
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            lstm_layers: Number of LSTM layers in the bidrectional LSTM g that embeds the support set (fce = True)
            lstm_input_size: Input size for the bidirectional and Attention LSTM. This is determined by the embedding
                dimension of the few shot encoder which is in turn determined by the size of the input data. Hence we
                have Omniglot -> 64, miniImageNet -> 1600.
            unrolling_steps: Number of unrolling steps to run the Attention LSTM
            device: Device on which to run computation
        """
        super().__init__()
        self.n = n
        self.k = k
        self.q = q
        self.fce = fce
        self.num_input_channels = num_input_channels
        self.encoder = encoder(self.num_input_channels)
        if self.fce:
            self.g = BidrectionalLSTM(lstm_input_size, lstm_layers).to(device)
            self.f = AttentionLSTM(lstm_input_size, unrolling_steps=unrolling_steps).to(device)

    def forward(self, inputs):
        pass


class BidrectionalLSTM(nn.Module):
    def __init__(self, size: int, layers: int):
        """Bidirectional LSTM used to generate fully conditional embeddings (FCE) of the support set as described
        in the Matching Networks paper.

        # Arguments
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            layers: Number of LSTM layers
        """
        super(BidrectionalLSTM, self).__init__()
        self.num_layers = layers
        self.batch_size = 1
        # Force input size and hidden size to be the same in order to implement
        # the skip connection as described in Appendix A.1 and A.2 of Matching Networks
        self.lstm = nn.LSTM(input_size=size,
                            num_layers=layers,
                            hidden_size=size,
                            bidirectional=True)

    def forward(self, inputs):
        # Give None as initial state and Pytorch LSTM creates initial hidden states
        output, (hn, cn) = self.lstm(inputs, None)

        forward_output = output[:, :, :self.lstm.hidden_size]
        backward_output = output[:, :, self.lstm.hidden_size:]

        # g(x_i, S) = h_forward_i + h_backward_i + g'(x_i) as written in Appendix A.2
        # AKA A skip connection between inputs and outputs is used
        output = forward_output + backward_output + inputs
        return output, hn, cn


class AttentionLSTM(nn.Module):
    def __init__(self, size: int, unrolling_steps: int):
        """Attentional LSTM used to generate fully conditional embeddings (FCE) of the query set as described
        in the Matching Networks paper.

        # Arguments
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            unrolling_steps: Number of steps of attention over the support set to compute. Analogous to number of
                layers in a regular LSTM
        """
        super(AttentionLSTM, self).__init__()
        self.unrolling_steps = unrolling_steps
        self.lstm_cell = nn.LSTMCell(input_size=size,
                                     hidden_size=size)

    def forward(self, support, queries):
        # Get embedding dimension, d
        if support.shape[-1] != queries.shape[-1]:
            raise(ValueError("Support and query set have different embedding dimension!"))

        batch_size = queries.shape[0]
        embedding_dim = queries.shape[1]

        h_hat = torch.zeros_like(queries).cuda().double()
        c = torch.zeros(batch_size, embedding_dim).cuda().double()

        for k in range(self.unrolling_steps):
            # Calculate hidden state cf. equation (4) of appendix A.2
            h = h_hat + queries

            # Calculate softmax attentions between hidden states and support set embeddings
            # cf. equation (6) of appendix A.2
            attentions = torch.mm(h, support.t())
            attentions = attentions.softmax(dim=1)

            # Calculate readouts from support set embeddings cf. equation (5)
            readout = torch.mm(attentions, support)

            # Run LSTM cell cf. equation (3)
            # h_hat, c = self.lstm_cell(queries, (torch.cat([h, readout], dim=1), c))
            h_hat, c = self.lstm_cell(queries, (h + readout, c))

        h = h_hat + queries

        return h
