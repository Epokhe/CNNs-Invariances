import torch.nn as nn
from typing import Any

import torch
from models.model_utils import Flatten, conv_block
import torch.nn.functional as F
import numpy as np
from framework_utils import make_cuda


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, input, hidden=None):
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class SequenceNtrain1cand(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def __init__(self, grayscale=False):
        super().__init__()
        size_embedding_frame = 64
        # self.image_embedding_trainings = nn.Sequential(conv_block(3, 64, bias=False, batch_norm=False),
        #                                      conv_block(64, 64, bias=False, batch_norm=False),
        #                                      conv_block(64, 64, bias=False, batch_norm=False),
        #                                      conv_block(64, 64, bias=False, batch_norm=False),
        #                                      nn.AdaptiveAvgPool2d(1),
        #                                      Flatten())  # output = 1 x 64
        self.image_embedding_candidates = nn.Sequential(conv_block(1 if grayscale else 3, 64),
                                                      conv_block(64, 64),
                                                      conv_block(64, 64),
                                                      conv_block(64, 64),
                                                      nn.AdaptiveAvgPool2d(1),
                                                      Flatten())  # output = 1 x 64

        hidden_size = 64
        size_embedding_output = 256
        self.encoder_fr2seq = EncoderRNN(input_size=64, hidden_size=hidden_size)
        # you could put another linear layer that transforms the hidden output from the first encoder to something for the second one
        # self.encoder_seq2obj = EncoderRNN(input_size=hidden_size, hidden_size=hidden_size)

        self.relation_net = nn.Sequential(nn.Linear(64, 256),
                                          nn.ReLU(True),
                                          nn.BatchNorm1d(256),
                                          nn.Linear(256, 8),
                                          nn.ReLU(True),
                                          nn.Linear(8, 1),
                                          nn.Sigmoid())

        self._initialize_weights()

    def forward(self, input):
        x, k, nSt, nFt, use_cuda = input
        fr_hidden = make_cuda(torch.randn(1, k, self.encoder_fr2seq.hidden_size), use_cuda)

        emb_all = self.image_embedding_candidates(x)
        emb_candidates = emb_all[:k]  # this only works for nSc and nFc = 1!
        emb_trainings = emb_all[k:]
        emb_sequence_batch = emb_trainings.reshape(k, nSt * nFt, 64)

        out, h = self.encoder_fr2seq(emb_sequence_batch, fr_hidden)

        relation_scores = self.relation_net(torch.abs(emb_candidates - h.squeeze(0)))
        return relation_scores


class SequenceMatchingNetSimple(nn.Module):
    # Very simple version. We compute an embedding of the image, feed it into an RNN, take the last output hidden activation
    # feed it into a linear classifier to get a class
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def __init__(self):
        super().__init__()
        size_embedding_frame = 64
        self.image_embedding = nn.Sequential(conv_block(3, 64, bias=False),
                                             conv_block(64, 64, bias=False),
                                             conv_block(64, 64, bias=False),
                                             conv_block(64, 64, bias=False),
                                             nn.AdaptiveAvgPool2d(1),
                                             Flatten())  # output = 1 x 64

        hidden_size = 64
        size_embedding_output = 256
        self.encoder_fr2seq = EncoderRNN(input_size=64, hidden_size=hidden_size)
        # you could put another linear layer that transforms the hidden output from the first encoder to something for the second one
        self.encoder_seq2obj = EncoderRNN(input_size=hidden_size, hidden_size=hidden_size)

        self.relation_net = nn.Sequential(nn.Linear(64, 256),
                                          nn.BatchNorm1d(256),
                                          nn.LeakyReLU(True),
                                          nn.Linear(256, 1),
                                          nn.Sigmoid())

        self._initialize_weights()
    def forward(self, frame, hidden):
        pass

##

