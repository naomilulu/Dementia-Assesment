#! python
# -*- coding: utf-8 -*-
# Author: kun
# @Time: 2019-10-29 20:39

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from core.util import init_weights, init_gate
from core.module import VGGExtractor, RNNLayer, ScaleDotAttention, LocationAwareAttention

class Whisper_Classifier(nn.Module):
    """
    ASR model, including Encoder/Decoder(s)
    """

    def __init__(self, ctc_weight, encoder, attention, decoder, emb_drop=0.0):
        super(Whisper_Classifier, self).__init__()
        device = torch.device('cuda:0')

        # Modules
#         self.model_path="openai/whisper-large"
#         self.whisper = WhisperModel.from_pretrained(self.model_path)
#         self.whisper = self.whisper.to(device)
#         self.whisper = self.whisper.half()
        # self.pro = nn.Linear(1280, 256)
        self.pro = nn.Linear(1024, 256)
        self.fc = nn.Linear(256, 1)
#         self.fc = self.fc.half()
        self.fc = self.fc

        # Init
        self.apply(init_weights)
#         for l in range(self.decoder.layer):
#             bias = getattr(self.decoder.layers, 'bias_ih_l{}'.format(l))
#             bias = init_gate(bias)

    def set_state(self, prev_state, prev_attn):
        ''' Setting up all memory states for beam decoding'''
        pass

    def create_msg(self):
        # Messages for user
        msg = []
#         msg.append('Model spec.| Encoder\'s downsampling rate of time axis is {}.'.format(
#             self.encoder.sample_rate))
#         if self.encoder.vgg:
#             msg.append(
#                 '           | VCC Extractor w/ time downsampling rate = 4 in encoder enabled.')
#         if self.enable_ctc:
#             msg.append('           | CTC training on encoder enabled ( lambda = {}).'.format(
#                 self.ctc_weight))
#         if self.enable_att:
#             msg.append('           | {} attention decoder enabled ( lambda = {}).'.format(
#                 self.attention.mode, 1 - self.ctc_weight))
        return msg

    def forward(self, last_hidden_state):
        '''
        Arguments
            audio_feature - [BxTxD] Acoustic feature with shape
            feature_len   - [B]     Length of each sample in a batch
            decode_step   - [int]   The maximum number of attention decoder steps
            tf_rate       - [0,1]   The probability to perform teacher forcing for each step
            teacher       - [BxLxD] Ground truth for teacher forcing with sentence length L
            emb_decoder   - [obj]   Introduces the word embedding decoder, different behavior for training/inference
                                    At training stage, this ONLY affects self-sampling (output remains the same)
                                    At inference stage, this affects output to become log prob. with distribution fusion
            get_dec_state - [bool]  If true, return decoder state [BxLxD] for other purpose
        '''
        # Init
#         print(audio_feat)
#         bs = audio_feat.shape[0]


        # Encode
#         outputs = self.whisper(audio_feat[0])
#         last_hidden_state = outputs.last_hidden_state
#         print(last_hidden_state)
#         print('last_hidden_state[0][-1]',last_hidden_state[0][-1].shape)
        linear_fea = self.pro(last_hidden_state)
        linear_fea = self.fc(linear_fea.mean(dim=1))
#         print('last_hidden_state[0][-1]', torch.isfinite(last_hidden_state[0][-1]).all())
#         print('linear_fea',linear_fea)
#         linear_fea = torch.squeeze(linear_fea, 1)
        out = torch.sigmoid(linear_fea)
#         print('out',out)
#         print(torch.isnan(self.fc.weight).any())
#         print('fcweight', torch.isfinite(self.fc.weight).all())
#         print('fcweight',self.fc.weight)
#         print(torch.isnan(self.fc.bias).any())
#         print('fcbias',self.fc.bias)

        return out


class Whisper_Regression(nn.Module):
    """
    ASR model, including Encoder/Decoder(s)
    """

    def __init__(self, ctc_weight, encoder, attention, decoder, emb_drop=0.0):
        super(Whisper_Regression, self).__init__()
        device = torch.device('cuda:0')

        # Modules
#         self.model_path="openai/whisper-large"
#         self.whisper = WhisperModel.from_pretrained(self.model_path)
#         self.whisper = self.whisper.to(device)
#         self.whisper = self.whisper.half()
        self.pro = nn.Linear(1280, 256)
        self.fc = nn.Linear(256, 1)
#         self.fc = self.fc.half()
        self.fc = self.fc

        # Init
        self.apply(init_weights)
#         for l in range(self.decoder.layer):
#             bias = getattr(self.decoder.layers, 'bias_ih_l{}'.format(l))
#             bias = init_gate(bias)

    def set_state(self, prev_state, prev_attn):
        ''' Setting up all memory states for beam decoding'''
        pass

    def create_msg(self):
        # Messages for user
        msg = []
#         msg.append('Model spec.| Encoder\'s downsampling rate of time axis is {}.'.format(
#             self.encoder.sample_rate))
#         if self.encoder.vgg:
#             msg.append(
#                 '           | VCC Extractor w/ time downsampling rate = 4 in encoder enabled.')
#         if self.enable_ctc:
#             msg.append('           | CTC training on encoder enabled ( lambda = {}).'.format(
#                 self.ctc_weight))
#         if self.enable_att:
#             msg.append('           | {} attention decoder enabled ( lambda = {}).'.format(
#                 self.attention.mode, 1 - self.ctc_weight))
        return msg

    def forward(self, last_hidden_state):
        '''
        Arguments
            audio_feature - [BxTxD] Acoustic feature with shape
            feature_len   - [B]     Length of each sample in a batch
            decode_step   - [int]   The maximum number of attention decoder steps
            tf_rate       - [0,1]   The probability to perform teacher forcing for each step
            teacher       - [BxLxD] Ground truth for teacher forcing with sentence length L
            emb_decoder   - [obj]   Introduces the word embedding decoder, different behavior for training/inference
                                    At training stage, this ONLY affects self-sampling (output remains the same)
                                    At inference stage, this affects output to become log prob. with distribution fusion
            get_dec_state - [bool]  If true, return decoder state [BxLxD] for other purpose
        '''
        # Init
#         print(audio_feat)
#         bs = audio_feat.shape[0]


        # Encode
#         outputs = self.whisper(audio_feat[0])
#         last_hidden_state = outputs.last_hidden_state
#         print(last_hidden_state)
#         print('last_hidden_state[0][-1]',last_hidden_state[0][-1].shape)
        linear_fea = self.pro(last_hidden_state)
        out = self.fc(linear_fea.mean(dim=1))
#         print('last_hidden_state[0][-1]', torch.isfinite(last_hidden_state[0][-1]).all())
#         print('linear_fea',linear_fea)
#         linear_fea = torch.squeeze(linear_fea, 1)
        # out = torch.sigmoid(linear_fea)
#         print('out',out)
#         print(torch.isnan(self.fc.weight).any())
#         print('fcweight', torch.isfinite(self.fc.weight).all())
#         print('fcweight',self.fc.weight)
#         print(torch.isnan(self.fc.bias).any())
#         print('fcbias',self.fc.bias)

        return out


class Combine_Classifier(nn.Module):
    """
    ASR model, including Encoder/Decoder(s)
    """

    def __init__(self, ctc_weight, encoder, attention, decoder, emb_drop=0.0):
        super(Combine_Classifier, self).__init__()
        device = torch.device('cuda:0')

        # Modules
#         self.model_path="openai/whisper-large"
#         self.whisper = WhisperModel.from_pretrained(self.model_path)
#         self.whisper = self.whisper.to(device)
#         self.whisper = self.whisper.half()
        self.pro = nn.Linear(1280, 256)
        self.pro2 = nn.Linear(768, 256)
        self.fc = nn.Linear(256, 1)
#         self.fc = self.fc.half()
        self.fc = self.fc

        # Init
        self.apply(init_weights)
#         for l in range(self.decoder.layer):
#             bias = getattr(self.decoder.layers, 'bias_ih_l{}'.format(l))
#             bias = init_gate(bias)

    def set_state(self, prev_state, prev_attn):
        ''' Setting up all memory states for beam decoding'''
        pass

    def create_msg(self):
        # Messages for user
        msg = []
#         msg.append('Model spec.| Encoder\'s downsampling rate of time axis is {}.'.format(
#             self.encoder.sample_rate))
#         if self.encoder.vgg:
#             msg.append(
#                 '           | VCC Extractor w/ time downsampling rate = 4 in encoder enabled.')
#         if self.enable_ctc:
#             msg.append('           | CTC training on encoder enabled ( lambda = {}).'.format(
#                 self.ctc_weight))
#         if self.enable_att:
#             msg.append('           | {} attention decoder enabled ( lambda = {}).'.format(
#                 self.attention.mode, 1 - self.ctc_weight))
        return msg

    def forward(self, acoustic_features, linguistic_features):
        '''
        Arguments
            audio_feature - [BxTxD] Acoustic feature with shape
            feature_len   - [B]     Length of each sample in a batch
            decode_step   - [int]   The maximum number of attention decoder steps
            tf_rate       - [0,1]   The probability to perform teacher forcing for each step
            teacher       - [BxLxD] Ground truth for teacher forcing with sentence length L
            emb_decoder   - [obj]   Introduces the word embedding decoder, different behavior for training/inference
                                    At training stage, this ONLY affects self-sampling (output remains the same)
                                    At inference stage, this affects output to become log prob. with distribution fusion
            get_dec_state - [bool]  If true, return decoder state [BxLxD] for other purpose
        '''
        # Init
#         print(audio_feat)
#         bs = audio_feat.shape[0]


        # Encode
#         outputs = self.whisper(audio_feat[0])
#         last_hidden_state = outputs.last_hidden_state
#         print(last_hidden_state)
#         print('last_hidden_state[0][-1]',last_hidden_state[0][-1].shape)
        linear_fea1 = self.pro(acoustic_features)
        linear_fea2 = self.pro2(linguistic_features)
        # print(linear_fea1.shape, linear_fea2.shape)
        combine = torch.cat((linear_fea1, linear_fea2), 1)
        linear_fea = self.fc(combine.mean(dim=1))
#         print('last_hidden_state[0][-1]', torch.isfinite(last_hidden_state[0][-1]).all())
#         print('linear_fea',linear_fea)
#         linear_fea = torch.squeeze(linear_fea, 1)
        out = torch.sigmoid(linear_fea)
#         print('out',out)
#         print(torch.isnan(self.fc.weight).any())
#         print('fcweight', torch.isfinite(self.fc.weight).all())
#         print('fcweight',self.fc.weight)
#         print(torch.isnan(self.fc.bias).any())
#         print('fcbias',self.fc.bias)

        return out

