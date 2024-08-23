#! python
# -*- coding: utf-8 -*-
# Author: kun
# @Time: 2019-10-29 20:36

import torch
from core.solver import BaseSolver

from core.whisper import Whisper_Regression
from core.optim import Optimizer
from core.data import load_whisper_dataset
from core.util import human_format, cal_er, feat_to_fig

import whisper

from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    WhisperModel,
    AutoFeatureExtractor, 
    WhisperForAudioClassification
)

import tensorboardX as tbx
from torch.utils.tensorboard import SummaryWriter

import numpy as np

class Solver(BaseSolver):
    ''' Solver for training'''

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)
        # Logger settings
        self.best_wer = {'att': 3.0, 'ctc': 3.0}
        self.bestloss=100
        # Curriculum learning affects data loader
        self.curriculum = self.config['hparas']['curriculum']
        
        model_checkpoint = "sanchit-gandhi/whisper-large-v2-ft-ls-960h" # whisper-large-v3
        self.whisper = WhisperForAudioClassification.from_pretrained(model_checkpoint)
        self.whisper = self.whisper.to(self.device)

#         self.whisper = self.whisper.half()
        # self.whisper.eval()
        self.whisper.freeze_encoder()
#     def load_my_state_dict(self, state_dict):
 
#         own_state = self.state_dict()
#         for name, param in state_dict.items():
#             if name not in own_state:
#                  continue
#             if isinstance(param, Parameter):
#                 # backwards compatibility for serialized parameters
#                 param = param.data
#             own_state[name].copy_(param)
       
    def load_ckpt(self):
        """
         Load ckpt if --load option is specified
        :return:
        """
        if self.paras.load:
            # Load weights
            ckpt = torch.load(self.paras.load, map_location=self.device if self.mode == 'train' else 'cpu')
            
            #-------------------------------------
            # pretrained_dict = {k: v for k, v in ckpt.items() if k in self.model.state_dict()}
            # self.model.state_dict().update(pretrained_dict) 
            # self.model.load_state_dict(self.model.state_dict())
                
            self.model.load_state_dict(ckpt['model'])
            #-------------------------------------

            if self.emb_decoder is not None:
                self.emb_decoder.load_state_dict(ckpt['emb_decoder'])
            # if self.amp:
            #    amp.load_state_dict(ckpt['amp'])
            # Load task-dependent items
            for k, v in ckpt.items():
                if type(v) is float:
                    metric, score = k, v
            if self.mode == 'train':
                self.step = ckpt['global_step']
#                 self.optimizer.load_opt_state_dict(ckpt['optimizer'])
                self.verbose('Load ckpt from {}, restarting at step {} '.format(
                    self.paras.load, self.step))
            else:
                self.model.eval()
                if self.emb_decoder is not None:
                    self.emb_decoder.eval()
                self.verbose('Evaluation target = {} (recorded {} = {:.2f} %)'.format(
                    self.paras.load, metric, score))

    def fetch_data(self, data):
        ''' Move data to device and compute text seq. length'''
        file, audio, feat, feat_len, txt = data
        feat = feat.to(self.device)
        feat_len = feat_len.to(self.device)
        txt = txt.to(self.device)
        txt_len = torch.sum(txt != 0, dim=-1)
        # print(type(audio))
        audio = np.array(audio)

        return feat, feat_len, txt, txt_len, audio, file

    def load_data(self):
        print("Load data for training/validation, store tokenizer and input/output shape")
        self.tr_set, self.dv_set= \
            load_whisper_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory,
                         self.curriculum > 0, **self.config['data'])

    def set_model(self):
        print("Setup ASR model and optimizer ")
        nonfreeze_keys = ['fc.weight', 'fc.bias']
        # Model
        self.model = Whisper_Regression( **
        self.config['model']).to(self.device)
        self.verbose(self.model.create_msg())
        model_paras = [{'params': self.model.parameters()}]

        print("# Losses")
        self.mseloss = torch.nn.MSELoss()
        print("# Note: zero_infinity=False is unstable?")
#         self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)


        print("# Optimizer")
        self.optimizer = Optimizer(model_paras, **self.config['hparas'])
        self.verbose(self.optimizer.create_msg())

        print("# Enable AMP if needed")
        self.enable_apex()

        # Automatically load pre-trained model if self.paras.load is given
        self.load_ckpt()
        for name, para in self.model.named_parameters():
            if para.requires_grad and name not in nonfreeze_keys:
                para.requires_grad = False
        for name, para in self.model.named_parameters():
            if para.requires_grad:print(name)
        non_frozen_parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = Optimizer(non_frozen_parameters, **self.config['hparas'])

        # ToDo: other training methods

    def exec(self):
        self.model.train()
        print("Training End-to-end ASR system")
        self.verbose('Total training steps {}.'.format(
            human_format(self.max_step)))
        self.step2=0
        total_epochs = 1000
        self.max_step = 15000
        n_epochs = 0
        self.timer.set()
        
        while n_epochs < total_epochs:
            # Renew dataloader to enable random sampling
            if self.curriculum > 0 and n_epochs == self.curriculum:
                self.verbose(
                    'Curriculum learning ends after {} epochs, starting random sampling.'.format(n_epochs))
                self.tr_set, _, _, _, _, _ = \
                    load_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory,
                                 False, **self.config['data'])
            print("self.tr_set: {}".format(len(self.tr_set)))
            total_loss = 0
            for data in self.tr_set:
                # print("Pre-step : update tf_rate/lr_rate and do zero_grad")
                tf_rate = self.optimizer.pre_step(self.step)

                # Fetch data
                feat, feat_len, txt, txt_len, audio, file = self.fetch_data(data)
                self.timer.cnt('rd')

                # Forward model
                # Note: txt should NOT start w/ <sos>
                with torch.no_grad():
                    predicted_ids = self.whisper(feat[0], output_attentions=True, output_hidden_states=True)
                    last_hidden_state = predicted_ids.hidden_states[-1]
                       
                output = self.model(last_hidden_state)[0]
#                 print(output.shape)
#                 print(output)
#                 print('txt:',txt)
                loss = self.mseloss(output, txt)
                total_loss+=loss
                


                self.timer.cnt('fw')

                # Backprop
                grad_norm = self.backward(loss)
                self.step += 1
                self.step2 += 1

                # Logger
#                 if (self.step == 1) or (self.step % self.PROGRESS_STEP == 0):
#                     self.progress('Tr stat | Loss - {:.2f} | Grad. Norm - {:.2f} | {}'
#                                   .format(loss.cpu().item(), grad_norm, self.timer.show()))
#                     self.write_log(
#                         'loss', {'tr_ctc': ctc_loss, 'tr': att_loss})

                #Validation
#                 if (self.step == 1) or (self.step % self.valid_step == 0):
#                     self.validate()

                # End of step
                # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
                torch.cuda.empty_cache()
                self.timer.set()
                if (self.step2 % 1000 == 0):
                    self.save_checkpoint(f'whisper-v2-{int(self.step2 / 1000)}k-r-final-fix.pth', 'loss',
                             loss, show_msg=False)
                if self.step2 > self.max_step:
                    break
            if self.step2 > self.max_step:
                break
            print("total_loss", total_loss, '\n')
            n_epochs += 1
            
        self.save_checkpoint('whisper-v2-15k-r-final-fix.pth', 'loss',
                             loss, show_msg=False)
        print(self.step2)
        self.log.close()

    def validate(self):
        # Eval mode
        self.model.eval()
        dev_wer = {'att': [], 'ctc': []}
        valid_loss = 0
        for i, data in enumerate(self.dv_set):
            self.progress('Valid step - {}/{}'.format(i + 1, len(self.dv_set)))
            # Fetch data
            feat, feat_len, txt, txt_len = self.fetch_data(data)

            # Forward model
            with torch.no_grad():
                predicted_ids = self.whisper(feat[0], output_attentions=True, output_hidden_states=True)
                last_hidden_state = predicted_ids.hidden_states[-1]
                       
            output = self.model(last_hidden_state)[0]

            loss = self.bceloss(output, txt)
            valid_loss+=loss
#             # Show some example on tensorboard
#             if i == len(self.dv_set) // 2:
#                 for i in range(min(len(txt), self.DEV_N_EXAMPLE)):
#                     if self.step == 1:
#                         self.write_log('true_text{}'.format(
#                             i), self.tokenizer.decode(txt[i].tolist()))
#                     if att_output is not None:
#                         self.write_log('att_align{}'.format(i), feat_to_fig(
#                             att_align[i, 0, :, :].cpu().detach()))
#                         self.write_log('att_text{}'.format(i), self.tokenizer.decode(
#                             att_output[i].argmax(dim=-1).tolist()))
#                     if ctc_output is not None:
#                         self.write_log('ctc_text{}'.format(i), self.tokenizer.decode(ctc_output[i].argmax(dim=-1).tolist(),
#                                                                                      ignore_repeat=True))

        # Ckpt if performance improves
        self.save_checkpoint('latest2.pth', 'loss',
                             loss, show_msg=False)
        if valid_loss < self.bestloss:
                self.bestloss = valid_loss
                self.save_checkpoint('best_biclass2.pth', 'loss', loss)
                print(f"bestloss{valid_loss}")
#         for task in ['att', 'ctc']:
#             dev_wer[task] = sum(dev_wer[task]) / len(dev_wer[task])
#             if dev_wer[task] < self.best_wer[task]:
#                 self.best_wer[task] = dev_wer[task]
#                 self.save_checkpoint('best_{}.pth'.format(
#                     task), 'wer', dev_wer[task])
#             self.write_log('wer', {'dv_' + task: dev_wer[task]})

        # Resume training
        
        self.model.train()
        if self.emb_decoder is not None:
            self.emb_decoder.train()

    def print_model(self):

        self.model = Whisper_Regression(self.feat_dim,
                         **self.config['model'])
#         nonfreeze_keys = ['decoder.layers.weight_ih_l1','decoder.layers.weight_hh_l1', 'decoder.layers.bias_ih_l1', 'decoder.layers.bias_hh_l1']
        nonfreeze_keys = ['fc.weight', 'fc.bias']

        # Plug-ins
        if ('emb' in self.config) and (self.config['emb']['enable']) \
                and (self.config['emb']['fuse'] > 0):
            from core.plugin import EmbeddingRegularizer
            self.emb_decoder = EmbeddingRegularizer(
                self.tokenizer, self.model.dec_dim, **self.config['emb'])

        ckpt = torch.load(self.paras.load, map_location=self.device if self.mode == 'train' else 'cpu')
        pretrained_dict = {k: v for k, v in ckpt.items() if k in self.model.state_dict()}
        self.model.state_dict().update(pretrained_dict) 
        self.model.load_state_dict(self.model.state_dict())
           
                
#         self.model.load_state_dict(ckpt['model'])
        print(self.model)
        for name, para in self.model.named_parameters():
            if para.requires_grad and name not in nonfreeze_keys:
                para.requires_grad = False
            print("-"*20)
            print(f"name: {name}")
            print("values: ")
            print(para)
        for name, para in self.model.named_parameters():
            if para.requires_grad:print(name)
        # Beam decoder
#         self.decoder = BeamDecoder(
#             self.model.cpu(), self.emb_decoder, **self.config['decode'])
#         self.verbose(self.decoder.create_msg())
        del self.model
#         del self.emb_decoder