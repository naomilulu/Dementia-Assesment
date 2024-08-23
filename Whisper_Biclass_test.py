#! python
# -*- coding: utf-8 -*-
# Author: kun
# @Time: 2019-10-29 20:36

import torch
from core.solver import BaseSolver

from core.whisper import Whisper_Classifier
from core.optim import Optimizer
from core.data import load_whisper_dataset
from core.util import human_format, cal_er, feat_to_fig

import whisper

from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    WhisperModel,
    AutoFeatureExtractor, 
    WhisperForAudioClassification,    
    Wav2Vec2Processor,Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor, Wav2Vec2Model
)


class Solver(BaseSolver):
    ''' Solver for training'''

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)
        # Logger settings
        self.best_wer = {'att': 3.0, 'ctc': 3.0}
        self.bestloss=100
        # Curriculum learning affects data loader
        self.curriculum = self.config['hparas']['curriculum']
        # self.options = whisper.DecodingOptions()
        # self.whisper = whisper.load_model("large")

        # model_checkpoint = "openai/whisper-large"  # distil-whisper/distil-large-v2
        # processor = WhisperProcessor.from_pretrained(model_checkpoint)
        # self.whisper = WhisperModel.from_pretrained(model_checkpoint)
        # self.whisper = self.whisper.to(self.device)
        # self.whisper.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")
        model_checkpoint = "sanchit-gandhi/whisper-large-v2-ft-ls-960h" # whisper-large-v3
        self.whisper = WhisperForAudioClassification.from_pretrained(model_checkpoint)
        self.whisper = self.whisper.to(self.device)

        model_checkpoint = "TencentGameMate/chinese-wav2vec2-large" # whisper-large-v3
        self.whisper = Wav2Vec2Model.from_pretrained(model_checkpoint)
        self.whisper = self.whisper.to(self.device)

#         self.whisper = self.whisper.half()
        self.whisper.eval()
        
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
#             pretrained_dict = {k: v for k, v in ckpt.items() if k in self.model.state_dict()}
#             self.model.state_dict().update(pretrained_dict) 
#             self.model.load_state_dict(self.model.state_dict())

           
                
            self.model.load_state_dict(ckpt['model'])
            

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
                self.verbose('Load ckpt from {}, restarting at step {} (recorded {} = {:.2f} %)'.format(
                    self.paras.load, self.step, metric, score))
            else:
                self.model.eval()
                if self.emb_decoder is not None:
                    self.emb_decoder.eval()
#                 self.verbose('Evaluation target = {} (recorded {} = {:.2f} %)'.format(
#                     self.paras.load, metric, score))

    def fetch_data(self, data):
        ''' Move data to device and compute text seq. length'''
        file, audio, feat, feat_len, txt = data
        feat = feat.to(self.device)
        feat_len = feat_len.to(self.device)
        txt = txt.to(self.device)
        txt_len = torch.sum(txt != 0, dim=-1)

        return feat, feat_len, txt, txt_len, audio, file

    def load_data(self):
        print("Load data for training/validation, store tokenizer and input/output shape")
        self.dv_set, self.tt_set = \
            load_whisper_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory,
                         self.curriculum > 0, **self.config['data'])


    def set_model(self):
        print("Setup ASR model and optimizer ")
        nonfreeze_keys = ['fc.weight', 'fc.bias']
        # Model
        self.model = Whisper_Classifier( **
        self.config['model']).to(self.device)
#         self.verbose(self.model.create_msg())
#         model_paras = [{'params': self.model.parameters()}]

#         print("# Losses")
#         self.bceloss = torch.nn.BCELoss()
#         print("# Note: zero_infinity=False is unstable?")
# #         self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)


#         print("# Optimizer")
#         self.optimizer = Optimizer(model_paras, **self.config['hparas'])
#         self.verbose(self.optimizer.create_msg())

#         print("# Enable AMP if needed")
#         self.enable_apex()

        # Automatically load pre-trained model if self.paras.load is given
        self.load_ckpt()
#         for name, para in self.model.named_parameters():
#             if para.requires_grad and name not in nonfreeze_keys:
#                 para.requires_grad = False
#         for name, para in self.model.named_parameters():
#             if para.requires_grad:print(name)
#         non_frozen_parameters = [p for p in self.model.parameters() if p.requires_grad]
#         self.optimizer = Optimizer(non_frozen_parameters, **self.config['hparas'])

        # ToDo: other training methods

    def exec(self):
        ''' Testing End-to-end ASR system '''
        names=[]
        hyps=[]
        txts=[]
        ans=[]

        # predicted_ids_set = []
        # model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

        # for i, data in enumerate(self.tt_set):
        #     feat, feat_len, txt, txt_len, audio, file = self.fetch_data(data)
        #     predicted_ids = model.generate(feat[0].to('cpu')).to(self.device)
        #     predicted_ids_set += predicted_ids
        #     print(len(predicted_ids_set), ":", "predicted_ids", predicted_ids.shape)
        #     # decoder_input_ids=predicted_ids_set[j].unsqueeze(0)

        for j, data in enumerate(self.tt_set):
            name, audio, feat, feat_len, txt = data
            feat = feat.to(self.device)
            feat_len = feat_len.to(self.device)
            txt = txt.to(self.device)

            # decoder_input_ids = (torch.tensor([[1, 1]]) * self.whisper.config.decoder_start_token_id).to(self.device)
            
            # with torch.no_grad():
            #     outputs = self.whisper(feat[0], decoder_input_ids=decoder_input_ids)
            #     last_hidden_state = outputs.last_hidden_state
            #     hyp = self.model(last_hidden_state)


            last_hidden_state_list = None
            with torch.no_grad():
                # outputs = self.whisper.generate(feat[0], output_hidden_states=True, return_dict_in_generate=True)
                # print(feat[0].shape[0])
                for i in range(feat[0].shape[0]):
                    # n = name[0].split('\\')[-1].split('.')[0]
                    # decoder_input_ids = torch.load(f"data_process/CTTsegment/predicted_ids/{n}.pt")
                    # outputs = self.whisper(feat[0], decoder_input_ids=decoder_input_ids, output_attentions=True, output_hidden_states=True)
                    # last_hidden_state = outputs.last_hidden_state
                    outputs = self.whisper(feat[0], output_attentions=True, output_hidden_states=True)
                    last_hidden_state = outputs.hidden_states[-1]

                    if(i == 0):
                        last_hidden_state_list = last_hidden_state
                    # else:
                    #     last_hidden_state_list = torch.cat((last_hidden_state_list, last_hidden_state), 0)     
                        # print(last_hidden_state_list.shape)
                # pipe = pipeline(self.whisper, device=self.device)
                # predictions = pipe(feat[0], chunk_length_s=30, stride_length_s=30)
                # last_hidden_state = predictions.last_hidden_state
            hyp = self.model(last_hidden_state_list)[0]

            an = ((hyp>=0.5) == (txt==1)).tolist()[0]
            print(name, ' ', hyp.tolist()[0], ' ', txt.tolist()[0], ' ', an)
            names.append(name[0])
            hyps.append(hyp.tolist()[0])
            txts.append(txt.tolist()[0])
            ans.append(((hyp>=0.5) == (txt==1)).tolist()[0])
            
            
        self.verbose('All done !')
        return names, hyps, txts, ans


    def validate(self):
        # Eval mode
        self.model.eval()
        dev_wer = {'att': [], 'ctc': []}

        for i, data in enumerate(self.dv_set):
            self.progress('Valid step - {}/{}'.format(i + 1, len(self.dv_set)))
            # Fetch data
            feat, feat_len, txt, txt_len, audio, file = self.fetch_data(data)

            # Forward model
            with torch.no_grad():
                output = self.model(feat, feat_len)

            loss = self.bceloss(output, txt)

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
        self.save_checkpoint('latest.pth', 'loss',
                             loss, show_msg=False)
        if loss < self.bestloss:
                self.bestloss = loss
                self.save_checkpoint('best_biclass.pth', 'loss', loss)
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
        self.model = Whisper_Classifier( **
        self.config['model']).to(self.device)
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