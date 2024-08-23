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

import torch.nn as nn

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
        # self.options = whisper.DecodingOptions()
        # self.whisper = whisper.load_model("large")

        model_checkpoint2 = "openai/whisper-large-v2"  # distil-whisper/distil-large-v2
        # self.processor = WhisperProcessor.from_pretrained(model_checkpoint)
        # self.whisper = WhisperModel.from_pretrained(model_checkpoint)
        # self.whisper = self.whisper.to(self.device)
        # self.whisper.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="zh", task="transcribe")
        model_checkpoint = "sanchit-gandhi/whisper-large-v2-ft-ls-960h" # whisper-large-v3
        self.whisper = WhisperForAudioClassification.from_pretrained(model_checkpoint)
        self.whisper = self.whisper.to(self.device)
        self.processor = WhisperProcessor.from_pretrained(model_checkpoint2)
        # self.wm = WhisperModel.from_pretrained(model_checkpoint).to(self.device)
        # self.generate = WhisperForConditionalGeneration.from_pretrained(model_checkpoint2).to(self.device)
        
        model_checkpoint = "TencentGameMate/chinese-wav2vec2-large" # whisper-large-v3
        self.whisper = Wav2Vec2Model.from_pretrained(model_checkpoint)
        self.whisper = self.whisper.to(self.device)
#         self.whisper = self.whisper.half()
        self.whisper.eval()
        # self.whisper.freeze_encoder()

        
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
        self.model = Whisper_Classifier( **
        self.config['model']).to(self.device)
        self.verbose(self.model.create_msg())
        model_paras = [{'params': self.model.parameters()}]

        print("# Losses")
        self.bceloss = torch.nn.BCELoss()
#         self.bceloss = self.bceloss.half()
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

        # predicted_ids_set = []
        # model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

        # for i, data in enumerate(self.tr_set):
        #     feat, feat_len, txt, txt_len, audio, file = self.fetch_data(data)
        #     predicted_ids = model.generate(feat[0].to('cpu')).to(self.device)
        #     predicted_ids_set += predicted_ids
        #     print(len(predicted_ids_set), ":", "predicted_ids", predicted_ids.shape)
        #     # decoder_input_ids=predicted_ids_set[j].unsqueeze(0)

        while n_epochs < total_epochs:
            # Renew dataloader to enable random sampling
            if self.curriculum > 0 and n_epochs == self.curriculum:
                self.verbose(
                    'Curriculum learning ends after {} epochs, starting random sampling.'.format(n_epochs))
                self.tr_set, _, _, _, _, _ = \
                    load_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory,
                                 False, **self.config['data'])
            total_loss = 0
            print("self.tr_set: {}".format(len(self.tr_set)))            
            
            for j, data in enumerate(self.tr_set):
                # print("Pre-step : update tf_rate/lr_rate and do zero_grad")
                tf_rate = self.optimizer.pre_step(self.step)

                # Fetch data
                feat, feat_len, txt, txt_len, audio, file = self.fetch_data(data)
                self.timer.cnt('rd')

                # decoder_input_ids = (torch.tensor([[1, 1]]) * self.whisper.config.decoder_start_token_id).to(self.device)

                # Forward model
                # Note: txt should NOT start w/ <sos>
                last_hidden_state_list = None
                with torch.no_grad():
                    # outputs = self.whisper.generate(feat[0], output_hidden_states=True, return_dict_in_generate=True)
                    # print(feat[0].shape[0])
                    for i in range(feat[0].shape[0]):
                        # n = file[0].split('\\')[-1].split('.')[0]
                        # decoder_input_ids = torch.load(f"data_process/CTTsegment/predicted_ids/{n}.pt")
                        # outputs = self.whisper(feat[0], decoder_input_ids=decoder_input_ids, output_attentions=True, output_hidden_states=True)
                        # last_hidden_state = outputs.last_hidden_state
                        predicted_ids = self.whisper(feat[0], output_attentions=True, output_hidden_states=True)
                        # last_hidden_state = predicted_ids.hidden_states[-1]
                        last_hidden_state = predicted_ids.last_hidden_state
                        
                        # last_hidden_state = outputs.decoder_hidden_states[31]

                        if(i == 0):
                            last_hidden_state_list = last_hidden_state
                        # else:
                        #     last_hidden_state_list = torch.cat((last_hidden_state_list, last_hidden_state), 0)     
                            # print(last_hidden_state_list.shape)
                    # pipe = pipeline(self.whisper, device=self.device)
                    # predictions = pipe(feat[0], chunk_length_s=30, stride_length_s=30)
                    # last_hidden_state = predictions.last_hidden_state
                output = self.model(last_hidden_state)[0]
                # print(self.whisper.config.classifier_proj_size)
                # print('output',output)

                # decoder_hidden_states = outputs.decoder_hidden_states
                # print("decoder_hidden_states: ", len(decoder_hidden_states), decoder_hidden_states[0].shape)
#                 print(output.shape)
#                 print(output)
#                 print('txt:',txt)
                loss = self.bceloss(output, txt)
#                 print(loss)
#                 for name, param in self.model.named_parameters():
#                     print(name, torch.isfinite(param.grad).all())
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
                    self.save_checkpoint(f'wav2vec-{int(self.step2 / 1000)}k-r-a-final.pth', 'loss',
                             loss, show_msg=False)
                # if (self.step2 % len(self.tr_set) == 0):
                #     self.save_checkpoint(f'whisper{int(self.step2 / len(self.tr_set))}n_with_id.pth', 'loss',
                #              loss, show_msg=False)
                
                if self.step2 > self.max_step:
                    break
            if self.step2 > self.max_step:
                break
            print("total_loss", total_loss, '\n')
            n_epochs += 1
        self.save_checkpoint('wav2vec-15k-r-a-final.pth', 'loss',
                             loss, show_msg=False)
        print(self.step2)
        self.log.close()

    def validate(self, round):
        # Eval mode
        self.model.eval()
        dev_wer = {'att': [], 'ctc': []}
        valid_loss = 0

        # predicted_ids_set = []
        # model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

        # for i, data in enumerate(self.dv_set):
        #     feat, feat_len, txt, txt_len, audio, file = self.fetch_data(data)
        #     predicted_ids = model.generate(feat[0].to('cpu')).to(self.device)
        #     predicted_ids_set += predicted_ids
        #     print(len(predicted_ids_set), ":", "predicted_ids", predicted_ids.shape)
        #     # decoder_input_ids=predicted_ids_set[j].unsqueeze(0)
        
        for i, data in enumerate(self.dv_set):
            self.progress('Valid step - {}/{}'.format(i + 1, len(self.dv_set)))
            # Fetch data
            feat, feat_len, txt, txt_len, audio, file = self.fetch_data(data)

            # decoder_input_ids = (torch.tensor([[1, 1]]) * self.whisper.config.decoder_start_token_id).to(self.device)
            # print(decoder_input_ids, self.whisper.config.decoder_start_token_id)

            # Forward model
#             with torch.no_grad():
#                 output = self.model(feat)
            with torch.no_grad():
                # outputs = self.whisper.generate(feat[0], output_hidden_states=True, return_dict_in_generate=True)

                # n = file[0].split('\\')[-1].split('.')[0]
                # decoder_input_ids = torch.load(f"data_process/CTTsegment/predicted_ids/{n}.pt")
                # outputs = self.whisper(feat[0], decoder_input_ids=decoder_input_ids, output_attentions=True, output_hidden_states=True)
                # last_hidden_state = outputs.last_hidden_state
                outputs = self.whisper(feat[0], output_attentions=True, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]

#                 print('last_hidden_state',last_hidden_state)
                output = self.model(last_hidden_state)[0]
                print('output',output)

                # encoder_last_hidden_state = outputs.encoder_last_hidden_state
                # decoder_hidden_states = outputs.decoder_hidden_states
                # encoder_hidden_states = outputs.encoder_hidden_states
                # decoder_attentions = outputs.decoder_attentions
                # encoder_attentions = outputs.encoder_attentions
                # cross_attentions = outputs.cross_attentions
                hidden_states = outputs.hidden_states
                attentions = outputs.attentions
                file_name = file[0].split("\\")[-1].split("C")[0]

                # print("last_hidden_state: ", last_hidden_state.shape, "encoder_last_hidden_state: ", encoder_last_hidden_state.shape)
                # print("decoder_hidden_states: ", len(decoder_hidden_states), decoder_hidden_states[0].shape)

#                 for name, param in self.model.named_parameters():
#                     print(name, torch.isfinite(p aram.grad).all())

            loss = self.bceloss(output, txt)
            valid_loss+=loss

            writer = SummaryWriter(log_dir='log/whisper_r_a_final_fix')
            # writer.add_histogram('encoder_last_hidden_state', encoder_last_hidden_state)
            # writer.add_histogram('decoder_hidden_states ', decoder_hidden_states)
            # writer.add_histogram('encoder_hidden_states ', encoder_hidden_states)
            # writer.add_histogram/('last_hidden_state', last_hidden_state)
            # writer.add_histogram('hidden_states', hidden_states)

            # for j in range(len(decoder_attentions)):
            #     writer.add_histogram('decoder_attentions', decoder_attentions[0])
            #     writer.add_histogram('encoder_attentions', encoder_attentions[0])
            #     writer.add_histogram('cross_attentions', cross_attentions[0])
            for j in range(len(attentions)):
                writer.add_histogram('attentions', attentions[0])

            for name, param in self.model.named_parameters():
                if 'weight' in name:   # find the fc weight
                    param_cpu = param.data.cpu()
                    writer.add_histogram(name, param_cpu.numpy())
                    writer.add_histogram(name+str(round), param_cpu.numpy())
                    # writer.add_scalar(f'{name}.mean', param_cpu.mean().item())
                    # writer.add_scalar(f'{name}.std', param_cpu.std().item())
                    # writer.add_scalar(f'{name}.importance', param_cpu.abs().mean().item())
                       
            import matplotlib.pyplot as plt
            import seaborn as sns
            import librosa.display
            import numpy as np
            import soundfile as sf
            import torch.nn.functional as F

            # get attention weights and mel spectrogram
            attentions_weights = attentions
            # cross_attention_weights = cross_attentions
            # decoder_attentions_weights = decoder_attentions
            # encoder_attentions_weights = encoder_attentions
            mel_spectrum = feat[0][0]
            # print(decoder_attentions_weights, outputs)
            # print(len(cross_attention_weights))
            # print(cross_attention_weights[0][0].shape)
            # print(len(decoder_attentions_weights))
            # print(decoder_attentions_weights[0][0].shape)
            # print(len(encoder_attentions_weights))
            # print(encoder_attentions_weights[0][0].shape)
            # print(mel_spectrum.shape)
            # cross_attention_weights = cross_attention_weights[0][0][0].squeeze(0).cpu().detach().numpy()
            attentions_weights = attentions_weights[-1]
            # attentions_weights = attentions_weights[0].squeeze(0).cpu().detach().numpy()
            # cross_attention_weights = cross_attention_weights[0].squeeze(0).cpu().detach().numpy()
            # decoder_attentions_weights = decoder_attentions_weights[0].squeeze(0).cpu().detach().numpy()
            # encoder_attentions_weights = encoder_attentions_weights[0].squeeze(0).cpu().detach().numpy()
            mel_spectrum = mel_spectrum.squeeze().cpu().detach().numpy()
            # print(cross_attention_weights.shape)
            # print(cross_attention_weights.mean(axis=0).shape)

            # mel spectrogram
            # mel = np.exp(mel_spectrum)
            mel_spect = librosa.feature.melspectrogram(y=audio[0], sr=16000, n_fft=400, hop_length=160, n_mels=128)
            mel_db = librosa.power_to_db(mel_spect, ref=np.max)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mel_db, x_axis='s', y_axis='mel', sr=16000, hop_length=160, n_fft=400, cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.show()

            # TensorBoard
            writer.add_figure('Mel Spectrum ' + file_name, plt.gcf(), global_step=0)


            # encoder attention map with visualizer
            from validation_visualize import visualize_head, visualize_heads, visualize_head_average, visualize_heads_average, visualize_heads_resized
            # target_shape = feat[0][0].shape
            # visualize_heads_resized(attentions_weights, cols=4, target_shape=target_shape)mport torch.nn.functional as F
            attentions_weights_fig = F.interpolate(attentions_weights, size=(3000, 3000), mode='bilinear', align_corners=False).squeeze(0).cpu().detach().numpy()
            visualize_heads(attentions_weights_fig, cols=4)

            # TensorBoard
            writer.add_figure('Attention Weights ' + file_name, plt.gcf(), global_step=0)


            attentions_weights_fig = F.interpolate(attentions_weights, size=(3000, 3000), mode='bilinear', align_corners=False).squeeze(0).cpu().detach().numpy()
            visualize_head_average(attentions_weights_fig)

            # TensorBoard
            writer.add_figure('Attention Weight Average' + file_name, plt.gcf(), global_step=0)


            # # encoder attention map 
            # # resize self-attention map size to mel spectrogram size
            # # target_shape = feat[0][0].shape
            # attention_resized = np.resize(attentions_weights.mean(axis=0), target_shape)

            # # # overlap
            # # alpha = 0.5  
            # # overlay = alpha * attention_resized + (1 - alpha) * mel_spectrum[0]

            # # resize attention map
            # plt.imshow(attention_resized, cmap='viridis', aspect='auto', origin='lower')
            # plt.title('Attention Weights Resize')
            # plt.colorbar()  
            # plt.xlabel('Time')
            # plt.ylabel('Mel Spectrogram Frequency')
            # plt.show()

            # # TensorBoard
            # writer.add_figure('Attention Weights Resize Average '+str(round)+' '+str(i), plt.gcf(), global_step=0)


            # # decoder attention map
            # plt.figure(figsize=(10, 6))
            # sns.heatmap(cross_attention_weights.mean(axis=0), cmap='viridis', annot=True, fmt=".2f", xticklabels=False, yticklabels=False)
            # plt.xlabel('Key Sequence')
            # plt.ylabel('Query Sequence')
            # plt.title('Cross Attention Weights Visualization')
            # plt.show()

            # # TensorBoard
            # writer.add_figure('Cross Attention Weights Map '+str(round)+' '+str(i), plt.gcf(), global_step=0)

            # # decoder attention map with visualizer
            # from validation_visualize import visualize_head, visualize_heads, visualize_head_average, visualize_heads_average
            # visualize_heads_average(cross_attention_weights, cols=4)

            # # TensorBoard
            # writer.add_figure('Cross Attention Weights '+str(round)+' '+str(i), plt.gcf(), global_step=0)


            # # attention weights v.s. spectrogram
            # plt.figure(figsize=(12, 6))
            # plt.plot(mel_spectrum[0], label='Mel Spectrum', linewidth=2)
            # plt.plot(cross_attention_weights, label='Attention Weights', linestyle='--', linewidth=2)
            # plt.xlabel('Time Steps')
            # plt.ylabel('Magnitude')
            # plt.title('Comparison of Mel Spectrum and Attention Weights')
            # plt.legend()

            # # TensorBoard
            # writer.add_figure('Mel Spectrum vs Attention Weights '+str(round)+' '+str(i), plt.gcf(), global_step=0)

            # # attention weights v.s. spectrogram 2
            # resize_cross_attention_weights = cross_attention_weights.mean(axis=0).resize((mel_db.shape[0], mel_db.shape[1]))
            # # Mel Spectrogram
            # fig, ax = plt.subplots(figsize=(10, 6))
            # librosa.display.specshow(mel_db, y_axis='mel', fmax=3000, x_axis='ms', cmap='viridis')
            # plt.colorbar(format='%+2.0f dB', label='Intensity')

            # # Cross-Attention Weights
            # sns.heatmap(cross_attention_weights.mean(axis=0), cmap='viridis', annot=True, fmt=".2f", alpha=0.3, ax=ax, cbar=False)
            
            # # TensorBoard
            # writer.add_figure('Mel Spectrum vs Cross-Attention Weights '+str(round)+' '+str(i), plt.gcf(), global_step=0)


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
        if valid_loss < self.bestloss:
                self.bestloss = valid_loss
                self.save_checkpoint('best_biclass.pth', 'loss', loss)
                print(f"bestloss{valid_loss}")
#         for task in ['att', 'ctc']:
#             dev_wer[task] = sum(dev_wer[task]) / len(dev_wer[task])
#             if dev_wer[task] < self.best_wer[task]:
#                 self.best_wer[task] = dev_wer[task]
#                 self.save_checkpoint('best_{}.pth'.format(
#                     task), 'wer', dev_wer[task])
#             self.write_log('wer', {'dv_' + task: dev_wer[task]})

        # # Resume training
        # self.model.train()
        # if self.emb_decoder is not None:
        #     self.emb_decoder.train()

    def print_model(self):
        self.model = Whisper_Classifier(
                         **self.config['model'])
#         nonfreeze_keys = ['decoder.layers.weight_ih_l1','decoder.layers.weight_hh_l1', 'decoder.layers.bias_ih_l1', 'decoder.layers.bias_hh_l1']
#         nonfreeze_keys = ['fc.weight', 'fc.bias']

        # Plug-ins
        if ('emb' in self.config) and (self.config['emb']['enable']) \
                and (self.config['emb']['fuse'] > 0):
            from core.plugin import EmbeddingRegularizer
            self.emb_decoder = EmbeddingRegularizer(
                self.tokenizer, self.model.dec_dim, **self.config['emb'])

        ckpt = torch.load(self.paras.load, map_location=self.device if self.mode == 'train' else 'cpu')
#         pretrained_dict = {k: v for k, v in ckpt.items() if k in self.model.state_dict()}
#         self.model.state_dict().update(pretrained_dict) 
#         self.model.load_state_dict(self.model.state_dict())
           
                
        self.model.load_state_dict(ckpt['model'])
        print(self.model)
        for name, para in self.model.named_parameters():
#             if para.requires_grad and name not in nonfreeze_keys:
#                 para.requires_grad = False
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

    def visualization(self):
        # Eval mode
        self.model.eval()
        dev_wer = {'att': [], 'ctc': []}
        valid_loss = 0
        
        from captum.attr import IntegratedGradients, Saliency, LayerGradCam
        from captum.attr import visualization as viz
        import matplotlib.pyplot as plt
        import seaborn as sns
        import librosa.display
        import numpy as np
        import soundfile as sf
        import torch.nn.functional as F
        
        for i, data in enumerate(self.dv_set):
            self.progress('Valid step - {}/{}'.format(i + 1, len(self.dv_set)))
            # Fetch data
            feat, feat_len, txt, txt_len, audio, file = self.fetch_data(data)
            
            file_name = file[0].split("\\")[-1].split("C")[0]
            
            writer = SummaryWriter(log_dir='log/whisper_r_a_final_fix')

            with torch.no_grad():
                outputs = self.whisper(feat[0], output_attentions=True, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]

#                 print('last_hidden_state',last_hidden_state)
                output = self.model(last_hidden_state)[0]
                print('output',output)
            
            attentions = outputs.attentions

            # Use Integrated Gradients
            ig = IntegratedGradients(self.model)
            attributions = ig.attribute(last_hidden_state, target=0)  # Assuming target class is 1 (dementia)

            attr = attributions.squeeze().cpu().numpy()
            plt.plot(attr)
            plt.title("Integrated Gradients Attributions")
            plt.xlabel("Time")
            plt.ylabel("Attribution")
            plt.show()

            # TensorBoard
            writer.add_figure('Integrated Gradients' + file_name, plt.gcf(), global_step=0)

            # print(attributions.shape, last_hidden_state.shape, np.transpose(attributions.squeeze().cpu().detach().numpy()).shape)

            # 在嵌入維度上求和 attribution
            attributions_sum = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()  # shape: (1500,)

            # 正規化 attribution 以便於可視化
            attributions_sum = (attributions_sum - np.min(attributions_sum)) / (np.max(attributions_sum) - np.min(attributions_sum))
            interp_attributions = np.interp(np.arange(3000), np.linspace(0, 2999, num=1500), attributions_sum)

            # 將映射後的 attribution 擴展到頻率維度
            attribution_image = np.tile(interp_attributions, (feat[0].shape[1], 1))  # shape: (80, 3000)
            # attribution_image = np.transpose(attribution_image, (1,0))
            attribution_image = attribution_image.reshape(*attribution_image.shape, 1)
            # print(attribution_image.shape, np.transpose(feat[0].cpu().detach().numpy(), (2,1,0)).shape)
            
            original_image = np.transpose(feat[0].cpu().detach().numpy(), (1,2,0))
            # print("feat", feat[0])
            # print("orginal", original_image)

            # Visualize the attributions
            viz.visualize_image_attr(
                None,
                original_image,
                method='original_image',
                title="Original Image"
            )
            
            # TensorBoard
            writer.add_figure('Original Image' + file_name, plt.gcf(), global_step=0)

            # Visualize the attributions
            viz.visualize_image_attr(
                attribution_image,
                original_image,
                method='blended_heat_map',
                sign='absolute_value',
                show_colorbar=True,
                title="Attributions for the input audio"
            )
            
            # TensorBoard
            writer.add_figure('Integrated Gradients Image ' + file_name, plt.gcf(), global_step=0)

            # Use Saliency
            saliency = Saliency(self.model)
            attr = saliency.attribute(last_hidden_state, target=0)  # Assuming target class is 1 (dementia)

            # Visualize the saliency map
            attr = attr.squeeze().cpu().numpy()
            plt.imshow(attr, aspect='auto', cmap='hot')
            plt.title("Saliency Map")
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.colorbar()
            plt.show()

            # TensorBoard
            writer.add_figure('Saliency ' + file_name, plt.gcf(), global_step=0)


            # # Use Layer Grad-CAM
            # layer_gc = LayerGradCam(self.model, self.model.fc)
            # attr = layer_gc.attribute(last_hidden_state, target=0)  # Assuming target class is 1 (dementia)

            # # Visualize the Grad-CAM
            # attr = attr.squeeze().cpu().numpy()
            # plt.imshow(attr, aspect='auto', cmap='hot')
            # plt.title("Grad-CAM")
            # plt.xlabel("Time")
            # plt.ylabel("Frequency")
            # plt.colorbar()
            # plt.show()
            
            # # TensorBoard
            # writer.add_figure('LayerGradCam' + file_name, plt.gcf(), global_step=0)


            # get attention weights and mel spectrogram
            attentions_weights = attentions
            mel_spectrum = feat[0][0]
            attentions_weights = attentions_weights[-1]
            mel_spectrum = mel_spectrum.squeeze().cpu().detach().numpy()

            # mel spectrogram
            mel_spect = librosa.feature.melspectrogram(y=audio[0], sr=16000, n_fft=400, hop_length=160, n_mels=128)
            mel_db = librosa.power_to_db(mel_spect, ref=np.max)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mel_db, x_axis='s', y_axis='mel', sr=16000, hop_length=160, n_fft=400, cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.show()

            # TensorBoard
            writer.add_figure('Mel Spectrum ' + file_name, plt.gcf(), global_step=0)


            # encoder attention map with visualizer
            from validation_visualize import visualize_head, visualize_heads, visualize_head_average, visualize_heads_average, visualize_heads_resized
            attentions_weights_fig = F.interpolate(attentions_weights, size=(3000, 3000), mode='bilinear', align_corners=False).squeeze(0).cpu().detach().numpy()
            visualize_heads(attentions_weights_fig, cols=4)

            # TensorBoard
            writer.add_figure('Attention Weights ' + file_name, plt.gcf(), global_step=0)


            attentions_weights_fig = F.interpolate(attentions_weights, size=(3000, 3000), mode='bilinear', align_corners=False).squeeze(0).cpu().detach().numpy()
            visualize_head_average(attentions_weights_fig)

            # TensorBoard
            writer.add_figure('Attention Weight Average ' + file_name, plt.gcf(), global_step=0)


            class CombinedModel(nn.Module):
                def __init__(self, whisper_c, model_c):
                    super().__init__()
                    self.whisper_c = whisper_c
                    self.model_c = model_c
                    
                    self.whisper_c.train()
                    # # Freeze the Whisper model parameters
                    # for param in self.whisper_c.parameters():
                    #     param.requires_grad = False

                def forward(self, mel_input):
                    outputs = self.whisper_c(mel_input, output_attentions=True, output_hidden_states=True)
                    last_hidden_state = outputs.hidden_states[-1]
                    
                    output = self.model_c(last_hidden_state)
                    return output

            # Assuming your custom model is defined as CustomModel
            combined_model = CombinedModel(self.whisper, self.model)

            # Create an instance of IntegratedGradients
            ig = IntegratedGradients(combined_model)
            saliency = Saliency(combined_model)

            # Compute attributions
            attributions = saliency.attribute(feat[0], target=0)  # Adjust target as needed

            # Visualize the attributions
            # You may need to sum across the frequency dimension if you want a 1D visualization
            attributions_sum = attributions.sum(dim=1).squeeze(0)
            plt.plot(attributions_sum.cpu().detach().numpy())
            plt.show()

            
            # TensorBoard
            writer.add_figure('Combine ' + file_name, plt.gcf(), global_step=0)
