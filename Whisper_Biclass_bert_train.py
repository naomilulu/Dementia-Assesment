#! python
# -*- coding: utf-8 -*-
# Author: kun
# @Time: 2019-10-29 20:36

import torch
from core.solver import BaseSolver

from core.bert import Bert_Classifier
from core.optim import Optimizer
from core.data import load_bert_dataset
from core.util import human_format, cal_er, feat_to_fig

import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertModel
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

        model_checkpoint = "google-bert/bert-base-chinese"
        # model_checkpoint = "distilbert/distilbert-base-multilingual-cased"
        
        # self.bert = BertModel.from_pretrained(model_checkpoint)
        self.token = AutoTokenizer.from_pretrained(model_checkpoint)
        self.bert = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
        self.bert = self.bert.to(self.device)

#         self.bert = self.bert.half()
        self.bert.eval()
        # self.bert.freeze_encoder()

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
        file, audio, feat, attention, typeid, feat_len, txt = data
        feat = feat[0].to(self.device)
        attention = attention[0].to(self.device)
        typeid = typeid[0].to(self.device)
        feat_len = feat_len.to(self.device)
        txt = txt.to(self.device)
        txt_len = torch.sum(txt != 0, dim=-1)
        # print(type(audio))
        audio = np.array(audio)

        return feat, attention, typeid, feat_len, txt, txt_len, audio, file

    def load_data(self):
        print("Load data for training/validation, store tokenizer and input/output shape")
        self.tr_set, self.dv_set= \
            load_bert_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory,
                         self.curriculum > 0, **self.config['data'])

    def set_model(self):
        print("Setup ASR model and optimizer ")
        nonfreeze_keys = ['fc.weight', 'fc.bias']
        # Model
        self.model = Bert_Classifier( **
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
                feat, attention, typeid, feat_len, txt, txt_len, audio, file = self.fetch_data(data)
                self.timer.cnt('rd')

                # Forward model
                # Note: txt should NOT start w/ <sos>
                with torch.no_grad():
                    # outputs = self.bert(input_ids=feat, attention_mask=attention, token_type_ids=typeid, output_attentions=True, output_hidden_states=True)
                    outputs = self.bert(input_ids=feat, attention_mask=attention, output_attentions=True, output_hidden_states=True)
                    last_hidden_state = outputs.hidden_states[12]
                    # print(len(outputs.hidden_states), outputs.hidden_states[0].shape)
                    # last_hidden_state = torch.cat(outputs.hidden_states[-4:], dim=0)
                    # last_hidden_state = torch.mean(torch.stack(outputs.hidden_states[-4:]), dim=0)
                
                output = self.model(last_hidden_state)[0]
                # print(self.bert.config.classifier_proj_size)
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
                # if (self.step2 % 1000 == 0):
                #     self.save_checkpoint(f'bert-v2-{int(self.step2 / 1000)}k-r-a-final-0.pth', 'loss',
                #              loss, show_msg=False)
                # if (self.step2 % len(self.tr_set) == 0):
                #     self.save_checkpoint(f'bert{int(self.step2 / len(self.tr_set))}n_with_id.pth', 'loss',
                #              loss, show_msg=False)
                
                if self.step2 > self.max_step:
                    break
            if self.step2 > self.max_step:
                break
            print("total_loss", total_loss, '\n')
            n_epochs += 1
        self.save_checkpoint('bert-v2-15k-r-a-final-s12.pth', 'loss',
                             loss, show_msg=False)
        print(self.step2)
        self.log.close()

    def validate(self, round):
        # Eval mode
        self.model.eval()
        dev_wer = {'att': [], 'ctc': []}
        valid_loss = 0

        for i, data in enumerate(self.dv_set):
            self.progress('Valid step - {}/{}'.format(i + 1, len(self.dv_set)))
            # Fetch data
            feat, attention, typeid, feat_len, txt, txt_len, audio, file = self.fetch_data(data)

            # Forward model
#             with torch.no_grad():
#                 output = self.model(feat)
            with torch.no_grad():
                outputs = self.bert(input_ids=feat, attention_mask=attention, token_type_ids=typeid, output_attentions=True, output_hidden_states=True)
                # last_hidden_state = outputs.hidden_states[-1]
                last_hidden_state = torch.cat(outputs.hidden_states[-4:], dim=0)

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

                # print("last_hidden_state: ", last_hidden_state.shape, "encoder_last_hidden_state: ", encoder_last_hidden_state.shape)
                # print("decoder_hidden_states: ", len(decoder_hidden_states), decoder_hidden_states[0].shape)

#                 for name, param in self.model.named_parameters():
#                     print(name, torch.isfinite(p aram.grad).all())

            loss = self.bceloss(output, txt)
            valid_loss+=loss

            writer = SummaryWriter(log_dir='log/cv11Lu_asr_lstm4atthead_allvocab-biclass2-5fold_sd0_bert')
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

            # encoder attention map with visualizer
            from bertviz import model_view, head_view

            inputs = feat[0]
            # outputs = self.bert(inputs)  # Run model
            attention = outputs[-1]  # Retrieve attention from model outputs
            # print(inputs)
            tokens = self.token.convert_ids_to_tokens(inputs)  # Convert input ids to token strings

            html_model_view = model_view(attention, tokens, html_action='return')
            with open("log/bert_o_d_final/model_view_{}.html".format(file[0]), 'w') as f:
                f.write(html_model_view.data)

            html_head_view = head_view(attention, tokens, html_action='return')
            with open("log/bert_o_d_final/head_view_{}.html".format(file[0]), 'w') as f:
                f.write(html_head_view.data)

            # TensorBoard
            # writer.add_figure('Attention Weights '+str(round)+' '+str(i), plt.gcf(), global_step=0)


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
        self.model = Bert_Classifier(
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
        
        from captum.attr import IntegratedGradients, Saliency, LayerGradCam, LayerIntegratedGradients, InterpretableEmbeddingBase, configure_interpretable_embedding_layer, remove_interpretable_embedding_layer, TokenReferenceBase
        from captum.attr import visualization as viz
        import matplotlib.pyplot as plt
        import seaborn as sns
        import librosa.display
        import numpy as np
        import soundfile as sf
        import torch.nn.functional as F
        import matplotlib
        matplotlib.rc('font', family='Microsoft JhengHei')
        
        for i, data in enumerate(self.dv_set):
            self.progress('Valid step - {}/{}'.format(i + 1, len(self.dv_set)))
            # Fetch data
            feat, attention, typeid, feat_len, txt, txt_len, audio, file = self.fetch_data(data)
            
            file_name = file[0]

            writer = SummaryWriter(log_dir='log/bert-v2-15k-r-final-2')

            with torch.no_grad():
                outputs = self.bert(input_ids=feat, attention_mask=attention, token_type_ids=typeid, output_attentions=True, output_hidden_states=True)
                # last_hidden_state = outputs.hidden_states[-1]
                last_hidden_state = torch.cat(outputs.hidden_states[-4:], dim=0)

#                 print('last_hidden_state',last_hidden_state)
            output = self.model(last_hidden_state)[0]
            print('output',output)

            # Define the Integrated Gradients instance
            ig = IntegratedGradients(self.model)

            # Attribute score using Integrated Gradients
            attributions, delta = ig.attribute(last_hidden_state,
                                            target=0,  # Example target index
                                            return_convergence_delta=True)

            # print(attributions.shape, last_hidden_state.shape, np.transpose(attributions.squeeze().cpu().detach().numpy(), (1,2,0)).shape)

            # Visualize the attributions
            viz.visualize_image_attr(
                np.transpose(attributions.squeeze().cpu().detach().numpy(), (1,2,0)),
                np.transpose(last_hidden_state.squeeze().cpu().detach().numpy(), (1,2,0)),
                method='heat_map',
                sign='all',
                show_colorbar=True,
                title="Attributions for the input text"
            )

            # TensorBoard
            writer.add_figure('Integrated Gradients ' + file_name, plt.gcf(), global_step=0)
          
            # # interpretable_emb = InterpretableEmbeddingBase(self.bert)
            # interpretable_emb = configure_interpretable_embedding_layer(self.bert, 'hidden_states')
            # input_emb = interpretable_emb.indices_to_embeddings(feat[0])
            # ig = IntegratedGradients(self.bert)
            # attribution = ig.attribute(input_emb, target=3)
            # remove_interpretable_embedding_layer(self.bert, interpretable_emb)

            # # Sum the attributions across embedding dimensions
            # attributions_sum = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()  # Shape: (sequence_length,)

            # # Normalize the attributions for visualization
            # attributions_sum = (attributions_sum - np.min(attributions_sum)) / (np.max(attributions_sum) - np.min(attributions_sum))

            # # Tokenize the input text to get token IDs and tokens
            # tokens = self.token.convert_ids_to_tokens(input_ids[0].tolist())  # Convert input_ids to tokens

            # # viz.visualize_text([visualization_data])
            # viz.visualize_text([viz.VisualizationDataRecord(
            #     attributions_sum,
            #     torch.max(torch.softmax(output[0], dim=0)),
            #     torch.argmax(output),
            #     torch.argmax(output),
            #     str(tokens),
            #     attributions_sum.sum(),
            #     tokens,
            #     delta
            # )])

            # TensorBoard
            # writer.add_figure('Integrated Gradients text' + file_name, plt.gcf(), global_step=0)

            # # Visualize attributions for each token
            # visualization_data = []
            # for token, score in zip(tokens, attributions_sum):
            #     visualization_data.append((token, score))

            # print(visualization_data)

            
            class CombinedModel(nn.Module):
                def __init__(self, bert_c, model_c):
                    super().__init__()
                    self.bert_c = bert_c
                    self.model_c = model_c
                    
                    self.bert_c.train()
                    self.model_c.train()
                    # Freeze the BERT model parameters
                    # for param in self.bert.parameters():
                    #     param.requires_grad = False

                def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, inputs_embeds=None):
                    # Get the input embeddings
                    if inputs_embeds is None:
                        inputs_embeds = self.bert_c.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
                    
                    outputs = self.bert_c(inputs_embeds=inputs_embeds, 
                              attention_mask=attention_mask,
                              output_hidden_states=True)
        
                    last_hidden_states = torch.cat(outputs.hidden_states[-4:], dim=0)
                    
                    output = self.model_c(last_hidden_states)
                    return output
                
                def get_embeddings(self, input_ids, token_type_ids):
                    return self.bert_c.bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

            combined_model = CombinedModel(self.bert, self.model)

            # Prepare your input
            input_ids = feat
            attention_mask = attention
            token_type_ids = typeid

            # Get the embeddings
            embeddings = combined_model.get_embeddings(input_ids, token_type_ids)
            embeddings.requires_grad_()

            # Create a wrapper function for Saliency
            def model_forward(embedded_inputs):
                return combined_model(attention_mask=attention_mask, token_type_ids=token_type_ids, inputs_embeds=embedded_inputs)

            # Create an instance of Saliency
            saliency = Saliency(model_forward)

            # Compute saliency
            saliency_scores = saliency.attribute(embeddings, target=0)  # Adjust target as needed

            # Process saliency scores
            saliency_sum = saliency_scores.sum(dim=-1).squeeze(0)
            saliency_sum = saliency_sum / torch.norm(saliency_sum)
            saliency_sum = saliency_sum.abs().detach().cpu().numpy()

            # Visualize the saliency scores
            tokens = self.token.convert_ids_to_tokens(input_ids[0])

            plt.figure(figsize=(20, 5))
            plt.bar(range(len(tokens)), saliency_sum[:len(tokens)])
            plt.xticks(range(len(tokens)), tokens, rotation=90)
            plt.xlabel('Tokens')
            plt.ylabel('Saliency Score')
            plt.title('Token Saliency')
            plt.tight_layout()
            plt.show()

            # TensorBoard
            writer.add_figure('Combine ' + file_name, plt.gcf(), global_step=0)

            # # Use Saliency
            # saliency = Saliency(wrapped_model)
            # attr = saliency.attribute(inputs=(input_ids, attention_mask, token_type_ids), target=0)  # Assuming target class is 1 (dementia)

            # # Sum the attributions across embedding dimensions
            # attr_sum = attr.sum(dim=-1).squeeze(0).detach().cpu().numpy()

            # # Tokenize input text
            # tokens = self.token.convert_ids_to_tokens(input_ids.squeeze().tolist())

            # # Visualize attributions for each token
            # visualization_data = [(token, score) for token, score in zip(tokens, attr_sum)]

            # viz.visualize_text([visualization_data])

            # # Visualize the saliency map
            # attr = attr.squeeze().cpu().numpy()
            # plt.imshow(attr, aspect='auto', cmap='hot')
            # plt.title("Saliency Map")
            # plt.xlabel("Time")
            # plt.ylabel("Frequency")
            # plt.colorbar()
            # plt.show()

            # TensorBoard
            # writer.add_figure('Saliency' + file_name, plt.gcf(), global_step=0)


            # # Use Layer Grad-CAM
            # layer_gc = LayerGradCam(wrapped_model, last_hidden_state)
            # attr = layer_gc.attribute(feat, target=0)  # Assuming target class is 1 (dementia)

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

            # # Configure interpretable embedding layer
            # interpretable_emb = configure_interpretable_embedding_layer(wrapped_model, 'embedding')

            # # Convert input ids to embeddings
            # input_emb = interpretable_emb.indices_to_embeddings(input_ids)
            # print(input_emb)

            # # Apply Integrated Gradients
            # ig = IntegratedGradients(wrapped_model)
            # attribution = ig.attribute((input_ids, attention_mask, token_type_ids), target=0)

            # # Remove interpretable embedding layer
            # remove_interpretable_embedding_layer(wrapped_model, interpretable_emb)3

            # print(feat_len, txt, txt_len, audio, file)


