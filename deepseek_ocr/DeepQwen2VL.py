import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
from transformers import Qwen2VLTextModel, Qwen2VLTextConfig, Qwen2VLModel
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    PatchMerger,
    Qwen2VLCausalLMOutputWithPast,
)

from transformers.cache_utils import Cache
from .deepencoder import build_sam_vit_b, build_clip_l, MlpProjector
from addict import Dict as ADict
import os
import math


class DeepQwenVLModelForCausalLM(Qwen2VLTextModel):
    def __init__(
        self,
        text_config: Qwen2VLTextConfig,
        output_hidden_size: int = 2048,  
    ):
        super(DeepQwenVLModelForCausalLM, self).__init__(text_config)
        
        self.output_hidden_size = output_hidden_size
        
        self.sam_model = build_sam_vit_b() 
        self.vision_model = build_clip_l() 
        
        self.deepseek_hidden_size = 2048
        
        self.projector = MlpProjector(
            ADict(projector_type="linear", input_dim=2048, n_embed=output_hidden_size)
        )
        
        embed_std = 1 / torch.sqrt(torch.tensor(output_hidden_size, dtype=torch.float32))
        self.image_newline = nn.Parameter(torch.randn(output_hidden_size) * embed_std)
        self.view_separator = nn.Parameter(torch.randn(output_hidden_size) * embed_std)
        
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.FloatTensor] = None,
        images_spatial_crop: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        sam_model = getattr(self, 'sam_model', None)
        vision_model = getattr(self, 'vision_model', None)

        if sam_model is not None and (input_ids.shape[1] != 1 or self.training) and torch.sum(images[0][1]).item() != 0:

            idx = 0
            
            for image, crop_shape in zip(images, images_spatial_crop):
                images_in_this_batch = []

                patches = image[0]
                image_ori = image[1]

                with torch.no_grad():
                # with torch.inference_mode(): 
                    
                    if torch.sum(patches).item() != 0:
                        # P, C, H, W = patches.shape
                        crop_flag = 1
                        local_features_1 = sam_model(patches)

                        local_features_2 = vision_model(patches, local_features_1)  
                        # vit_time = time.time()
                        local_features = torch.cat((local_features_2[:, 1:], local_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
                        local_features = self.projector(local_features)

                        global_features_1 = sam_model(image_ori)
                        global_features_2 = vision_model(image_ori, global_features_1) 
                        global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
                        global_features = self.projector(global_features)

                        print('=====================')
                        print('BASE: ', global_features.shape)
                        print('PATCHES: ', local_features.shape)
                        print('=====================')

                        _, hw, n_dim = global_features.shape
                        h = w = int(hw ** 0.5)

                        _2, hw2, n_dim2 = local_features.shape
                        h2 = w2 = int(hw2 ** 0.5)

                        width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]

                        global_features = global_features.view(h, w, n_dim)

                        global_features = torch.cat(
                            [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                        )

                        global_features = global_features.view(-1, n_dim)


                        local_features = local_features.view(height_crop_num, width_crop_num, h2, w2, n_dim2).permute(0, 2, 1, 3, 4).reshape(height_crop_num*h2, width_crop_num*w2, n_dim2)
                        local_features = torch.cat(
                            [local_features, self.image_newline[None, None, :].expand(height_crop_num * h2, 1, n_dim2)], dim=1
                        )
                        local_features = local_features.view(-1, n_dim2)

                        global_local_features = torch.cat([local_features, global_features, self.view_separator[None, :]], dim=0)

                        # end_time = time.time()

                        # print('sam: ', sam_time - start_time)
                        # print('vit: ', vit_time - sam_time)
                        # print('all: ', end_time - start_time)

                        # exit()
                   
                    else:
                        global_features_1 = sam_model(image_ori)
                        global_features_2 = vision_model(image_ori, global_features_1) 
                        global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
                        global_features = self.projector(global_features)
                        print('=====================')
                        print('BASE: ', global_features.shape)
                        print('NO PATCHES')
                        print('=====================')
                        _, hw, n_dim = global_features.shape
                        h = w = int(hw ** 0.5)


                        global_features = global_features.view(h, w, n_dim)

                        global_features = torch.cat(
                            [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                        )

                        global_features = global_features.view(-1, n_dim)

                        global_local_features = torch.cat([global_features, self.view_separator[None, :]], dim=0)

                    images_in_this_batch.append(global_local_features)
                

                # print(inputs_embeds.shape)

                if images_in_this_batch:
                    images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                    # exit()

                    inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1).cuda(), images_in_this_batch)

                idx += 1

        outputs = super(DeepQwenVLModelForCausalLM, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, position_ids = position_ids,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # loss = None
        # if labels is not None:
        #     from torch.nn import CrossEntropyLoss
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     loss_fct = CrossEntropyLoss()
        #     shift_logits = shift_logits.view(-1, self.config.vocab_size)
        #     shift_labels = shift_labels.view(-1)
        #     shift_labels = shift_labels.to(shift_logits.device)
        #     loss = loss_fct(shift_logits, shift_labels)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def load_pretrained_vision(self, pretrained_path: str):
        """Load pretrained DeepSeek OCR vision weights."""
        try:
            from transformers import AutoModelForCausalLM
            pretrained = AutoModelForCausalLM.from_pretrained(
                pretrained_path, 
                trust_remote_code=True
            )
            
            # Load SAM weights
            if hasattr(pretrained, 'model') and hasattr(pretrained.model, 'sam_model'):
                self.sam_model.load_state_dict(
                    pretrained.model.sam_model.state_dict(), 
                    strict=False
                )
                print(f"Loaded SAM weights from {pretrained_path}")
            
            # Load CLIP weights
            if hasattr(pretrained, 'model') and hasattr(pretrained.model, 'vision_model'):
                self.vision_model.load_state_dict(
                    pretrained.model.vision_model.state_dict(), 
                    strict=False
                )
                print(f"Loaded CLIP weights from {pretrained_path}")
            
            # Load projector weights (may need adjustment if dimensions differ)
            if hasattr(pretrained, 'model') and hasattr(pretrained.model, 'projector'):
                try:
                    self.projector.load_state_dict(
                        pretrained.model.projector.state_dict(), 
                        strict=False
                    )
                    print(f"Loaded projector weights from {pretrained_path}")
                except RuntimeError as e:
                    print(f"Projector dimension mismatch, will train from scratch: {e}")
            
            # Load special tokens if available
            if hasattr(pretrained, 'model') and hasattr(pretrained.model, 'image_newline'):
                self.image_newline.data.copy_(pretrained.model.image_newline.data)
                print(f"Loaded image_newline from {pretrained_path}")
            
            if hasattr(pretrained, 'model') and hasattr(pretrained.model, 'view_seperator'):
                self.view_separator.data.copy_(pretrained.model.view_seperator.data)
                print(f"Loaded view_separator from {pretrained_path}")
                
            del pretrained
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed to load pretrained vision weights from {pretrained_path}: {e}")

__all__ = [
    "DeepQwenVLModelForCausalLM",
]
