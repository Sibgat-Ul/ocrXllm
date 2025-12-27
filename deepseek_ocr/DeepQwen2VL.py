import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
from transformers import Qwen2VLTextModel, Qwen2VLTextConfig, Qwen2VLPreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from .deepencoder import build_sam_vit_b, build_clip_l, MlpProjector
from addict import Dict as ADict
import os
import math


class DeepQwenOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None

class DeepQwenCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None

class DeepQwenVLPreTrainedModel(PreTrainedModel):
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _supports_static_cache = True
    _supports_attention_backend = True


class DeepQwenVLModel(Qwen2VLTextModel):
    
    def __init__( self, text_config: Qwen2VLTextConfig, output_hidden_size: int = 2048):

        super(DeepQwenVLModel, self).__init__(text_config)
        
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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
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

        outputs = super().forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, position_ids = position_ids,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        output = DeepQwenOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        return output if return_dict else output.to_tuple() 


class DeepQwenVLForCausalLM(DeepQwenVLModel, GenerationMixin):
    def __init__(
        self,
        text_config: Qwen2VLTextConfig,
        output_hidden_size: int = 2048,  
    ):
        super().__init__(text_config, output_hidden_size=output_hidden_size)
        self.config = text_config
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

        outputs = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, 
            use_cache=use_cache, 
            position_ids = position_ids,
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, 
            images=images,
            images_seq_mask=images_seq_mask, 
            images_spatial_crop=images_spatial_crop,
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

        return DeepQwenCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask = None,
        inputs_embeds = None,
        cache_position = None,
        images = None,
        images_seq_mask = None,
        images_spatial_crop = None,
        **kwargs,
        ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            **kwargs,
        )

        # Qwen2-VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if model_inputs["cache_position"][0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs
    
    def load_pretrained_vision(self, pretrained_path: str):
        try:
            from safetensors import safe_open
        except ImportError:
            raise ImportError("Please install safetensors to load the pretrained vision model.")
        
        assert os.path.exists(pretrained_path), f"Pretrained path {pretrained_path} does not exist."

        vision_weights = {}
        with safe_open(f"{pretrained_path}/model-00001-of-000001.safetensors", framework="pt", device="cpu") as f:
            for k in f.keys():
                vision_weights[k] = f.get_tensor(k)
        
        prefixes = {
            "sam_model": "model.sam_model.",
            "vision_model": "model.vision_model.",
            "projector": "model.projector.",
        }

        try:
            for p in prefixes.keys():
                state_dict = {}

                for k, v in vision_weights.items():
                    if k.startswith(prefixes[p]):
                        new_key = k[len(prefixes[p]):]
                        state_dict[new_key] = v
                
                getattr(self, p).load_state_dict(state_dict, strict=False)
            
            print("Pretrained vision model loaded successfully.")
        except Exception as e:
            print("Error loading pretrained vision model:", e)
            raise e



__all__ = [
    "DeepQwenVLModel",
    "DeepQwenVLForCausalLM",
]
