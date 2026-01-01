import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
from transformers import Qwen2VLTextModel, Qwen2VLTextConfig, Qwen2VLPreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from PIL import Image, ImageOps
from .deepencoder import build_sam_vit_b, build_clip_l, MlpProjector
from addict import Dict as ADict
import os
import math
from .preprocessors_utils_dq import (
    format_messages,
    load_pil_images,
    text_encode,
    BasicImageTransform,
    dynamic_preprocess,
    re_match,
    process_image_with_refs,
    NoEOSTextStreamer,
)
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class DeepQwenOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None

@dataclass
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
            return_dict=True
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


    def infer(self, tokenizer, prompt='', image_file='', output_path = '', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False, eval_mode=False):
        self.disable_torch_init()

        # Qwen2.5-VL native message format
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"{image_file}",
                    },
                    {"type": "text", "text": f"{prompt}"},
                ],
            }
        ]
        
        # Use tokenizer's chat template to format the prompt
        formatted_prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        patch_size = 16
        downsample_ratio = 4
        images = load_pil_images(conversation)

        valid_img_tokens = 0
        ratio = 1

        image_draw = images[0].copy()

        w,h = image_draw.size
        ratio = 1 - ((max(w, h) - min(w, h)) / (max(w, h)))
    

        image_transform=BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
        images_seq_mask = []

        # Qwen2VL tokens
        image_token = '<|image_pad|>'  # Qwen2VL's image placeholder
        image_token_id = 151655  # Qwen2VL's <|image_pad|> token ID
        text_splits = formatted_prompt.split(image_token)

        images_list, images_crop_list, images_seq_mask = [], [], []
        tokenized_str = []
        images_spatial_crop = []
        for text_sep, image in zip(text_splits, images):

            tokenized_sep = text_encode(tokenizer, text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            if crop_mode:

                if image.size[0] <= 640 and image.size[1] <= 640:
                    crop_ratio = [1, 1]

                else:
                    if crop_mode:
                        # best_width, best_height = select_best_resolution(image.size, self.candidate_resolutions)
                        images_crop_raw, crop_ratio = dynamic_preprocess(image)
                    else:
                        # best_width, best_height = self.image_size, self.image_size
                        crop_ratio = [1, 1]
                
                """process the global view"""
                # image = image.resize((base_size, base_size))
                global_view = ImageOps.pad(image, (base_size, base_size),
                                        color=tuple(int(x * 255) for x in image_transform.mean))
                
                if base_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif base_size == 1280:
                    valid_img_tokens += int(400 * ratio)
                # elif base_size == 640:
                #     valid_img_tokens += int(100 * ratio)
                



                
                images_list.append(image_transform(global_view).to(torch.bfloat16))

                # global_view_tensor = image_transform(global_view).to(torch.bfloat16)

                width_crop_num, height_crop_num = crop_ratio

                images_spatial_crop.append([width_crop_num, height_crop_num])
                
                
                if width_crop_num > 1 or height_crop_num > 1:
                    """process the local views"""
                    
                    for i in range(len(images_crop_raw)):
                        images_crop_list.append(image_transform(images_crop_raw[i]).to(torch.bfloat16))
                
                if image_size == 640:
                    valid_img_tokens += len(images_crop_list) * 100

                num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
                num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)



                """add image tokens"""

                

                tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
                tokenized_image += [image_token_id]
                if width_crop_num > 1 or height_crop_num > 1:
                    tokenized_image += ([image_token_id] * (num_queries * width_crop_num) + [image_token_id]) * (
                                num_queries * height_crop_num)
                tokenized_str += tokenized_image
                images_seq_mask += [True] * len(tokenized_image)
                # num_image_tokens.append(len(tokenized_image))

            else:
                # best_width, best_height = self.image_size, self.image_size
                # print(image.size, (best_width, best_height)) # check the select_best_resolutions func

                """process the global view"""
                if image_size <= 640:
                    print('directly resize')
                    image = image.resize((image_size, image_size))
                # else:
                global_view = ImageOps.pad(image, (image_size, image_size),
                                        color=tuple(int(x * 255) for x in image_transform.mean))
                images_list.append(image_transform(global_view).to(torch.bfloat16))

                if base_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif base_size == 1280:
                    valid_img_tokens += int(400 * ratio)
                elif base_size == 640:
                    valid_img_tokens += int(100 * 1)
                elif base_size == 512:
                    valid_img_tokens += int(64 * 1)

                width_crop_num, height_crop_num = 1, 1

                images_spatial_crop.append([width_crop_num, height_crop_num])


                """add image tokens"""
                num_queries = math.ceil((image_size // patch_size) / downsample_ratio)

                tokenized_image = ([image_token_id] * num_queries + [image_token_id]) * num_queries
                tokenized_image += [image_token_id]
                # tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]) * (
                #             num_queries * height_crop_num)
                tokenized_str += tokenized_image
                images_seq_mask += [True] * len(tokenized_image)
                # num_image_tokens.append(len(tokenized_image))
        

        """process the last text split"""
        tokenized_sep = text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        # Qwen2VL has NO bos_token (bos_token_id is None)
        # The chat template already handles proper formatting

        input_ids = torch.LongTensor(tokenized_str)


        

        images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)


        if len(images_list) == 0:
            images_ori = torch.zeros((1, 3, image_size, image_size))
            images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)
            images_crop = torch.zeros((1, 3, base_size, base_size))

        else:
            images_ori = torch.stack(images_list, dim=0)
            images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0)
            else:
                images_crop = torch.zeros((1, 3, base_size, base_size))



        if not eval_mode:
            streamer = NoEOSTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    output_ids = self.generate(
                        input_ids.unsqueeze(0).cuda(),
                        images=[(images_crop.cuda(), images_ori.cuda())],
                        images_seq_mask = images_seq_mask.unsqueeze(0).cuda(),
                        images_spatial_crop = images_spatial_crop,
                        # do_sample=False,
                        # num_beams = 1,
                        temperature=0.0,
                        eos_token_id=tokenizer.eos_token_id,
                        streamer=streamer,
                        max_new_tokens=8192,
                        no_repeat_ngram_size = 20,
                        use_cache = True
                        )

        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    output_ids = self.generate(
                        input_ids.unsqueeze(0).cuda(),
                        images=[(images_crop.cuda(), images_ori.cuda())],
                        images_seq_mask = images_seq_mask.unsqueeze(0).cuda(),
                        images_spatial_crop = images_spatial_crop,
                        # do_sample=False,
                        # num_beams = 1,
                        temperature=0.0,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=8192,
                        no_repeat_ngram_size = 35,
                        use_cache = True
                        )
                

        # Check if conversation has image (handle both string and list content formats)
        has_image = any(
            (isinstance(item, dict) and item.get('type') == 'image')
            for msg in conversation
            for item in (msg.get('content', []) if isinstance(msg.get('content'), list) else [])
        )
        
        if has_image and eval_mode:
                outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).cuda().shape[1]:], skip_special_tokens=False)
                # Qwen2VL's EOS token is <|im_end|>
                stop_str = tokenizer.eos_token or '<|im_end|>'
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()

                return outputs
        
        if has_image and test_compress:
            outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).cuda().shape[1]:], skip_special_tokens=False)
            pure_texts_outputs_token_length = len(text_encode(tokenizer, outputs, bos=False, eos=False))
            print('='*50)
            print('image size: ', (w, h))
            print('valid image tokens: ', int(valid_img_tokens))
            print('output texts tokens (valid): ', pure_texts_outputs_token_length)
            print('compression ratio: ', round(pure_texts_outputs_token_length/valid_img_tokens, 2))
            print('='*50)


        if has_image and save_results:
            outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).cuda().shape[1]:], skip_special_tokens=False)
            # Qwen2VL's EOS token
            stop_str = tokenizer.eos_token or '<|im_end|>'

            print('='*15 + 'save results:' + '='*15)
            
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            matches_ref, matches_images, mathes_other = re_match(outputs)
            # print(matches_ref)
            result = process_image_with_refs(image_draw, matches_ref, output_path)


            for idx, a_match_image in enumerate(tqdm(matches_images, desc="image")):
                outputs = outputs.replace(a_match_image, '![](images/' + str(idx) + '.jpg)\n')
            
            for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
                outputs = outputs.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')


            # if 'structural formula' in conversation[0]['content']:
            #     outputs = '<smiles>' + outputs + '</smiles>'
            with open(f'{output_path}/result.mmd', 'w', encoding = 'utf-8') as afile:
                afile.write(outputs)

            if 'line_type' in outputs:
                import matplotlib.pyplot as plt
                lines = eval(outputs)['Line']['line']

                line_type = eval(outputs)['Line']['line_type']
                # print(lines)

                endpoints = eval(outputs)['Line']['line_endpoint']

                fig, ax = plt.subplots(figsize=(3,3), dpi=200)
                ax.set_xlim(-15, 15)
                ax.set_ylim(-15, 15)

                for idx, line in enumerate(lines):
                    try:
                        p0 = eval(line.split(' -- ')[0])
                        p1 = eval(line.split(' -- ')[-1])

                        if line_type[idx] == '--':
                            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color='k')
                        else:
                            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth = 0.8, color = 'k')

                        ax.scatter(p0[0], p0[1], s=5, color = 'k')
                        ax.scatter(p1[0], p1[1], s=5, color = 'k')
                    except:
                        pass

                for endpoint in endpoints:

                    label = endpoint.split(': ')[0]
                    (x, y) = eval(endpoint.split(': ')[1])
                    ax.annotate(label, (x, y), xytext=(1, 1), textcoords='offset points', 
                                fontsize=5, fontweight='light')
                

                plt.savefig(f'{output_path}/geo.jpg')
                plt.close()

            result.save(f"{output_path}/result_with_boxes.jpg")
