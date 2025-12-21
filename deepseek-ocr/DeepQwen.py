import torch
import torch.nn as nn
from typing import Optional
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel, Qwen2_5_VLConfig, Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modular_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    PatchMerger,
    Qwen2RMSNorm,
)
from .modeling_deepseekocr import DeepseekOCRConfig, DeepseekOCRModel, MlpProjector
import os
import math

class Qwen2_5_VLPatchMerger(PatchMerger):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__(dim, context_dim, spatial_merge_size)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)

class DeepseekOCRVisualAdapter(nn.Module):
    def __init__(self, spatial_merge_size: int = 2, pretrained_vision_path: str = None):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        
        deepseek_config = DeepseekOCRConfig()
        self.deepseek_vision = DeepseekOCRModel(deepseek_config)
        
        if pretrained_vision_path is not None:
            try:
                from transformers import AutoModel
                pretrained_model = AutoModel.from_pretrained(pretrained_vision_path, trust_remote_code=True)
                self.deepseek_vision.load_state_dict(pretrained_model.state_dict(), strict=False)
                print(f"loaded {pretrained_vision_path}")
            except Exception as e:
                print(f"failed to load {pretrained_vision_path}: {e}")
        
        self.merger = MlpProjector(
            input_dim=deepseek_config.proj_dim,
            output_dim=deepseek_config.proj_dim,
            spatial_merge_size=spatial_merge_size)
    
    def forward(self, pixel_values: torch.Tensor, grid_thw: Optional[torch.LongTensor] = None, **kwargs):
   
        with torch.no_grad():
            vision_features = self.deepseek_vision.sam_model(pixel_values)  
            image_embeds = self.deepseek_vision.projector(vision_features) 
        
        image_embeds = self.merger(image_embeds)
        
        return image_embeds


class DeepQwen2_5VLConfig(Qwen2_5_VLConfig):

    model_type = "DeepQwen2_5VL"
    
    def __init__(self, text_config=None, vision_config=None, **kwargs):
        if text_config is None:
            text_config = Qwen2_5_VLTextConfig()
        if vision_config is None:
            vision_config = Qwen2_5_VLVisionConfig()
        
        self.text_config = text_config
        self.vision_config = vision_config
        
        super().__init__(
            text_config=text_config,
            vision_config=vision_config,
            **kwargs
        )

class DeepQwen2_5VLModel(Qwen2_5_VLModel):
    config_class = DeepQwen2_5VLConfig

    def __init__(
        self, 
        config: DeepQwen2_5VLConfig,
        pretrained_vision_path: str = "deepseek-ai/deepseek-ocr",
        pretrained_text_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    ):
        super().__init__(config)
        
        self.visual = DeepseekOCRVisualAdapter(
            spatial_merge_size=config.vision_config.spatial_merge_size,
            pretrained_vision_path=pretrained_vision_path
        )
        
        try:
            from transformers import AutoModel
            pretrained_qwen = AutoModel.from_pretrained(pretrained_text_path, trust_remote_code=True)
            
            if hasattr(pretrained_qwen, 'language_model'):
                self.language_model.load_state_dict(pretrained_qwen.language_model.state_dict())
            
            print(f"loaded {pretrained_text_path}")
        except Exception as e:
            print(f"failed to load {pretrained_text_path}: {e}")
            print("Using randomly initialized text model")

        
class DeepQwen2_5VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    config_class = DeepQwen2_5VLConfig

    def __init__(
        self,
        config: DeepQwen2_5VLConfig,
        pretrained_vision_path: str = "deepseek-ai/deepseek-ocr",
        pretrained_text_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    ):
        super().__init__(config)
        
        self.model.visual = DeepseekOCRVisualAdapter(
            spatial_merge_size=config.vision_config.spatial_merge_size,
            pretrained_vision_path=pretrained_vision_path
        )
        
        try:
            from transformers import AutoModelForCausalLM
            pretrained_qwen = AutoModelForCausalLM.from_pretrained(
                pretrained_text_path, 
                trust_remote_code=True
            )
            
            if hasattr(pretrained_qwen.model, 'text_model'):
                self.model.text_model.load_state_dict(
                    pretrained_qwen.model.text_model.state_dict()
                )
            
            if hasattr(pretrained_qwen, 'lm_head'):
                self.lm_head.load_state_dict(pretrained_qwen.lm_head.state_dict())
            
            print(f"loaded {pretrained_text_path}")
        except Exception as e:
            print(f"failed to load {pretrained_text_path}: {e}")
            print("Using randomly initialized model")

        def infer(self, tokenizer, prompt='', image_file='', output_path = '', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False, eval_mode=False):
            self.disable_torch_init()

            os.makedirs(output_path, exist_ok=True)
            os.makedirs(f'{output_path}/images', exist_ok=True)

            if prompt and image_file:
                conversation = [
                    {
                        "role": "<|User|>",
                        # "content": "<image>\n<|grounding|>Given the layout of the image. ",
                        "content": f'{prompt}',
                        # "content": "君不见黄河之水天上来的下一句是什么？",
                        # "content": "<image>\nFree OCR. ",
                        # "content": "<image>\nParse the figure. ",
                        # "content": "<image>\nExtract the text in the image. ",
                        "images": [f'{image_file}'],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
            
            elif prompt:
                conversation = [
                    {
                        "role": "<|User|>",
                        # "content": "<image>\n<|grounding|>Given the layout of the image. ",
                        "content": f'{prompt}',
                        # "content": "君不见黄河之水天上来的下一句是什么？",
                        # "content": "<image>\nFree OCR. ",
                        # "content": "<image>\nParse the figure. ",
                        # "content": "<image>\nExtract the text in the image. ",
                        # "images": [f'{image_file}'],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
            else:
                assert False, f'prompt is none!'
            
            prompt = format_messages(conversations=conversation, sft_format='plain', system_prompt='')

            patch_size = 16
            downsample_ratio = 4
            images = load_pil_images(conversation)

            valid_img_tokens = 0
            ratio = 1

            image_draw = images[0].copy()

            w,h = image_draw.size
            # print(w, h)
            ratio = 1 - ((max(w, h) - min(w, h)) / (max(w, h)))
        

            image_transform=BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
            images_seq_mask = []

            image_token = '<image>'
            image_token_id = 128815
            text_splits = prompt.split(image_token)

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

            """add the bos tokens"""
            bos_id = 0
            tokenized_str = [bos_id] + tokenized_str 
            images_seq_mask = [False] + images_seq_mask



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
                    

            if '<image>' in conversation[0]['content'] and eval_mode:
                    outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).cuda().shape[1]:])
                    stop_str = '<｜end▁of▁sentence｜>'
                    if outputs.endswith(stop_str):
                        outputs = outputs[:-len(stop_str)]
                    # re_match
                    outputs = outputs.strip()

                    return outputs
            
            if '<image>' in conversation[0]['content'] and test_compress:
                outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).cuda().shape[1]:])
                pure_texts_outputs_token_length = len(text_encode(tokenizer, outputs, bos=False, eos=False))
                print('='*50)
                print('image size: ', (w, h))
                print('valid image tokens: ', int(valid_img_tokens))
                print('output texts tokens (valid): ', pure_texts_outputs_token_length)
                print('compression ratio: ', round(pure_texts_outputs_token_length/valid_img_tokens, 2))
                print('='*50)


            if '<image>' in conversation[0]['content'] and save_results:
                outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).cuda().shape[1]:])
                stop_str = '<｜end▁of▁sentence｜>'

                print('='*15 + 'save results:' + '='*15)
                
                # # # # conv.messages[-1][-1] = outputs
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