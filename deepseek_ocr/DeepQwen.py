import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel, Qwen2_5_VLConfig, Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modular_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    PatchMerger,
    Qwen2RMSNorm,
    Qwen2VLModelOutputWithPast,
)
from transformers.cache_utils import Cache
from .deepencoder import build_sam_vit_b, build_clip_l, MlpProjector
from .modeling_deepseekocr import DeepseekOCRConfig
from addict import Dict as ADict
import os
import math


class DeepseekVisionEncoder(nn.Module):
    def __init__(
        self, 
        output_hidden_size: int = 2048,  
        use_spatial_merge: bool = False,  
        spatial_merge_size: int = 2, 
    ):
        super().__init__()
        
        self.output_hidden_size = output_hidden_size
        self.use_spatial_merge = use_spatial_merge
        self.spatial_merge_size = spatial_merge_size
        
        self.sam_model = build_sam_vit_b() 
        self.vision_model = build_clip_l() 
        
        self.deepseek_hidden_size = 2048
        
        self.projector = MlpProjector(
            ADict(projector_type="linear", input_dim=2048, n_embed=output_hidden_size)
        )
        
        embed_std = 1 / torch.sqrt(torch.tensor(output_hidden_size, dtype=torch.float32))
        self.image_newline = nn.Parameter(torch.randn(output_hidden_size) * embed_std)
        self.view_separator = nn.Parameter(torch.randn(output_hidden_size) * embed_std)
        
        if use_spatial_merge and spatial_merge_size > 1:
            self.spatial_merger = PatchMerger(
                d_model=output_hidden_size,
                context_dim=output_hidden_size,
                spatial_merge_size=spatial_merge_size,
            )
        else:
            self.spatial_merger = None

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
            # inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = self.get_input_embeddings()(input_ids)

        sam_model = getattr(self, 'sam_model', None)
        # sam_model = self.sam_model
        vision_model = getattr(self, 'vision_model', None)

        if sam_model is not None and (input_ids.shape[1] != 1 or self.training) and torch.sum(images[0][1]).item() != 0:

            idx = 0
            
            # sam_model = torch.jit.script(sam_model)
            # start_time = time.time()

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

                        global_local_features = torch.cat([local_features, global_features, self.view_seperator[None, :]], dim=0)

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

                        global_local_features = torch.cat([global_features, self.view_seperator[None, :]], dim=0)

                    images_in_this_batch.append(global_local_features)
                

                # print(inputs_embeds.shape)

                if images_in_this_batch:
                    images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                    # exit()

                    inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1).cuda(), images_in_this_batch)

                idx += 1
        if self.spatial_merger is not None:
            inputs_embeds = self.spatial_merger(
                inputs_embeds,
            )
            
        else:
            return inputs_embeds
    
    def load_pretrained_vision(self, pretrained_path: str):
        """Load pretrained DeepSeek OCR vision weights."""
        try:
            pretrained = torch.load(
                pretrained_path, 
                map_location='cpu'
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

from transformers import Qwen2VLModel, Qwen2VLConfig, PreTrainedModel, Qwen2VLModelOutputWithPast

class DeepQwen2VLModelOutputWithPast(Qwen2VLModelOutputWithPast):
    pass

class DeepQwen2VLConfig(Qwen2VLConfig):
    """
    Configuration for the hybrid DeepQwen model.
    
    Key differences from Qwen2-VL config:
    - deepseek_vision_hidden_size for combined SAM+CLIP features (2048)
    - use_deepseek_structure to control token formatting
    """
    
    model_type = "DeepQwen2VL"
    
    def __init__(
        self, 
        text_config=None, 
        vision_config=None,
        deepseek_vision_hidden_size: int = 2048,
        use_spatial_merge: bool = False,
        **kwargs
    ):
        if text_config is None:
            text_config = Qwen2VLConfig().text_config
        if vision_config is None:
            vision_config = Qwen2VLConfig().vision_config
        
        self.deepseek_vision_hidden_size = deepseek_vision_hidden_size
        self.use_spatial_merge = use_spatial_merge
        
        super().__init__(
            text_config=text_config,
            vision_config=vision_config,
            **kwargs
        )

class DeepQwen2VL(PreTrainedModel):
    config: DeepQwen2VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2VLDecoderLayer", "Qwen2VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.get_text_config().initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, Qwen2RMSNorm):
            module.weight.data.fill_(1.0)

class DeepQwen2VL(Qwen2VLModel):
    config_class = DeepQwen2VLConfig

    def __init__(self, config: DeepQwen2VLConfig):
        # Important: Call grandparent __init__ to avoid creating Qwen's visual encoder
        nn.Module.__init__(self)
        self.config = config
        
        self.visual = DeepseekVisionEncoder(
            output_hidden_size=config.text_config.hidden_size,
            use_spatial_merge=getattr(config, 'use_spatial_merge', False),
            spatial_merge_size=getattr(config.vision_config, 'spatial_merge_size', 2),
        )
        
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLTextModel
        
        self.language_model = Qwen2VLTextModel._from_config(config.text_config)
        
        self.post_init()
    
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

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
        return_dict: Optional[bool] = None
    ) -> Union[tuple, Qwen2VLModelOutputWithPast]:
        ## TO BE IMPLEMENTED: Similar to DeepQwen2_5VLModel forward()
        raise NotImplementedError("DeepQwen2VLModel forward() not implemented yet.")
    pass
    
class DeepQwen2_5VLConfig(Qwen2_5_VLConfig):
    """
    Configuration for the hybrid DeepQwen model.
    
    Key differences from Qwen2.5-VL config:
    - No spatial_merge_size needed (DeepSeek uses different token structure)
    - deepseek_vision_hidden_size for combined SAM+CLIP features (2048)
    - use_deepseek_structure to control token formatting
    """
    
    model_type = "DeepQwen2_5VL"
    
    def __init__(
        self, 
        text_config=None, 
        vision_config=None,
        deepseek_vision_hidden_size: int = 2048,
        use_spatial_merge: bool = False,
        **kwargs
    ):
        if text_config is None:
            text_config = Qwen2_5_VLTextConfig()
        if vision_config is None:
            vision_config = Qwen2_5_VLVisionConfig()
        
        self.deepseek_vision_hidden_size = deepseek_vision_hidden_size
        self.use_spatial_merge = use_spatial_merge
        
        super().__init__(
            text_config=text_config,
            vision_config=vision_config,
            **kwargs
        )


class DeepQwen2_5VLModel(Qwen2_5_VLModel):
    """
    Hybrid model combining DeepSeek OCR's vision encoder with Qwen2.5-VL's language model.
    
    Key differences from Qwen2_5_VLModel:
    - Uses DeepseekVisionEncoder instead of Qwen's visual encoder
    - Does NOT use spatial_merge_size (DeepSeek has different token organization)
    - Vision output is directly projected to language model dimension
    """
    
    config_class = DeepQwen2_5VLConfig

    def __init__(self, config: DeepQwen2_5VLConfig):
        # Important: Call grandparent __init__ to avoid creating Qwen's visual encoder
        nn.Module.__init__(self)
        self.config = config
        
        # Initialize DeepSeek vision encoder
        self.visual = DeepseekVisionEncoder(
            output_hidden_size=config.text_config.hidden_size,
            use_spatial_merge=getattr(config, 'use_spatial_merge', False),
            spatial_merge_size=getattr(config.vision_config, 'spatial_merge_size', 2),
        )
        
        # Initialize Qwen's language components (from Qwen2_5_VLModel)
        # These will be loaded from pretrained weights
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel
        
        # Initialize language model properly
        self.language_model = Qwen2_5_VLTextModel._from_config(config.text_config)
        
        # Post init
        self.post_init()
    
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_features(
        self, 
        pixel_values: torch.FloatTensor, 
        image_grid_thw: Optional[torch.LongTensor] = None
    ) -> List[torch.Tensor]:
        """
        Encode images using DeepSeek's vision encoder.
        
        Note: Unlike Qwen2.5-VL, DeepSeek doesn't use grid_thw for spatial merging.
        The grid_thw is used here only to split outputs per image.
        
        Args:
            pixel_values: Preprocessed image tensor (B, C, H, W)
            image_grid_thw: Grid dimensions for each image (t, h, w) - used for splitting only
            
        Returns:
            List of image embeddings, one per image
        """
        # Get dtype from model parameters
        if hasattr(self.visual.projector, 'layers'):
            proj_dtype = self.visual.projector.layers[0].weight.dtype
        else:
            proj_dtype = next(self.visual.projector.parameters()).dtype
            
        pixel_values = pixel_values.type(proj_dtype)
        
        # Get image embeddings from DeepSeek encoder
        # Returns shape (B * num_patches, hidden_size)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        
        # Split by images based on grid dimensions
        if image_grid_thw is not None:
            # For DeepSeek encoder, calculate tokens differently than Qwen
            # Each 1024x1024 image produces 64x64 = 4096 patches from SAM/CLIP
            # But after projection, it's just flattened tokens
            B = pixel_values.shape[0]
            
            if image_embeds.dim() == 2:
                # Output is (total_tokens, hidden_size), need to split by image
                # Assume equal tokens per image when dimensions match
                num_images = image_grid_thw.shape[0]
                if num_images > 0:
                    tokens_per_image = image_embeds.shape[0] // num_images
                    image_embeds = list(torch.split(image_embeds, tokens_per_image))
                else:
                    image_embeds = [image_embeds]
            else:
                # Output is (B, num_patches, hidden_size)
                image_embeds = [img.squeeze(0) if img.dim() > 2 else img for img in image_embeds.unbind(0)]
        else:
            # Single image or no splitting needed
            if image_embeds.dim() == 2:
                image_embeds = [image_embeds]
            else:
                image_embeds = [img for img in image_embeds.unbind(0)]
            
        return image_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
      
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Process images if provided
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0)
            
            # Find image token positions
            if input_ids is None:
                image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                image_mask = image_mask.all(-1)
            else:
                image_mask = input_ids == self.config.image_token_id

            n_image_tokens = image_mask.sum()
            n_image_features = image_embeds.shape[0]
            
            # Validate token count matches
            if n_image_tokens != n_image_features:
                print(f"Warning: Image token mismatch - tokens: {n_image_tokens}, features: {n_image_features}")
                # Truncate or pad features to match
                if n_image_features > n_image_tokens:
                    image_embeds = image_embeds[:n_image_tokens]
                # If fewer features, this will fail - user needs to check preprocessing
            
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # Forward through language model
        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        return outputs

    @classmethod
    def from_pretrained_components(
        cls,
        config: DeepQwen2_5VLConfig,
        pretrained_vision_path: str = "deepseek-ai/deepseek-ocr",
        pretrained_text_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        **kwargs
    ):
        """
        Create a model with pretrained weights from both DeepSeek OCR and Qwen2.5-VL.
        """
        # Initialize the model
        model = cls(config)
        
        # Load DeepSeek vision weights
        model.visual.load_pretrained_vision(pretrained_vision_path)
        
        # Load Qwen language model weights
        try:
            from transformers import Qwen2_5_VLModel as QwenModel
            pretrained_qwen = QwenModel.from_pretrained(
                pretrained_text_path, 
                trust_remote_code=True
            )
            
            # Load language model weights
            if hasattr(pretrained_qwen, 'language_model'):
                model.language_model.load_state_dict(
                    pretrained_qwen.language_model.state_dict(),
                    strict=False
                )
                print(f"Loaded Qwen language model from {pretrained_text_path}")
            
            del pretrained_qwen
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed to load Qwen language model from {pretrained_text_path}: {e}")
            print("Using randomly initialized language model")
        
        return model


class DeepQwen2_5VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """
    DeepQwen model for conditional generation (e.g., image captioning, VQA).
    
    Combines DeepSeek OCR's vision encoder with Qwen2.5-VL's language model head.
    """
    
    config_class = DeepQwen2_5VLConfig

    def __init__(self, config: DeepQwen2_5VLConfig):
        super().__init__(config)
        
        # Replace the model with our hybrid version
        self.model = DeepQwen2_5VLModel(config)

    @classmethod
    def from_pretrained_components(
        cls,
        config: DeepQwen2_5VLConfig = None,
        pretrained_vision_path: str = "deepseek-ai/deepseek-ocr",
        pretrained_text_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        **kwargs
    ):
        """
        Create a model with pretrained weights from both DeepSeek OCR and Qwen2.5-VL.
        """
        # Load config from Qwen if not provided
        if config is None:
            from transformers import AutoConfig
            base_config = AutoConfig.from_pretrained(pretrained_text_path, trust_remote_code=True)
            config = DeepQwen2_5VLConfig(
                text_config=base_config.text_config if hasattr(base_config, 'text_config') else None,
                vision_config=base_config.vision_config if hasattr(base_config, 'vision_config') else None,
            )
        
        # Initialize the model
        model = cls(config)
        
        # Load DeepSeek vision weights
        model.model.visual.load_pretrained_vision(pretrained_vision_path)
        
        # Load Qwen weights
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as QwenCausalLM
            pretrained_qwen = QwenCausalLM.from_pretrained(
                pretrained_text_path, 
                trust_remote_code=True
            )
            
            # Load language model weights
            if hasattr(pretrained_qwen, 'model') and hasattr(pretrained_qwen.model, 'language_model'):
                model.model.language_model.load_state_dict(
                    pretrained_qwen.model.language_model.state_dict(),
                    strict=False
                )
                print(f"Loaded Qwen language model from {pretrained_text_path}")
            
            # Load lm_head weights
            if hasattr(pretrained_qwen, 'lm_head'):
                model.lm_head.load_state_dict(
                    pretrained_qwen.lm_head.state_dict()
                )
                print(f"Loaded lm_head from {pretrained_text_path}")
            
            del pretrained_qwen
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed to load Qwen weights from {pretrained_text_path}: {e}")
            print("Using randomly initialized model components")
        
        return model

    def infer(
        self, 
        processor,
        prompt: str = '',
        image_file: str = '',
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        **generate_kwargs
    ):
        """
        Run inference using Qwen2.5-VL's processor and generation pipeline.
        
        Args:
            processor: Qwen2.5-VL processor for tokenization and image preprocessing
            prompt: Text prompt for the model
            image_file: Path to the image file
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 for deterministic)
            **generate_kwargs: Additional arguments for generate()
            
        Returns:
            Generated text response
        """
        from PIL import Image
        
        # Prepare conversation
        messages = []
        if image_file:
            image = Image.open(image_file).convert('RGB')
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            })
        else:
            messages.append({
                "role": "user", 
                "content": [{"type": "text", "text": prompt}]
            })
        
        # Process inputs
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output_ids = self.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                **generate_kwargs
            )
        
        # Decode output
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        response = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return response.strip()

    def disable_torch_init(self):
        """Disable random weight initialization for faster loading."""
        import torch.nn.init as init
        setattr(init, "kaiming_uniform_", lambda x, *args, **kwargs: x)
        setattr(init, "uniform_", lambda x, *args, **kwargs: x)
        setattr(init, "normal_", lambda x, *args, **kwargs: x)


# Export all model classes
__all__ = [
    "DeepseekVisionEncoder",
    "DeepQwen2_5VLConfig", 
    "DeepQwen2_5VLModel",
    "DeepQwen2_5VLForConditionalGeneration",
]
