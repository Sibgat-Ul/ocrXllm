import torch
import torch.nn as nn
from typing import Optional
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel, Qwen2_5_VLConfig, Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modular_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    PatchMerger,
    Qwen2RMSNorm,
)
from .modeling_deepseekocr import DeepseekOCRConfig, DeepseekOCRModel

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
        
        self.merger = Qwen2_5_VLPatchMerger(
            dim=2048,                   
            context_dim=1280,    
            spatial_merge_size=spatial_merge_size
        )
    
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
