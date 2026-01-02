from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import TextStreamer
from transformers.tokenization_utils import PreTrainedTokenizer as T
from abc import ABC
import re
import numpy as np


def load_image(image_path):
    try:
        image = Image.open(image_path)
        corrected_image = ImageOps.exif_transpose(image)

        return corrected_image
        
    except Exception as e:
        print(f"error: {e}")

        return None


def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    # pattern1 = r'<\|ref\|>.*?<\|/ref\|>\n'
    # new_text1 = re.sub(pattern1, '', text, flags=re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):

    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, ouput_path):

    image_width, image_height = image.size
    
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    font = ImageFont.load_default()

    img_idx = 0
    
    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                
                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))

                color_a = color + (20, )
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)

                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{ouput_path}/images/{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1
                        
                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        text_x = x1
                        text_y = max(0, y1 - 15)
                            
                        
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                                    fill=(255, 255, 255, 30))
                        
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, output_path):

    result_image = draw_bounding_boxes(image, ref_texts, output_path)
    
    return result_image


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, min_num=2, max_num=9, image_size=640, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    # print(f"target_ratios: {target_ratios}")

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, 
        target_ratios, 
        orig_width,
        orig_height, 
        image_size
    )
    # print(f"target_aspect_ratio: {target_aspect_ratio}")

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks
    # print(f"Number of processed images: {len(processed_images)}, Blocks: {blocks}")

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio


def normalize_transform(mean, std):
    if mean is None and std is None:
        transform = None
    elif mean is None and std is not None:
        mean = [0.] * len(std)
        transform = transforms.Normalize(mean=mean, std=std)
    elif mean is not None and std is None:
        std = [1.] * len(mean)
        transform = transforms.Normalize(mean=mean, std=std)
    else:
        transform = transforms.Normalize(mean=mean, std=std)

    return transform

def format_messages(
        tokenizer: T,
        conversations: List[Dict[str, str]],
        system_prompt: str = "",
):
    if system_prompt is not None and system_prompt != "":
        sys_prompt = {
            "role": "system",
            "content": system_prompt,
        }
        conversations = [sys_prompt] + conversations

    sft_prompt = tokenizer.apply_chat_template(
        conversations,
    )

    return sft_prompt


def text_encode(tokenizer, text: str, bos: bool = True, eos: bool = False):
    """
    Encode text with optional BOS/EOS tokens.
    
    Note: Qwen2VL tokenizer has bos_token_id=None, so we skip BOS for Qwen.
    The chat template handles special tokens automatically.
    """
    t = tokenizer.encode(text, add_special_tokens=False)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    # Only add BOS if tokenizer has one AND bos=True
    if bos and bos_id is not None:
        t = [bos_id] + t
    
    # Only add EOS if tokenizer has one AND eos=True
    if eos and eos_id is not None:
        t = t + [eos_id]

    return t

def load_pil_images(conversations: List[Dict[str, str]]) -> List[Image.Image]:
    pil_images = []

    for message in conversations:
        pil_image = None
    
        if message["role"].lower() == "user":
            if isinstance(message["content"], List):
                for d in message["content"]:
                    if d.get("type", "") == "image":
                        # Support both "image" (Qwen format) and "data" keys
                        image_path = d.get("image") or d.get("data", "")
                        pil_image = load_image(image_path)

            elif isinstance(message["content"], Dict):
                if message["content"].get("type", "") == "image":
                    # Support both "image" (Qwen format) and "data" keys
                    image_path = message["content"].get("image") or message["content"].get("data", "")
                    pil_image = load_image(image_path)

            if pil_image is not None:
                pil_images.append(pil_image)

    return pil_images


class BaseTransform(ABC):

    def set_rng(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass

    @property
    def default_shape(self):
        raise NotImplementedError


class BasicImageTransform(BaseTransform):
    def __init__(
        self, 
        mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        normalize: bool = True
    ):
        self.mean = mean
        self.std = std
    
        transform_pipelines = [
            transforms.ToTensor()
        ]

        normalize = normalize_transform(mean, std) if normalize else nn.Identity()
        if normalize is not None:
            transform_pipelines.append(normalize)

        self.transform = transforms.Compose(transform_pipelines)
    
    def __call__(self, x):
        x = self.transform(x)
        return x

class NoEOSTextStreamer(TextStreamer):
    
    def on_finalized_text(self, text: str, stream_end: bool = False):
        eos_text = self.tokenizer.decode([self.tokenizer.eos_token_id], skip_special_tokens=False)
        text = text.replace(eos_text, "\n")
        print(text, flush=True, end="")




# @title Create datacollator

import torch
import math
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence
import io

# Use local functions (Qwen-compatible) instead of DeepSeek's versions
# from deepseek_ocr.modeling_deepseekocr import (
#     format_messages,
#     text_encode,
#     BasicImageTransform,
#     dynamic_preprocess,
# )


@dataclass
class DeepQwenDataCollator:
    """
    Data collator for DeepQwen model using Qwen2VL tokenizer.
    
    This collator processes images using DeepSeek OCR's dynamic cropping algorithm
    while maintaining compatibility with Qwen2VL's tokenization format.
    
    Key token mappings (Qwen2VL):
        - image_token: <|image_pad|> (id=151655)
        - vision_start: <|vision_start|> (id=151652)
        - vision_end: <|vision_end|> (id=151653)
        - eos_token: <|im_end|> (id=151645)
        - NO bos_token (bos_token_id is None)
    
    Args:
        tokenizer: Qwen2VL Tokenizer
        model: Model
        image_size: Size for image patches (default: 640)
        base_size: Size for global view (default: 1024)
        crop_mode: Whether to use dynamic cropping for large images
        train_on_responses_only: If True, only train on assistant responses (mask user prompts)
    """
    tokenizer: T
    model: Any
    image_size: int = 640
    base_size: int = 1024
    crop_mode: bool = True
    train_on_responses_only: bool = True

    def __init__(
        self,
        tokenizer,
        model,
        image_size: int = 640,
        base_size: int = 1024,
        crop_mode: bool = True,
        train_on_responses_only: bool = True,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.image_size = image_size
        self.base_size = base_size
        self.crop_mode = crop_mode
        self.dtype = model.dtype  # Get dtype from model
        self.train_on_responses_only = train_on_responses_only
        
        # Qwen2VL specific token IDs
        # <|image_pad|> = 151655
        self.image_token_id = getattr(tokenizer, 'image_token_id', None)
        if self.image_token_id is None:
            # Fallback: try to get from added_tokens or use default Qwen2VL value
            self.image_token_id = 151655  # Qwen2VL's <|image_pad|>
        
        self.image_token = tokenizer.decode([self.image_token_id], skip_special_tokens=False)
        
        # Vision wrapper tokens for Qwen2VL format
        self.vision_start_token_id = getattr(tokenizer, 'vision_start_token_id', 151652)
        self.vision_end_token_id = getattr(tokenizer, 'vision_end_token_id', 151653)

        self.image_transform = BasicImageTransform(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            normalize=True
        )
        self.patch_size = 16
        self.downsample_ratio = 4

        # Qwen2VL has NO bos_token (bos_token_id is None)
        # The chat template handles conversation formatting
        self.bos_id = tokenizer.bos_token_id  # Will be None for Qwen2VL
        self.eos_id = tokenizer.eos_token_id  # 151645 for Qwen2VL
        self.pad_token_id = tokenizer.pad_token_id  # 151643 for Qwen2VL

    def deserialize_image(self, image_data) -> Image.Image:
        """Convert image data (bytes dict or PIL Image) to PIL Image in RGB mode"""
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        elif isinstance(image_data, dict) and 'bytes' in image_data:
            image_bytes = image_data['bytes']
            image = Image.open(io.BytesIO(image_bytes))
            return image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image format: {type(image_data)}")

    def calculate_image_token_count(self, image: Image.Image, crop_ratio: Tuple[int, int]) -> int:
        """Calculate the number of tokens this image will generate"""
        num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
        num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

        width_crop_num, height_crop_num = crop_ratio

        if self.crop_mode:
            img_tokens = num_queries_base * num_queries_base + 1
            if width_crop_num > 1 or height_crop_num > 1:
                img_tokens += (num_queries * width_crop_num + 1) * (num_queries * height_crop_num)
        else:
            img_tokens = num_queries * num_queries + 1

        return img_tokens

    def process_image(self, image: Image.Image) -> Tuple[List, List, List, List, Tuple[int, int]]:
        """
        Process a single image based on crop_mode and size thresholds

        Returns:
            Tuple of (images_list, images_crop_list, images_spatial_crop, tokenized_image, crop_ratio)
        """
        images_list = []
        images_crop_list = []
        images_spatial_crop = []

        if self.crop_mode:
            # Determine crop ratio based on image size
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = (1, 1)
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = dynamic_preprocess(
                    image, min_num=2, max_num=9,
                    image_size=self.image_size, use_thumbnail=False
                )

            # Process global view with padding
            global_view = ImageOps.pad(
                image, (self.base_size, self.base_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean)
            )
            images_list.append(self.image_transform(global_view).to(self.dtype))

            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])

            # Process local views (crops) if applicable
            if width_crop_num > 1 or height_crop_num > 1:
                for crop_img in images_crop_raw:
                    images_crop_list.append(
                        self.image_transform(crop_img).to(self.dtype)
                    )

            # Calculate image tokens
            num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

            tokenized_image = ([self.image_token_id] * num_queries_base + [self.image_token_id]) * num_queries_base
            tokenized_image += [self.image_token_id]

            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]) * (
                    num_queries * height_crop_num)

        else:  # crop_mode = False
            crop_ratio = (1, 1)
            images_spatial_crop.append([1, 1])

            # For smaller base sizes, resize; for larger, pad
            if self.base_size <= 640:
                resized_image = image.resize((self.base_size, self.base_size), Image.LANCZOS)
                images_list.append(self.image_transform(resized_image).to(self.dtype))
            else:
                global_view = ImageOps.pad(
                    image, (self.base_size, self.base_size),
                    color=tuple(int(x * 255) for x in self.image_transform.mean)
                )
                images_list.append(self.image_transform(global_view).to(self.dtype))

            num_queries = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            tokenized_image = ([self.image_token_id] * num_queries + [self.image_token_id]) * num_queries
            tokenized_image += [self.image_token_id]

        return images_list, images_crop_list, images_spatial_crop, tokenized_image, crop_ratio

    def process_single_sample(self, messages: List[Dict]) -> Dict[str, Any]:
            """
            Process a single conversation into model inputs.
            
            Expected message format (Qwen2.5-VL native style):
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": <PIL.Image or path or bytes>},
                        {"type": "text", "text": "Describe this image."}
                    ]
                },
                {
                    "role": "assistant", 
                    "content": [{"type": "text", "text": "This is a description..."}]
                }
            ]
            
            Also supports string content for backward compatibility.
            """

            # --- 1. Setup ---
            tokenized_str = []
            images_seq_mask = []
            images_list, images_crop_list, images_spatial_crop = [], [], []

            prompt_token_count = -1  # Index to start training
            assistant_started = False

            # Qwen2VL has NO bos_token, so we don't add one

            for message in messages:
                role = message["role"].lower()  # Normalize role to lowercase
                content = message["content"]

                # Check if this is the assistant's turn
                if role == "assistant":
                    if not assistant_started:
                        # This is the split point. All tokens added *so far*
                        # are part of the prompt.
                        prompt_token_count = len(tokenized_str)
                        assistant_started = True

                # Process content based on format
                if isinstance(content, list):
                    # Qwen2.5-VL native format: content is a list of typed items
                    content_parts = []
                    
                    for item in content:
                        item_type = item.get("type", "")
                        
                        if item_type == "image":
                            # Get image data from various possible keys
                            image_data = item.get("image") or item.get("data")
                            if image_data is not None:
                                pil_image = self.deserialize_image(image_data)
                                
                                # Process the image through DeepSeek's encoder
                                img_list, crop_list, spatial_crop, tok_img, _ = self.process_image(pil_image)
                                
                                images_list.extend(img_list)
                                images_crop_list.extend(crop_list)
                                images_spatial_crop.extend(spatial_crop)
                                
                                # Add image placeholder tokens
                                tokenized_str.extend(tok_img)
                                images_seq_mask.extend([True] * len(tok_img))
                                
                        elif item_type == "text":
                            text = item.get("text", "")
                            
                            # For assistant, append EOS at the end of all text
                            if role == "assistant" and item == content[-1]:
                                if self.tokenizer.eos_token:
                                    text = f"{text.strip()}{self.tokenizer.eos_token}"
                            
                            # Tokenize the text
                            tokenized_text = text_encode(self.tokenizer, text, bos=False, eos=False)
                            tokenized_str.extend(tokenized_text)
                            images_seq_mask.extend([False] * len(tokenized_text))
                            
                else:
                    # Legacy format: content is a string (backward compatibility)
                    text_content = content
                    
                    # For assistant, append EOS token
                    if role == "assistant" and self.tokenizer.eos_token:
                        text_content = f"{text_content.strip()}{self.tokenizer.eos_token}"
                    
                    # Tokenize the text
                    tokenized_text = text_encode(self.tokenizer, text_content, bos=False, eos=False)
                    tokenized_str.extend(tokenized_text)
                    images_seq_mask.extend([False] * len(tokenized_text))

            # --- 2. Validation and Final Prep ---
            # If we never found an assistant message, we're in a weird state
            # (e.g., user-only prompt). We mask everything.
            if not assistant_started:
                print("Warning: No assistant message found in sample. Masking all tokens.")
                prompt_token_count = len(tokenized_str)

            # Prepare image tensors
            images_ori = torch.stack(images_list, dim=0)
            images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype=torch.long)

            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0)
            else:
                images_crop = torch.zeros((1, 3, self.base_size, self.base_size), dtype=self.dtype)

            return {
                "input_ids": torch.tensor(tokenized_str, dtype=torch.long),
                "images_seq_mask": torch.tensor(images_seq_mask, dtype=torch.bool),
                "images_ori": images_ori,
                "images_crop": images_crop,
                "images_spatial_crop": images_spatial_crop_tensor,
                "prompt_token_count": prompt_token_count, # This is now accurate
            }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.
        
        Expected feature format:
        {
            "prompt": str,  # The user's question/instruction
            "response": str,  # The assistant's response
            "image": PIL.Image or bytes dict  # The image
        }
        
        This will be converted to Qwen2.5-VL native conversation format:
        [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": <PIL.Image>},
                    {"type": "text", "text": "<prompt>"}
                ]
            },
            {
                "role": "assistant", 
                "content": [{"type": "text", "text": "<response>"}]
            }
        ]
        """
        batch_data = []

        # Process each sample
        for feature in features:
            try:
                # Use Qwen2.5-VL native message format
                # content is a list of typed items: {"type": "image", ...} or {"type": "text", ...}
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": feature['image']},
                            {"type": "text", "text": feature['prompt']}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": feature["response"]}
                        ]
                    }
                ]
                
                processed = self.process_single_sample(messages)
                batch_data.append(processed)
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

        if not batch_data:
            raise ValueError("No valid samples in batch")

        # Extract lists
        input_ids_list = [item['input_ids'] for item in batch_data]
        images_seq_mask_list = [item['images_seq_mask'] for item in batch_data]
        prompt_token_counts = [item['prompt_token_count'] for item in batch_data]

        # Pad sequences using Qwen2VL's pad_token_id (151643 = <|endoftext|>)
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)
        images_seq_mask = pad_sequence(images_seq_mask_list, batch_first=True, padding_value=False)

        # Create labels
        labels = input_ids.clone()

        # Mask padding tokens
        labels[labels == self.pad_token_id] = -100

        # Mask image tokens (model shouldn't predict these)
        labels[images_seq_mask] = -100

        # Mask user prompt tokens when train_on_responses_only=True (only train on assistant responses)
        if self.train_on_responses_only:
            for idx, prompt_count in enumerate(prompt_token_counts):
                if prompt_count > 0:
                    labels[idx, :prompt_count] = -100

        # Create attention mask
        attention_mask = (input_ids != self.pad_token_id).long()

        images_batch = []
        for item in batch_data:
            images_batch.append((item['images_crop'], item['images_ori']))

        images_spatial_crop = torch.cat([item['images_spatial_crop'] for item in batch_data], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images_batch,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }

