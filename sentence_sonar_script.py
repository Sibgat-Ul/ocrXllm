# %%
# !pip install transformers
# !pip install bitsandbytes -q

import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler

from datasets import load_dataset
from transformers import M2M100ForConditionalGeneration, AutoTokenizer, SiglipImageProcessor
from transformers.modeling_outputs import BaseModelOutput
from huggingface_hub import snapshot_download, login

from PIL import Image
from tqdm import tqdm


# model_name = "mtmlt/sonar-nllb-200-1.3B"
# model_dir = snapshot_download(model_name)

# model = M2M100ForConditionalGeneration.from_pretrained(model_dir)
# tokenizer = AutoTokenizer.from_pretrained(model_dir)

# def encode_mean_pool(texts, tokenizer, encoder, lang='eng_Latn', norm=False):
#     tokenizer.src_lang = lang
#     with torch.inference_mode():
#         batch = tokenizer(texts, return_tensors='pt', padding=True)
#         seq_embs = encoder(**batch)
#         mask = batch.attention_mask
#         mean_emb = (seq_embs.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.unsqueeze(-1).sum(1)
#         if norm:
#             mean_emb = torch.nn.functional.normalize(mean_emb)
#     return mean_emb, seq_embs, mask

# sentences = ['My name is SONAR.', 'I can embed the sentences into vector space.']

# %%
model_name = "mtmlt/sonar-nllb-200-1.3B"
model_dir = snapshot_download(model_name)
img_model_dir = snapshot_download("google/siglip-base-patch16-384")

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

decoder_model = M2M100ForConditionalGeneration.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.tgt_lang = 'eng_Latn'
tokenizer.src_lang = 'eng_Latn'
decoder_model.eval()  

decoder_encoder = decoder_model.model.encoder  
decoder_model.eval()

# %%
import torch.nn as nn

import torch
import torch.nn as nn
from transformers import SiglipVisionModel, get_cosine_schedule_with_warmup

class SonarImageEnc(nn.Module):
    def __init__(self, path="google/siglip-base-patch16-384", sonar_dim=1024):
        super().__init__()
        self.vision_encoder = SiglipVisionModel.from_pretrained(path, torch_dtype="auto")
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        vdim = self.vision_encoder.config.hidden_size

        self.proj = nn.Sequential(
            nn.Linear(vdim, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, sonar_dim)
            
        )
        self.norm = nn.LayerNorm(sonar_dim)

    def forward(self, pixel_values):
        h = self.vision_encoder(pixel_values).last_hidden_state
        h = self.proj(h)
        h = h.mean(dim=1) 
        return self.norm(h) 

image_encoder = SonarImageEnc(path=img_model_dir).to(device)

# %%
def text_to_sonar_embedding(texts=None, tokenizer=None, encoder_model=None, batch=None, lang='eng_Latn', device='cpu', norm=False):
    if batch is None:
        tokenizer.src_lang = lang
        batch = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    encoder_model.to(device)
    with torch.no_grad():
        seq_embs = encoder_model(**batch)
    mask = batch['attention_mask']
    mean_emb = (seq_embs.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    if norm:
        mean_emb = F.normalize(mean_emb, dim=-1)
    return mean_emb, seq_embs, mask, batch

for param in decoder_model.parameters():
    param.requires_grad = False

# %%
from huggingface_hub import snapshot_download

# dset = snapshot_download("pixparse/cc12m-wds", allow_patterns=["cc12m-train-0000.tar"], repo_type="dataset")

# dset = snapshot_download("pixparse/cc12m-wds", allow_patterns=["cc12m-train-000[0-9].tar"], repo_type="dataset")
# dset = snapshot_download("pixparse/cc12m-wds", allow_patterns=["cc12m-train-001[0-9].tar"], repo_type="dataset")
# dset = snapshot_download("pixparse/cc12m-wds", allow_patterns=["cc12m-train-002[0-9].tar"], repo_type="dataset")
# dset = snapshot_download("pixparse/cc12m-wds", allow_patterns=["cc12m-train-003[0-9].tar"], repo_type="dataset")
# dset = snapshot_download("pixparse/cc12m-wds", allow_patterns=["cc12m-train-004[0-9].tar"], repo_type="dataset")
# dset = snapshot_download("pixparse/cc12m-wds", allow_patterns=["cc12m-train-005[0-9].tar"], repo_type="dataset")

dset = snapshot_download("romrawinjp/multilingual-coco", repo_type="dataset")

# %%
dataset_train = load_dataset(dset, streaming=True)

# %%
image_processor = SiglipImageProcessor.from_pretrained(img_model_dir)  

def normalize_caption(cap_item):
    """
    Extract first valid string from nested tuple/list structure.
    Returns the string or None if invalid.
    """
    if isinstance(cap_item, (tuple, list)):
        for elem in cap_item:
            if isinstance(elem, str) and elem.strip():
                return elem
        return None
    elif isinstance(cap_item, str) and cap_item.strip():
        return cap_item
    return None

def process_img(batch):    
    lang_to_sonar = {
        "en": "eng_Latn",
        "ar": "arb_Arab",
        "de": "dan_Latn",
        "ru": "rus_Cyrl"
    }
    
    rgb_images = [img.convert("RGB") for img in batch["image"]]
    processed_img = image_processor(images=rgb_images, return_tensors="pt")
    
    return {
        "pixel_values": processed_img["pixel_values"],
        "eng_Latn": [normalize_caption(cap) for cap in batch["en"]],
        "arb_Arab": [normalize_caption(cap) for cap in batch["ar"]],
        "dan_Latn": [normalize_caption(cap) for cap in batch["de"]],
        "rus_Cyrl": [normalize_caption(cap) for cap in batch["ru"]]
    }
    
def process_img_val(batch):    
    rgb_images = [img.convert("RGB") for img in batch["jpg"]]
    processed_img = image_processor(images=rgb_images, return_tensors="pt")
    
    return {
        "pixel_values": processed_img["pixel_values"],
        "caption": batch["txt"]
    }

import os as _os
_hf_token = _os.getenv("HF_TOKEN")
if _hf_token:
    login(_hf_token)
else:
    print("HF_TOKEN not set; proceeding without login.")

dataset_train = load_dataset(dset, streaming=True)
dataset_train = dataset_train.map(process_img, batched=True, batch_size=32, remove_columns=list(next(iter(dataset_train.values())).features)).with_format("torch")

# %%
val_hf_dataset = load_dataset("pixparse/cc12m-wds", data_files="cc12m-train-0999.tar")
val_hf_dataset = val_hf_dataset["train"]
val_hf_dataset = val_hf_dataset.select(range(0, 1000))
val_hf_dataset = val_hf_dataset.map(process_img_val, batched=True, batch_size=128, remove_columns=val_hf_dataset.column_names, keep_in_memory=True)

# %%
train_hf_dataset = dataset_train["train"]
batch_size = 3
train_loader = DataLoader(train_hf_dataset, batch_size=batch_size)

val_hf_dataset.set_format("torch")
val_loader = DataLoader(val_hf_dataset, batch_size=batch_size, shuffle=False)

# %%
num_epochs = 5
save_dir = "./adapter_checkpoints"
os.makedirs(save_dir, exist_ok=True)

# %%
import gc

gc.collect()

# %%
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import os
import random
import bitsandbytes as bnb
from torch.amp import GradScaler
from transformers import get_cosine_schedule_with_warmup

def validate(
    image_encoder,
    decoder_model,
    val_loader,
    tokenizer,
    device,
    max_batches=None,
    max_new_tokens=48,
    num_beams=5
):
    image_encoder.eval()
    decoder_model.eval()
    
    total_ce_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches and i >= max_batches:
                break

            pixel_values = batch['pixel_values'].to(device)
            captions = batch['caption']  

            tokenizer.src_lang = 'eng_Latn'
            tokenized_en = tokenizer(
                captions, return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(device)
            
            labels = tokenized_en['input_ids'].clone()
            labels[labels == tokenizer.pad_token_id] = -100

            img_vec = image_encoder(pixel_values)         
            # img_vec = F.normalize(img_vec, dim=-1)        
            enc_out = BaseModelOutput(last_hidden_state=img_vec.unsqueeze(1)) 

            enc_mask = torch.ones(img_vec.size(0), 1, dtype=torch.long, device=device)

            outputs_en = decoder_model(
                encoder_outputs=enc_out,
                attention_mask=enc_mask,
                labels=labels,
                return_dict=True
            )
            ce_loss = outputs_en.loss
            total_ce_loss += ce_loss.item()
            
            generated_en = decoder_model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=img_vec.unsqueeze(1)),
                forced_bos_token_id=tokenizer.convert_tokens_to_ids('eng_Latn'),
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            preds_en = tokenizer.batch_decode(generated_en, skip_special_tokens=True)

            tokenizer.src_lang = 'ben_Beng'
            generated_bn = decoder_model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=img_vec.unsqueeze(1)),
                forced_bos_token_id=tokenizer.convert_tokens_to_ids('ben_Beng'),
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            preds_bn = tokenizer.batch_decode(generated_bn, skip_special_tokens=True)
            tokenizer.src_lang = 'eng_Latn'  

            if i % 50 == 0:
                print(f"\n{'='*70}")
                print(f"Batch {i} | Zero-Shot Bengali Inspection")
                print(f"{'='*70}")
                print(f"True EN:       {captions[:2]}")
                print(f"Generated EN:  {preds_en[:2]}")
                print(f"Generated BEN: {preds_bn[:2]}")
                print(f"CE Loss: {ce_loss.item():.3f} | IMG vec std: {img_vec.std():.4f}")

            num_batches += 1

    avg_ce = total_ce_loss / num_batches if num_batches > 0 else 0

    print(f"\n{'='*70}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total batches: {num_batches}")
    print(f"Avg CE Loss: {avg_ce:.4f} (lower is better)")
    print(f"{'='*70}\n")

    image_encoder.train()
    decoder_model.train()
    
    return avg_ce

# %%
def generate_image_embed(
        image_encoder, 
        decoder_model, 
        pixel_values, 
        labels, 
        normalize=False, 
        device="cuda"
    ):

    img_vec = image_encoder(pixel_values) 

    if normalize:
        img_vec = F.normalize(img_vec, dim=-1)

    enc_out = BaseModelOutput(last_hidden_state=img_vec.unsqueeze(1))  

    enc_mask = torch.ones(img_vec.size(0), 1, dtype=torch.long, device=device)

    outputs = decoder_model(
        encoder_outputs=enc_out,
        attention_mask=enc_mask,         
        labels=labels,
        return_dict=True
    )

    return outputs, img_vec

def generate_text_embed(
    tokenized,
    encoder_model=None,
    decoder_model=None,
):
    seq_embs = encoder_model(**tokenized)
    mask = tokenized['attention_mask']
    text_vec = (seq_embs.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    
    text_out = BaseModelOutput(last_hidden_state=text_vec.unsqueeze(1))
    text_mask = torch.ones(text_vec.size(0), 1, dtype=torch.long, device=text_vec.device)
    outputs = decoder_model(
        encoder_outputs=text_out,
        attention_mask=text_mask,         
        labels=tokenized['input_ids'],
        return_dict=True
    )

    return outputs, text_vec

def train_multilingual(
    image_encoder,
    decoder_model,
    tokenizer,
    train_loader,
    val_loader,
    device='cuda',
    max_steps=12000,
    grad_accum=48,
    lr=2e-4,
    eval_every=1000,
    save_dir='./checkpoints',
    log_every=500,
    max_new_tokens=48,
    num_beams=5
):
    image_encoder.train().to(device)
    decoder_model.eval().to(device)
    text_enc = decoder_model.model.encoder
    for p in decoder_model.parameters():
        p.requires_grad = False
    for p in text_enc.parameters():
        p.requires_grad = False

    langs = ["eng_Latn"]

    optimizer = bnb.optim.AdamW8bit(
        filter(lambda p: p.requires_grad, image_encoder.parameters()),
        lr=lr, weight_decay=0.01
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=max_steps
    )

    global_step, best_val = 0, float('inf')
    os.makedirs(save_dir, exist_ok=True)

    while global_step < max_steps:
        epoch = (global_step // 82800) + 1
        pbar = tqdm(train_loader, desc=f"Ep {epoch}")

        for micro_step, batch in enumerate(pbar):
            pixel_values = batch['pixel_values'].to(device, non_blocking=True)
            captions, lang = None, None
            for l in langs:
                if batch.get(l):
                    captions, lang = batch[l], l
                    break
            if captions is None:
                continue

            tokenizer.src_lang = lang
            tok = tokenizer(captions, return_tensors='pt', padding=True,
                            truncation=True, max_length=128).to(device)
            labels = tok.input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            outputs_text, text_vec = generate_text_embed(
                tokenized=tok,
                encoder_model=text_enc,
                decoder_model=decoder_model,
            )
            
            outputs, img_vec = generate_image_embed(
                image_encoder,
                decoder_model,
                pixel_values,
                labels,
                normalize=False,
                device=device
            )

            loss_mse = F.mse_loss(img_vec, text_vec)

            loss = (outputs.loss + loss_mse + 0.5*outputs_text.loss) / grad_accum

            loss.backward()
            if (micro_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(image_encoder.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                pbar.set_postfix({'loss': loss.item() * grad_accum,
                                  'lr': f"{scheduler.get_last_lr()[0]:.2e}"})

                # Log training captions every log_every steps
                if global_step % log_every == 0:
                    image_encoder.eval()
                    with torch.no_grad():
                        img_vec_eval = image_encoder(pixel_values)
                        enc_out = BaseModelOutput(last_hidden_state=img_vec_eval.unsqueeze(1))
                        enc_mask = torch.ones(img_vec_eval.size(0), 1, dtype=torch.long, device=device)
                        
                        # Generate English captions
                        generated_en = decoder_model.generate(
                            encoder_outputs=enc_out,
                            attention_mask=enc_mask,
                            forced_bos_token_id=tokenizer.convert_tokens_to_ids('eng_Latn'),
                            max_new_tokens=max_new_tokens,
                            num_beams=num_beams,
                        )
                        preds_en = tokenizer.batch_decode(generated_en, skip_special_tokens=True)
                        
                        # Generate Bengali captions (zero-shot)
                        generated_bn = decoder_model.generate(
                            encoder_outputs=enc_out,
                            attention_mask=enc_mask,
                            forced_bos_token_id=tokenizer.convert_tokens_to_ids('ben_Beng'),
                            max_new_tokens=max_new_tokens,
                            num_beams=num_beams,
                        )
                        preds_bn = tokenizer.batch_decode(generated_bn, skip_special_tokens=True)
                    
                    print(f"\n{'='*70}")
                    print(f"Step {global_step} | Training Caption Inspection")
                    print(f"{'='*70}")
                    print(f"True EN:       {captions[:2]}")
                    print(f"Generated EN:  {preds_en[:2]}")
                    print(f"Generated BEN: {preds_bn[:2]}")
                    print(f"Loss: {loss.item() * grad_accum:.3f} | IMG vec std: {img_vec_eval.std():.4f}")
                    print(f"{'='*70}\n")
                    
                    image_encoder.train()

                if global_step % eval_every == 0:
                    val_loss = validate(image_encoder, decoder_model, val_loader,
                                        tokenizer, device, max_batches=30)
                    print(f"\nStep {global_step}  val-loss {val_loss:.4f}")
                    if val_loss < best_val:
                        best_val = val_loss
                        torch.save({'step': global_step,
                                    'encoder': image_encoder.state_dict(),
                                    'opt': optimizer.state_dict()},
                                   f"{save_dir}/best.pt")
                if global_step >= max_steps:
                    break
        if global_step >= max_steps:
            break

    print(f"\nDone  best val-loss {best_val:.4f}")
    return global_step, best_val

# %%
global_step, best_loss = train_multilingual(
    image_encoder=image_encoder,
    decoder_model=decoder_model,
    tokenizer=tokenizer,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    max_steps=12000,
    grad_accum=10,
    lr=5e-5,
    eval_every=1000,
    save_dir=save_dir
)

# %%



