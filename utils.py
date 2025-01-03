import io
import torch
import requests
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torchvision.transforms import ToTensor
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    UniPCMultistepScheduler,
)
from comfy.model_management import get_torch_device

SCHEDULERS = {
    'DDIM' : DDIMScheduler,
    'DDPM' : DDPMScheduler,
    'DEISMultistep' : DEISMultistepScheduler,
    'DPMSolverMultistep' : DPMSolverMultistepScheduler,
    'DPMSolverSinglestep' : DPMSolverSinglestepScheduler,
    'EulerAncestralDiscrete' : EulerAncestralDiscreteScheduler,
    'EulerDiscrete' : EulerDiscreteScheduler,
    'HeunDiscrete' : HeunDiscreteScheduler,
    'KDPM2AncestralDiscrete' : KDPM2AncestralDiscreteScheduler,
    'KDPM2Discrete' : KDPM2DiscreteScheduler,
    'UniPCMultistep' : UniPCMultistepScheduler
}

def token_auto_concat_embeds(pipe, positive, negative):
    device = get_torch_device()
    max_length = pipe.tokenizer.model_max_length
    positive_length = pipe.tokenizer(positive, return_tensors="pt").input_ids.shape[-1]
    negative_length = pipe.tokenizer(negative, return_tensors="pt").input_ids.shape[-1]
    
    print(f'Token length is model maximum: {max_length}, positive length: {positive_length}, negative length: {negative_length}.')
    if max_length < positive_length or max_length < negative_length:
        print('Concatenated embedding.')
        if positive_length > negative_length:
            positive_ids = pipe.tokenizer(positive, return_tensors="pt").input_ids.to(device)
            negative_ids = pipe.tokenizer(negative, truncation=False, padding="max_length", max_length=positive_ids.shape[-1], return_tensors="pt").input_ids.to(device)
        else:
            negative_ids = pipe.tokenizer(negative, return_tensors="pt").input_ids.to(device)  
            positive_ids = pipe.tokenizer(positive, truncation=False, padding="max_length", max_length=negative_ids.shape[-1],  return_tensors="pt").input_ids.to(device)
    else:
        positive_ids = pipe.tokenizer(positive, truncation=False, padding="max_length", max_length=max_length,  return_tensors="pt").input_ids.to(device)
        negative_ids = pipe.tokenizer(negative, truncation=False, padding="max_length", max_length=max_length, return_tensors="pt").input_ids.to(device)
    
    positive_concat_embeds = []
    negative_concat_embeds = []
    positive_pooled_embeds = []
    negative_pooled_embeds = []
    
    for i in range(0, positive_ids.shape[-1], max_length):
        # Get both text embeddings and pooled embeddings
        text_embeds, pooled_embeds = pipe.text_encoder(positive_ids[:, i: i + max_length], return_dict=False)
        positive_concat_embeds.append(text_embeds)
        positive_pooled_embeds.append(pooled_embeds)
        
        text_embeds, pooled_embeds = pipe.text_encoder(negative_ids[:, i: i + max_length], return_dict=False)
        negative_concat_embeds.append(text_embeds)
        negative_pooled_embeds.append(pooled_embeds)
    
    positive_prompt_embeds = torch.cat(positive_concat_embeds, dim=1)
    negative_prompt_embeds = torch.cat(negative_concat_embeds, dim=1)
    
    # For pooled embeddings, we take the mean of all segments
    positive_pooled_prompt_embeds = torch.stack(positive_pooled_embeds).mean(dim=0)
    negative_pooled_prompt_embeds = torch.stack(negative_pooled_embeds).mean(dim=0)
    
    return (
        positive_prompt_embeds, 
        negative_prompt_embeds,
        positive_pooled_prompt_embeds,
        negative_pooled_prompt_embeds
    )


def convert_images_to_tensors(images: list[Image.Image]):
    return torch.stack([np.transpose(ToTensor()(image), (1, 2, 0)) for image in images])

def convert_tensors_to_images(images: torch.tensor):
    return [Image.fromarray(np.clip(255. * image.to("cpu").numpy(), 0, 255).astype(np.uint8)) for image in images]

def resize_images(images: list[Image.Image], size: tuple[int, int]):
    return [image.resize(size) for image in images]
