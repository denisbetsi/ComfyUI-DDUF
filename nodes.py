import copy
import os
import torch
from safetensors.torch import load_file
from .utils import SCHEDULERS, token_auto_concat_embeds, convert_images_to_tensors
from comfy.model_management import get_torch_device
import folder_paths
from diffusers import StableDiffusionPipeline, AutoencoderKL, AutoencoderTiny
from diffusers import DiffusionPipeline


class DDUFLoader:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.float32
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ), }}

    RETURN_TYPES = ("PIPELINE", "AUTOENCODER", "SCHEDULER",)
    FUNCTION = "create_pipeline"
    CATEGORY = "Diffusers"

    def create_pipeline(self, ckpt_name):
        dduf_path = folder_paths.get_folder_paths("checkpoints")

        pipe = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=dduf_path[0],
            dduf_file=ckpt_name,
            torch_dtype=self.dtype,
            cache_dir=self.tmp_dir,
        )
        
        return ((pipe, self.tmp_dir), pipe.vae, pipe.scheduler)

class DiffusersPipelineLoader:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.float32
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ), }}

    RETURN_TYPES = ("PIPELINE", "AUTOENCODER", "SCHEDULER",)

    FUNCTION = "create_pipeline"

    CATEGORY = "Diffusers"

    def create_pipeline(self, ckpt_name):
        ckpt_cache_path = os.path.join(self.tmp_dir, ckpt_name)
        
        StableDiffusionPipeline.from_single_file(
            pretrained_model_link_or_path=folder_paths.get_full_path("checkpoints", ckpt_name),
            torch_dtype=self.dtype,
            cache_dir=self.tmp_dir,
        ).save_pretrained(ckpt_cache_path, safe_serialization=True)
        
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=ckpt_cache_path,
            torch_dtype=self.dtype,
            cache_dir=self.tmp_dir,
        )
        return ((pipe, ckpt_cache_path), pipe.vae, pipe.scheduler)

class DiffusersSchedulerLoader:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.float32
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE", ),
                "scheduler_name": (list(SCHEDULERS.keys()), ), 
            }
        }

    RETURN_TYPES = ("SCHEDULER",)

    FUNCTION = "load_scheduler"

    CATEGORY = "Diffusers"

    def load_scheduler(self, pipeline, scheduler_name):
        print(f"Pipeline type: {type(pipeline)}")
        print(f"Pipeline attributes: {dir(pipeline)}")
        
        if isinstance(pipeline, tuple):
            print("Pipeline is a tuple:", pipeline)
            pipeline_obj = pipeline[0]
        else:
            print("Pipeline is an object")
            pipeline_obj = pipeline

        # Try to get existing scheduler config
        try:
            if hasattr(pipeline_obj, 'scheduler') and pipeline_obj.scheduler is not None:
                print("Using existing scheduler config")
                existing_scheduler = pipeline_obj.scheduler
                config_dict = existing_scheduler.config
                
                # Create new scheduler with existing config
                scheduler = SCHEDULERS[scheduler_name].from_config(config_dict)
                
                # Set dtype if the scheduler supports it
                if hasattr(scheduler, 'dtype'):
                    scheduler.dtype = self.dtype
                
                return (scheduler,)
                
        except Exception as e:
            print(f"Failed to use existing scheduler config: {str(e)}")

        # If that fails, try creating a default scheduler
        try:
            print("Attempting to create default scheduler")
            scheduler = SCHEDULERS[scheduler_name]()
            
            # Set dtype if the scheduler supports it
            if hasattr(scheduler, 'dtype'):
                scheduler.dtype = self.dtype
                
            return (scheduler,)
            
        except Exception as e:
            print(f"Failed to create default scheduler: {str(e)}")
            raise

class DiffusersModelMakeup:
    def __init__(self):
        self.torch_device = get_torch_device()
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE", ), 
                "scheduler": ("SCHEDULER", ),
                "autoencoder": ("AUTOENCODER", ),
            }, 
        }

    RETURN_TYPES = ("MAKED_PIPELINE",)

    FUNCTION = "makeup_pipeline"

    CATEGORY = "Diffusers"

    def makeup_pipeline(self, pipeline, scheduler, autoencoder):
        # Unpack pipeline tuple if it's a tuple, otherwise use the pipeline object directly
        pipeline = pipeline[0] if isinstance(pipeline, tuple) else pipeline
        pipeline.vae = autoencoder
        pipeline.scheduler = scheduler
        pipeline = pipeline.to(self.torch_device)
        return (pipeline,)

class DiffusersSimpleSampler:
    def __init__(self):
        self.torch_device = get_torch_device()
        
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "maked_pipeline": ("MAKED_PIPELINE", ),
            "prompt": ("STRING", {"multiline": True}),
            "width": ("INT", {"default": 1360, "min": 1, "max": 8192, "step": 1}),
            "height": ("INT", {"default": 768, "min": 1, "max": 8192, "step": 1}),
            "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "max_sequence_length": ("INT", {"default": 256, "min": 1, "max": 1024}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "Diffusers"

    def sample(self, maked_pipeline, prompt, height, width, steps, cfg, seed, max_sequence_length):
        images = maked_pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=torch.Generator(self.torch_device).manual_seed(seed),
            max_sequence_length=max_sequence_length,
        ).images
        
        return (convert_images_to_tensors(images),)

NODE_CLASS_MAPPINGS = {
    "DiffusersPipelineLoader": DiffusersPipelineLoader,
    "DiffusersSchedulerLoader": DiffusersSchedulerLoader,
    "DiffusersModelMakeup": DiffusersModelMakeup,
    "DiffusersSimpleSampler": DiffusersSimpleSampler,
    "DDUFLoader": DDUFLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusersPipelineLoader": "Diffusers Pipeline Loader",
    "DiffusersSchedulerLoader": "Diffusers Scheduler Loader",
    "DiffusersModelMakeup": "Diffusers Model Makeup",
    "DiffusersSimpleSampler": "Diffusers Simple Sampler",
    "DDUFLoader": "DDUF Loader",
}
