import os

from modules import shared
from modules.paths import models_path

MODEL_SAVE_PATH = shared.cmd_opts.ckpt_dir or os.path.join(
    models_path, "Stable-diffusion")
AUTOPRUNE_PATH = os.path.join(models_path, "Autoprune")
AUTOPRUNE_FAILED_PATH = os.path.join(AUTOPRUNE_PATH, "Failed")
COMPONENT_SAVE_PATH = os.path.join(models_path, "Components")
VAE_SAVE_PATH = shared.cmd_opts.vae_dir or os.path.join(models_path, "VAE")
MODEL_EXT = [".ckpt", ".pt", ".pth", ".safetensors"]

LOAD_PATHS = [
    os.path.join(models_path, "Stable-diffusion"),
    os.path.join(models_path, "Components"),
    os.path.join(models_path, "VAE"),
    os.path.join(models_path, "Lora"),
]

COMPONENT_EXT = {
    "UNET-v1": ".unet.pt",
    "EMA-UNET-v1": ".unet.pt",
    "UNET-v2": ".unet-v2.pt",
    "UNET-v2-Depth": ".unet-v2-d.pt",
    "VAE-v1": ".vae.pt",
    "CLIP-v1": ".clip.pt",
    "CLIP-v2": ".clip-v2.pt",
    "Depth-v2": ".depth.pt"
}

EMA_PREFIX = "model_ema."

METADATA = {'epoch': 0, 'global_step': 0, 'pytorch-lightning_version': '1.6.0'}

IDENTIFICATION = {
    "VAE": {
        "SD-v1": 0,
        "SD-v2": 869,
        "NAI": 2982,
        "WD-VAE-v1": 155,
        "WD-VAE-v2": 41
    },
    "CLIP-v1": {
        "SD-v1": 0,
    },
    "CLIP-v2": {
        "SD-v2": 1141,
        "WD-v1-4": 2543
    }
}

COMPONENTS = {
    "UNET-v1-SD": {
        "keys": {},
        "source": "UNET-v1-SD.txt",
        "prefix": "model.diffusion_model."
    },
    "UNET-v1-EMA": {
        "keys": {},
        "source": "UNET-v1-EMA.txt",
        "prefix": "model_ema.diffusion_model"
    },
    "UNET-v1-Inpainting": {
        "keys": {},
        "source": "UNET-v1-Inpainting.txt",
        "prefix": "model.diffusion_model."
    },
    "UNET-v1-Pix2Pix": {
        "keys": {},
        "source": "UNET-v1-Pix2Pix.txt",
        "prefix": "model.diffusion_model."
    },
    "UNET-v1-Pix2Pix-EMA": {
        "keys": {},
        "source": "UNET-v1-Pix2Pix-EMA.txt",
        "prefix": "model_ema.diffusion_model"
    },
    "UNET-v2-SD": {
        "keys": {},
        "source": "UNET-v2-SD.txt",
        "prefix": "model.diffusion_model."
    },
    "UNET-v2-Inpainting": {
        "keys": {},
        "source": "UNET-v2-Inpainting.txt",
        "prefix": "model.diffusion_model."
    },
    "UNET-v2-Depth": {
        "keys": {},
        "source": "UNET-v2-Depth.txt",
        "prefix": "model.diffusion_model."
    },
    "VAE-v1-SD": {
        "keys": {},
        "source": "VAE-v1-SD.txt",
        "prefix": "first_stage_model."
    },
    "CLIP-v1-SD": {
        "keys": {},
        "source": "CLIP-v1-SD.txt",
        "prefix": "cond_stage_model.transformer.text_model."
    },
    "CLIP-v1-NAI": {
        "keys": {},
        "source": "CLIP-v1-SD.txt",
        "prefix": "cond_stage_model.transformer."
    },
    "CLIP-v2-SD": {
        "keys": {},
        "source": "CLIP-v2-SD.txt",
        "prefix": "cond_stage_model.model."
    },
    "CLIP-v2-WD": {
        "keys": {},
        "source": "CLIP-v2-WD.txt",
        "prefix": "cond_stage_model.model."
    },
    "Depth-v2-SD": {
        "keys": {},
        "source": "Depth-v2-SD.txt",
        "prefix": "depth_model.model."
    },
    "LoRA-v1-CLIP": {
        "keys": {},
        "shapes": {},
        "source": "LoRA-v1-CLIP.txt",
        "prefix": ""
    },
    "LoRA-v1A-CLIP": {
        "keys": {},
        "shapes": {},
        "source": "LoRA-v1A-CLIP.txt",
        "prefix": ""
    },
    "LoRA-v1-UNET": {
        "keys": {},
        "shapes": {},
        "source": "LoRA-v1-UNET.txt",
        "prefix": ""
    },
    "LoRA-v1A-UNET": {
        "keys": {},
        "shapes": {},
        "source": "LoRA-v1A-UNET.txt",
        "prefix": ""
    },
    "ControlNet-v1-SD": {
        "keys": {},
        "shapes": {},
        "source": "ControlNet-v1-SD.txt",
        "prefix": "control_model."
    },
}

COMPONENT_CLASS = {
    "UNET-v1-SD": "UNET-v1",
    "UNET-v1-EMA": "EMA-UNET-v1",
    "UNET-v1-Inpainting": "UNET-v1",
    "UNET-v1-Pix2Pix": "UNET-v1-Pix2Pix",
    "UNET-v1-Pix2Pix-EMA": "EMA-UNET-v1-Pix2Pix",
    "UNET-v2-SD": "UNET-v2",
    "UNET-v2-Inpainting": "UNET-v2",
    "UNET-v2-Depth": "UNET-v2-Depth",
    "VAE-v1-SD": "VAE-v1",
    "CLIP-v1-SD": "CLIP-v1",
    "CLIP-v1-NAI": "CLIP-v1",
    "CLIP-v2-SD": "CLIP-v2",
    "CLIP-v2-WD": "CLIP-v2",
    "Depth-v2-SD": "Depth-v2",
    "LoRA-v1-UNET": "LoRA-v1-UNET",
    "LoRA-v1-CLIP": "LoRA-v1-CLIP",
    "LoRA-v1A-UNET": "LoRA-v1-UNET",
    "LoRA-v1A-CLIP": "LoRA-v1-CLIP",
    "ControlNet-v1-SD": "ControlNet-v1",
}

OPTIONAL = [
    ("alphas_cumprod", (1000,)),
    ("alphas_cumprod_prev", (1000,)),
    ("betas", (1000,)),
    ("log_one_minus_alphas_cumprod", (1000,)),
    ("model_ema.decay", ()),
    ("model_ema.num_updates", ()),
    ("posterior_log_variance_clipped", (1000,)),
    ("posterior_mean_coef1", (1000,)),
    ("posterior_mean_coef2", (1000,)),
    ("posterior_variance", (1000,)),
    ("sqrt_alphas_cumprod", (1000,)),
    ("sqrt_one_minus_alphas_cumprod", (1000,)),
    ("sqrt_recip_alphas_cumprod", (1000,)),
    ("sqrt_recipm1_alphas_cumprod", (1000,)),
    ("logvar", (1000,)),
]

ARCHITECTURES = {
    "UNET-v1": {
        "classes": ["UNET-v1"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "UNET-v1-Pix2Pix": {
        "classes": ["UNET-v1-Pix2Pix"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "UNET-v2": {
        "classes": ["UNET-v2"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "UNET-v2-Depth": {
        "classes": ["UNET-v2-Depth"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "VAE-v1": {
        "classes": ["VAE-v1"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "CLIP-v1": {
        "classes": ["CLIP-v1"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "CLIP-v2": {
        "classes": ["CLIP-v2"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "Depth-v2": {
        "classes": ["Depth-v2"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "ControlNet-v1": {
        "classes": ["ControlNet-v1"],
        "optional": [],
        "required": [],
        "prefixed": False
    },
    "SD-v1": {
        "classes": ["UNET-v1", "VAE-v1", "CLIP-v1"],
        "optional": OPTIONAL,
        "required": [],
        "prefixed": True
    },
    "SD-v1-Pix2Pix": {
        "classes": ["UNET-v1-Pix2Pix", "VAE-v1", "CLIP-v1"],
        "optional": OPTIONAL,
        "required": [],
        "prefixed": True
    },
    "SD-v1-ControlNet": {
        "classes": ["UNET-v1", "VAE-v1", "CLIP-v1", "ControlNet-v1"],
        "optional": OPTIONAL,
        "required": [],
        "prefixed": True
    },
    "SD-v2": {
        "classes": ["UNET-v2", "VAE-v1", "CLIP-v2"],
        "optional": OPTIONAL,
        "required": [],
        "prefixed": True
    },
    "SD-v2-Depth": {
        "classes": ["UNET-v2-Depth", "VAE-v1", "CLIP-v2", "Depth-v2"],
        "optional": OPTIONAL,
        "required": [],
        "prefixed": True
    },
    "EMA-v1": {
        "classes": ["EMA-UNET-v1"],
        "optional": OPTIONAL,
        "required": [],
        "prefixed": True
    },
    "EMA-v1-Pix2Pix": {
        "classes": ["EMA-UNET-v1-Pix2Pix"],
        "optional": OPTIONAL,
        "required": [],
        "prefixed": True
    },
    # standalone component architectures, for detecting broken models
    "UNET-v1-BROKEN": {
        "classes": ["UNET-v1"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "UNET-v1-Pix2Pix-BROKEN": {
        "classes": ["UNET-v1-Pix2Pix"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "UNET-v2-BROKEN": {
        "classes": ["UNET-v2"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "UNET-v2-Depth-BROKEN": {
        "classes": ["UNET-v2-Depth"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "VAE-v1-BROKEN": {
        "classes": ["VAE-v1"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "CLIP-v1-BROKEN": {
        "classes": ["CLIP-v1"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "CLIP-v2-BROKEN": {
        "classes": ["CLIP-v2"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "Depth-v2-BROKEN": {
        "classes": ["Depth-v2"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "ControlNet-v1-BROKEN": {
        "classes": ["ControlNet-v1"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "LoRA-v1-UNET": {
        "classes": ["LoRA-v1-UNET"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "LoRA-v1-CLIP": {
        "classes": ["LoRA-v1-CLIP"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
    "LoRA-v1": {
        "classes": ["LoRA-v1-CLIP", "LoRA-v1-UNET"],
        "optional": [],
        "required": [],
        "prefixed": True
    },
}
