import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import FluxTransformer2DModel
from utils.utils import get_bbox_from_mask

# Import the Kontext pipeline (adjust path as needed)
sys.path.append('/workspace/libcom')
from diffusers import FluxKontextPipeline


dtype = torch.bfloat16

# Configuration
target_id = "ana_2"
source_id = "taylor_2"
data_dir = "/workspace/outputs"
view_aligned_dir = os.path.join(data_dir, "view_aligned/shape_hi3dgen__texture_mvadapter", f"{target_id}_to_{source_id}")

###   example  #####
ref_dir = os.path.join(view_aligned_dir, "warping/target_image.png")
ref_mask_dir = os.path.join(view_aligned_dir, "warping/target_hair_mask.png")
image_dir = os.path.join(data_dir, "bald/w_seg/image", f"{source_id}.png")
image_mask_dir = ref_mask_dir

ref_list = [ref_dir]
ref_list.sort()

ref_mask_list = [ref_mask_dir]
ref_mask_list.sort()

image_list = [image_dir]
image_list.sort()

image_mask_list = [image_mask_dir]
image_mask_list.sort()
###   example  #####


class FluxKontextBlender:
    """
    Flux Kontext based hair blending adapted for HairPort pipeline.
    """
    
    def __init__(self, device=0, lora_path=None):
        """
        Initialize Flux Kontext blending model.
        
        Args:
            device: GPU device ID
            lora_path: Path to blending LoRA weights
        """
        self.device = f"cuda:{device}" if isinstance(device, int) else device
        weight_path = "black-forest-labs/FLUX.1-Kontext-dev"
        
        # Use provided lora_path or default
        if lora_path is None:
            lora_path = "/workspace/HairPort/Hairdar/flux_kontext_blending.safetensors"
            lora_path_harmonization = "/workspace/HairPort/Hairdar/flux_kontext_harmonization.safetensors"
        self.build_model(weight_path, lora_path, lora_path_harmonization)
    
    def build_model(self, weight_path, lora_path, lora_path_harmonization):
        """Build the Flux Kontext pipeline."""
        transformer = FluxTransformer2DModel.from_pretrained(
            weight_path, 
            subfolder="transformer"
        )
        transformer.requires_grad_(False)
        
        self.pipeline = FluxKontextPipeline.from_pretrained(
            weight_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16
        )
        
        if os.path.exists(lora_path):
            self.pipeline.load_lora_weights(lora_path)
            self.pipeline.load_lora_weights(lora_path_harmonization)
        
        self.pipeline = self.pipeline.to(self.device, dtype=torch.bfloat16)
    
    def generate_composite_from_mask(self, base_image, reference_image, reference_mask, target_size=(1024, 1024)):
        """
        Generate initial composite by pasting reference hair onto base image using mask.
        
        Args:
            base_image: PIL Image - target/base image
            reference_image: PIL Image - reference image with hair
            reference_mask: PIL Image - binary mask of hair region
            target_size: Tuple - output image size
            
        Returns:
            PIL Image - initial composite image
        """
        # Resize all inputs to target size
        base_img = base_image.convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
        ref_img = reference_image.convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
        ref_mask = reference_mask.convert('L').resize(target_size, Image.Resampling.NEAREST)
        
        # Convert to numpy
        base_np = np.array(base_img)
        ref_np = np.array(ref_img)
        mask_np = np.array(ref_mask)
        
        # Create binary mask (>128 is foreground)
        mask_binary = (mask_np > 128).astype(np.uint8)
        mask_3ch = np.stack([mask_binary, mask_binary, mask_binary], axis=-1)
        
        # Composite: where mask is 1, use reference; otherwise use base
        composite_np = np.where(mask_3ch, ref_np, base_np)
        
        return Image.fromarray(composite_np)
    
    @torch.no_grad()
    def blend(self, base_image_path, reference_image_path, reference_mask_path, 
              output_path, prompt='natural hair blending', 
              num_samples=1, sample_steps=28, guidance_scale=2.5, 
              seed=321, save_intermediates=True):
        """
        Blend reference hair onto base image using Flux Kontext.
        
        Args:
            base_image_path: Path to target/base image
            reference_image_path: Path to reference image with hair
            reference_mask_path: Path to hair mask
            output_path: Path to save output
            prompt: Text prompt for generation
            num_samples: Number of samples to generate
            sample_steps: Number of denoising steps
            guidance_scale: CFG scale
            seed: Random seed
            save_intermediates: Whether to save intermediate results
            
        Returns:
            Path to saved output image
        """
        # Create intermediate directory
        if save_intermediates:
            output_dir = os.path.dirname(output_path)
            intermediate_dir = os.path.join(output_dir, "intermediates_kontext")
            os.makedirs(os.path.join(intermediate_dir, "01_input"), exist_ok=True)
            os.makedirs(os.path.join(intermediate_dir, "02_composite"), exist_ok=True)
            os.makedirs(os.path.join(intermediate_dir, "03_blended"), exist_ok=True)
        
        # Load images
        base_img = Image.open(base_image_path).convert('RGB')
        ref_img = Image.open(reference_image_path).convert('RGB')
        ref_mask = Image.open(reference_mask_path).convert('L')
        
        # Save inputs
        if save_intermediates:
            base_img.save(os.path.join(intermediate_dir, "01_input", "base_image.png"))
            ref_img.save(os.path.join(intermediate_dir, "01_input", "reference_image.png"))
            ref_mask.save(os.path.join(intermediate_dir, "01_input", "reference_mask.png"))
        
        # Generate initial composite
        composite_img = self.generate_composite_from_mask(base_img, ref_img, ref_mask)
        
        if save_intermediates:
            composite_img.save(os.path.join(intermediate_dir, "02_composite", "initial_composite.png"))
        
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Generate blended results
        blended_images = []
        for i in range(num_samples):
            blended_img = self.pipeline(
                image=composite_img,
                prompt="Connect the hair to scalp.",
                prompt_2="High-quality photograph of a female. The hair is a medium-length, layered brown with subtle lighter highlights that catch the light, falling in soft, wavy strands that cascade down from the temples and crown, framing the face and draping gently over the shoulders and upper torso. The roots show moderate volume, with the hair growing in a slightly forward-swept direction from the scalp, allowing the strands to rest loosely around the ears and neck, creating a natural, relaxed drape without constriction or tension.",
                height=composite_img.height,
                width=composite_img.width,
                guidance_scale=2.5,
                num_inference_steps=40,
                max_sequence_length=512,
            ).images[0]
            
            blended_images.append(blended_img)
            
            if save_intermediates:
                blended_img.save(os.path.join(intermediate_dir, "03_blended", f"blended_{i}.png"))
        
        # Save final output (first sample)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        blended_images[0].save(output_path)
        print(f"Output image saved to: {output_path}")
        
        if save_intermediates:
            print(f"Intermediate results saved to: {intermediate_dir}")
            print(f"  - 01_input: Original inputs")
            print(f"  - 02_composite: Initial composite before blending")
            print(f"  - 03_blended: Blended results")
        
        return output_path


def process_image_kontext(base_image_path, base_mask_path, reference_image_path, 
                          ref_mask_path, output_path, seed=321, save_intermediates=True):
    """
    Process images using Flux Kontext blending.
    
    Note: base_mask_path is not used in Kontext pipeline (only ref_mask_path is used).
    
    Args:
        base_image_path: Path to the target/base image
        base_mask_path: Path to the base mask (not used in Kontext)
        reference_image_path: Path to the reference image
        ref_mask_path: Path to the reference mask
        output_path: Path where the output image will be saved
        seed: Random seed for generation
        save_intermediates: Whether to save intermediate images
    
    Returns:
        Path to the saved output image
    """
    blender = FluxKontextBlender(device=0)
    
    result_path = blender.blend(
        base_image_path=base_image_path,
        reference_image_path=reference_image_path,
        reference_mask_path=ref_mask_path,
        output_path=output_path,
        num_samples=1,
        sample_steps=50,
        guidance_scale=4.0,
        seed=seed,
        save_intermediates=save_intermediates
    )
    
    return result_path


if __name__ == "__main__":
    # Example usage with the same paths from app_no_gradio.py
    base_image_path = image_list[0]
    base_mask_path = image_mask_list[0]
    reference_image_path = ref_list[0]
    ref_mask_path = ref_mask_list[0]
    output_path = "./output_result_kontext.png"
    
    result_path = process_image_kontext(
        base_image_path=base_image_path,
        base_mask_path=base_mask_path,
        reference_image_path=reference_image_path,
        ref_mask_path=ref_mask_path,
        output_path=output_path,
        seed=321
    )
    
    print(f"Processing complete! Result saved to: {result_path}")
