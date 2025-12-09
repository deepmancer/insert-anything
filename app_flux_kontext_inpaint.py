import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import FluxTransformer2DModel


from diffusers import FluxKontextInpaintPipeline


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
# os.path.join(view_aligned_dir, "blended.png")
# os.path.join(data_dir, "bald/w_seg/image", f"{source_id}.png")
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


class FluxKontextInpainter:
    """
    Flux Kontext Inpaint pipeline for hair transfer with reference guidance.
    """
    
    def __init__(self, device=0, lora_path=None):
        """
        Initialize Flux Kontext Inpaint model.
        
        Args:
            device: GPU device ID
            lora_path: Path to inpainting LoRA weights (optional)
        """
        self.device = f"cuda:{device}" if isinstance(device, int) else device
        weight_path = "black-forest-labs/FLUX.1-Kontext-dev"
        
        # Use provided lora_path or default
        if lora_path is None:
            lora_path = "/workspace/HairPort/Hairdar/flux_kontext_inpaint.safetensors"
        
        self.lora_path = lora_path
        self.build_model(weight_path, lora_path)
    
    def build_model(self, weight_path, lora_path):
        """Build the Flux Kontext Inpaint pipeline."""
        transformer = FluxTransformer2DModel.from_pretrained(
            weight_path, 
            subfolder="transformer"
        )
        transformer.requires_grad_(False)
        
        self.pipeline = FluxKontextInpaintPipeline.from_pretrained(
            weight_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16
        )
        
        # Load LoRA weights if they exist
        if os.path.exists(lora_path):
            self.pipeline.load_lora_weights(lora_path)
            print(f"Loaded LoRA weights from: {lora_path}")
        else:
            print(f"LoRA weights not found at {lora_path}, using base model")
        
        self.pipeline = self.pipeline.to(self.device, dtype=torch.bfloat16)
    
    def prepare_mask(self, mask_path, target_size=(1024, 1024), kernel_size=(10, 10), iterations=2):
        """
        Load and prepare the mask by dilating it.
        
        Args:
            mask_path: Path to the mask image
            target_size: Tuple - output size
            kernel_size: Tuple - dilation kernel size
            iterations: Number of dilation iterations
            
        Returns:
            PIL Image - dilated mask
        """
        # Load and resize mask
        mask = Image.open(mask_path).convert('L').resize(target_size, Image.Resampling.NEAREST)
        mask_np = np.array(mask)
        
        # Binarize mask (>128 is foreground)
        mask_binary = np.where(mask_np > 128, 1, 0).astype(np.uint8)
        
        # Dilate the mask to expand the region
        kernel = np.ones(kernel_size, np.uint8)
        mask_dilated = cv2.dilate(mask_binary, kernel, iterations=iterations)
        
        # Convert back to 0-255 range
        mask_dilated = (mask_dilated * 255).astype(np.uint8)
        
        return Image.fromarray(mask_dilated)
    
    @torch.no_grad()
    def inpaint(self, source_image_path, reference_image_path, mask_path, 
                output_path, 
                prompt="add hair to the image while preserving the identity, style, and background",
                num_samples=1, 
                sample_steps=28, 
                guidance_scale=3.5,
                strength=1.0,
                seed=321, 
                save_intermediates=True,
                target_size=(1024, 1024),
                mask_kernel_size=(10, 10),
                mask_iterations=2):
        """
        Inpaint hair from reference onto source image using Flux Kontext Inpaint.
        
        Args:
            source_image_path: Path to source/target image (where hair will be added)
            reference_image_path: Path to reference image (with desired hair)
            mask_path: Path to hair mask
            output_path: Path to save output
            prompt: Text prompt for generation
            num_samples: Number of samples to generate
            sample_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            strength: Denoising strength (0.0-1.0, higher = more changes)
            seed: Random seed
            save_intermediates: Whether to save intermediate results
            target_size: Output image size
            mask_kernel_size: Kernel size for mask dilation
            mask_iterations: Number of dilation iterations
            
        Returns:
            Path to saved output image
        """
        # Create intermediate directory
        if save_intermediates:
            output_dir = os.path.dirname(output_path)
            intermediate_dir = os.path.join(output_dir, "intermediates_kontext_inpaint")
            os.makedirs(os.path.join(intermediate_dir, "01_input"), exist_ok=True)
            os.makedirs(os.path.join(intermediate_dir, "02_mask_processing"), exist_ok=True)
            os.makedirs(os.path.join(intermediate_dir, "03_inpainted"), exist_ok=True)
        
        # Load and resize source image
        source_img = Image.open(source_image_path).convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
        
        # Load and resize reference image
        reference_img = Image.open(reference_image_path).convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
        hair_mask = Image.open(mask_path).convert('L').resize(target_size, Image.Resampling.NEAREST)
        # Make RGB pixels black in reference_img where hair_mask is black
        reference_np = np.array(reference_img)
        hair_mask_np = np.array(hair_mask)

        # Create a binary mask (hair_mask black pixels = 0, white pixels = 255)
        mask_binary = hair_mask_np > 128

        # Set pixels to black where mask is black (False in mask_binary)
        reference_np[~mask_binary] = [0, 0, 0]

        # Convert back to PIL Image
        reference_img = Image.fromarray(reference_np)

        # Prepare dilated mask
        dilated_mask = self.prepare_mask(mask_path, target_size, mask_kernel_size, mask_iterations)
        
        # Save inputs
        if save_intermediates:
            source_img.save(os.path.join(intermediate_dir, "01_input", "source_image.png"))
            reference_img.save(os.path.join(intermediate_dir, "01_input", "reference_image.png"))
            
            # Save original mask
            original_mask = Image.open(mask_path).convert('L').resize(target_size, Image.Resampling.NEAREST)
            original_mask.save(os.path.join(intermediate_dir, "02_mask_processing", "original_mask.png"))
            
            # Save dilated mask
            dilated_mask.save(os.path.join(intermediate_dir, "02_mask_processing", "dilated_mask.png"))
            
            # Create visualization showing mask expansion
            mask_comparison = np.hstack([
                np.array(original_mask),
                np.array(dilated_mask)
            ])
            Image.fromarray(mask_comparison).save(
                os.path.join(intermediate_dir, "02_mask_processing", "mask_comparison.png"))
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Generate inpainted results
        inpainted_images = []
        for i in range(num_samples):
            print(f"Generating sample {i+1}/{num_samples}...")
            
            inpainted_img = self.pipeline(
                image=source_img,
                image_reference=reference_img,
                mask_image=dilated_mask,
                prompt=prompt,
                height=target_size[1],
                width=target_size[0],
                guidance_scale=guidance_scale,
                strength=strength,
                num_inference_steps=sample_steps,
                max_sequence_length=512,
                generator=generator,
            ).images[0]
            
            inpainted_images.append(inpainted_img)
            
            if save_intermediates:
                inpainted_img.save(os.path.join(intermediate_dir, "03_inpainted", f"inpainted_{i}.png"))
        
        # Save final output (first sample)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        inpainted_images[0].save(output_path)
        print(f"Output image saved to: {output_path}")
        
        if save_intermediates:
            print(f"\nIntermediate results saved to: {intermediate_dir}")
            print(f"  - 01_input: Source and reference images")
            print(f"  - 02_mask_processing: Original and dilated masks")
            print(f"  - 03_inpainted: Inpainted results")
            print(f"\nMask dilation: {mask_kernel_size} kernel, {mask_iterations} iterations")
            print(f"Prompt: '{prompt}'")
            print(f"Guidance scale: {guidance_scale}, Strength: {strength}")
        
        return output_path


def process_image_kontext_inpaint(source_image_path, reference_image_path, 
                                   mask_path, output_path, 
                                   prompt="add hair to the image while preserving the identity, style, and background",
                                   guidance_scale=3.5,
                                   strength=1.0,
                                   seed=321, 
                                   save_intermediates=True):
    """
    Process images using Flux Kontext Inpaint pipeline.
    
    Args:
        source_image_path: Path to the source/target image
        reference_image_path: Path to the reference image with hair
        mask_path: Path to the hair mask
        output_path: Path where the output image will be saved
        prompt: Text prompt for generation
        guidance_scale: Classifier-free guidance scale
        strength: Denoising strength (0.0-1.0)
        seed: Random seed for generation
        save_intermediates: Whether to save intermediate images
    
    Returns:
        Path to the saved output image
    """
    inpainter = FluxKontextInpainter(device=0)
    
    result_path = inpainter.inpaint(
        source_image_path=source_image_path,
        reference_image_path=reference_image_path,
        mask_path=mask_path,
        output_path=output_path,
        prompt=prompt,
        num_samples=1,
        sample_steps=28,
        guidance_scale=1.2,
        strength=strength,
        seed=seed,
        save_intermediates=save_intermediates,
        target_size=(1024, 1024),
        mask_kernel_size=(3, 3),
        mask_iterations=1,
    )
    
    return result_path


if __name__ == "__main__":
    # Example usage with the same paths structure
    source_image_path = image_list[0]  # Source/target image (where hair will be added)
    reference_image_path = ref_list[0]  # Reference image (with desired hair)
    mask_path = ref_mask_list[0]  # Hair mask
    output_path = "./output_result_kontext_inpaint.png"
    
    print("=" * 80)
    print("Flux Kontext Inpaint Hair Transfer")
    print("=" * 80)
    print(f"Source image: {source_image_path}")
    print(f"Reference image: {reference_image_path}")
    print(f"Mask: {mask_path}")
    print(f"Output: {output_path}")
    print("=" * 80)
    
    result_path = process_image_kontext_inpaint(
        source_image_path=source_image_path,
        reference_image_path=reference_image_path,
        mask_path=mask_path,
        output_path=output_path,
        prompt="add hair to the image while preserving the identity, style, and background",
        guidance_scale=3.5,
        strength=1.0,
        seed=321,
        save_intermediates=True
    )
    
    print(f"\nProcessing complete! Result saved to: {result_path}")
