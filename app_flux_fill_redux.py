import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline


dtype = torch.bfloat16
size = (1024, 1024)

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


class FluxFillReduxInpainter:
    """
    Flux Fill Inpainting with Redux prior for hair transfer.
    Combines FLUX.1-Fill-dev for inpainting with FLUX.1-Redux-dev for reference guidance.
    """
    
    def __init__(self, device=0, lora_path=None):
        """
        Initialize Flux Fill + Redux pipelines.
        
        Args:
            device: GPU device ID
            lora_path: Path to LoRA weights (optional)
        """
        self.device = f"cuda:{device}" if isinstance(device, int) else device
        
        # Default LoRA path
        if lora_path is None:
            lora_path = "/workspace/HairPort/Hairdar/insert_anything_lora.safetensors"
        
        self.lora_path = lora_path
        self.build_model(lora_path)
    
    def build_model(self, lora_path):
        """Build the Flux Fill and Redux pipelines."""
        # Initialize Flux Fill pipeline
        self.fill_pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype=dtype
        ).to(self.device)
        
        # Load LoRA weights if available
        if os.path.exists(lora_path):
            self.fill_pipe.load_lora_weights(lora_path)
            print(f"Loaded LoRA weights from: {lora_path}")
        else:
            print(f"LoRA weights not found at {lora_path}, using base model")
        
        # Initialize Redux prior pipeline with text encoders and tokenizers for prompt support
        self.redux_pipe = FluxPriorReduxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Redux-dev",
            text_encoder=self.fill_pipe.text_encoder,
            text_encoder_2=self.fill_pipe.text_encoder_2,
            tokenizer=self.fill_pipe.tokenizer,
            tokenizer_2=self.fill_pipe.tokenizer_2,
            torch_dtype=dtype
        ).to(self.device)
        
        print("Flux Fill + Redux pipelines initialized with text encoders and tokenizers")
    
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
    
    def prepare_reference_image(self, reference_path, mask_path, target_size=(1024, 1024)):
        """
        Prepare reference image by masking out non-hair regions.
        
        Args:
            reference_path: Path to reference image
            mask_path: Path to hair mask
            target_size: Target size
            
        Returns:
            PIL Image - masked reference image with white background
        """
        reference_img = Image.open(reference_path).convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
        hair_mask = Image.open(mask_path).convert('L').resize(target_size, Image.Resampling.NEAREST)
        
        # Convert to numpy
        reference_np = np.array(reference_img)
        mask_np = np.array(hair_mask)
        
        # Create binary mask
        mask_binary = mask_np > 128
        
        # Set pixels to white where mask is black (background)
        reference_np[~mask_binary] = [255, 255, 255]
        
        return Image.fromarray(reference_np)
    
    @torch.no_grad()
    def inpaint(self, source_image_path, reference_image_path, mask_path, 
                output_path,
                prompt="Paste the hair",
                prompt_2="The hair is a medium-length, layered brown with subtle lighter highlights that catch the light, falling in soft, wavy strands that cascade down from the temples and crown, framing the face and draping gently over the shoulders and upper torso. The roots show moderate volume, with the hair growing in a slightly forward-swept direction from the scalp, allowing the strands to rest loosely around the ears and neck, creating a natural, relaxed drape without constriction or tension.",
                num_samples=1, 
                sample_steps=50, 
                guidance_scale=30.0,
                seed=666, 
                save_intermediates=True,
                target_size=(1024, 1024),
                mask_kernel_size=(10, 10),
                mask_iterations=2):
        """
        Inpaint hair using Flux Fill with Redux prior guidance.
        
        Args:
            source_image_path: Path to source/target image (where hair will be added)
            reference_image_path: Path to reference image (with desired hair)
            mask_path: Path to hair mask
            output_path: Path to save output
            prompt: Primary text prompt for Redux prior
            prompt_2: Secondary detailed text prompt for Redux prior
            num_samples: Number of samples to generate
            sample_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
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
            intermediate_dir = os.path.join(output_dir, "intermediates_flux_fill_redux")
            os.makedirs(os.path.join(intermediate_dir, "01_input"), exist_ok=True)
            os.makedirs(os.path.join(intermediate_dir, "02_mask_processing"), exist_ok=True)
            os.makedirs(os.path.join(intermediate_dir, "03_reference_processing"), exist_ok=True)
            os.makedirs(os.path.join(intermediate_dir, "04_redux_embeddings"), exist_ok=True)
            os.makedirs(os.path.join(intermediate_dir, "05_inpainted"), exist_ok=True)
        
        # Load and resize source image
        source_img = Image.open(source_image_path).convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
        
        # Prepare reference image (masked with white background)
        masked_reference_img = self.prepare_reference_image(reference_image_path, mask_path, target_size)
        
        # Prepare dilated mask
        dilated_mask = self.prepare_mask(mask_path, target_size, mask_kernel_size, mask_iterations)
        
        # Save inputs
        if save_intermediates:
            source_img.save(os.path.join(intermediate_dir, "01_input", "source_image.png"))
            
            # Save original reference
            ref_original = Image.open(reference_image_path).convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
            ref_original.save(os.path.join(intermediate_dir, "01_input", "reference_image_original.png"))
            
            # Save original mask
            original_mask = Image.open(mask_path).convert('L').resize(target_size, Image.Resampling.NEAREST)
            original_mask.save(os.path.join(intermediate_dir, "02_mask_processing", "original_mask.png"))
            
            # Save dilated mask
            dilated_mask.save(os.path.join(intermediate_dir, "02_mask_processing", "dilated_mask.png"))
            
            # Create mask comparison
            mask_comparison = np.hstack([
                np.array(original_mask),
                np.array(dilated_mask)
            ])
            Image.fromarray(mask_comparison).save(
                os.path.join(intermediate_dir, "02_mask_processing", "mask_comparison.png"))
            
            # Save masked reference
            masked_reference_img.save(os.path.join(intermediate_dir, "03_reference_processing", "masked_reference.png"))
        
        # Generate Redux prior embeddings from reference
        print("Generating Redux prior embeddings from reference...")
        print(f"Using prompt: {prompt}")
        print(f"Using prompt_2: {prompt_2[:100]}...")  # Print first 100 chars
        
        redux_output = self.redux_pipe(
            masked_reference_img,
            prompt=prompt,
            # prompt_2=prompt_2
        )
        
        if save_intermediates:
            # Save info about redux embeddings and prompts
            with open(os.path.join(intermediate_dir, "04_redux_embeddings", "embeddings_generated.txt"), 'w') as f:
                f.write("Redux prior embeddings generated from masked reference image\n")
                f.write(f"Embedding keys: {list(redux_output.keys())}\n\n")
                f.write(f"Prompt 1:\n{prompt}\n\n")
                f.write(f"Prompt 2:\n{prompt_2}\n")
        
        # Create inpainting input by compositing source with masked area
        source_np = np.array(source_img)
        mask_np = np.array(dilated_mask)
        
        # Create 3-channel mask
        mask_3ch = np.stack([mask_np, mask_np, mask_np], axis=-1) / 255.0
        
        # Composite: keep source outside mask, white inside mask for inpainting
        composite_np = source_np.copy()
        composite_np = (composite_np * (1 - mask_3ch) + 255 * mask_3ch).astype(np.uint8)
        composite_img = Image.fromarray(composite_np)
        
        if save_intermediates:
            composite_img.save(os.path.join(intermediate_dir, "01_input", "composite_for_inpainting.png"))
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            generator = torch.Generator(self.device).manual_seed(seed)
        else:
            generator = None
        
        # Generate inpainted results
        inpainted_images = []
        for i in range(num_samples):
            print(f"Generating sample {i+1}/{num_samples}...")
            
            inpainted_img = self.fill_pipe(
                image=composite_img,
                mask_image=dilated_mask,
                height=target_size[1],
                width=target_size[0],
                guidance_scale=3.0,
                num_inference_steps=28,
                max_sequence_length=512,
                generator=generator,
                **redux_output,  # Pass Redux prior embeddings
            ).images[0]
            
            inpainted_images.append(inpainted_img)
            
            if save_intermediates:
                inpainted_img.save(os.path.join(intermediate_dir, "05_inpainted", f"inpainted_{i}.png"))
        
        # Save final output (first sample)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        inpainted_images[0].save(output_path)
        print(f"Output image saved to: {output_path}")
        
        if save_intermediates:
            print(f"\nIntermediate results saved to: {intermediate_dir}")
            print(f"  - 01_input: Source, reference, and composite images")
            print(f"  - 02_mask_processing: Original and dilated masks")
            print(f"  - 03_reference_processing: Masked reference for Redux")
            print(f"  - 04_redux_embeddings: Redux prior information with prompts")
            print(f"  - 05_inpainted: Inpainted results")
            print(f"\nMask dilation: {mask_kernel_size} kernel, {mask_iterations} iterations")
            print(f"Guidance scale: {guidance_scale}, Steps: {sample_steps}")
            print(f"Prompts used for Redux prior conditioning")
        
        return output_path


def process_image_flux_fill_redux(source_image_path, reference_image_path, 
                                   mask_path, output_path,
                                   prompt="Add hair to the image while preserving the identity, style, and background.",
                                   prompt_2="The hair is a medium-length, layered brown with subtle lighter highlights that catch the light, falling in soft, wavy strands that cascade down from the temples and crown, framing the face and draping gently over the shoulders and upper torso. The roots show moderate volume, with the hair growing in a slightly forward-swept direction from the scalp, allowing the strands to rest loosely around the ears and neck, creating a natural, relaxed drape without constriction or tension.",
                                   guidance_scale=30.0,
                                   num_inference_steps=50,
                                   seed=666, 
                                   save_intermediates=True):
    """
    Process images using Flux Fill + Redux pipeline.
    
    Args:
        source_image_path: Path to the source/target image
        reference_image_path: Path to the reference image with hair
        mask_path: Path to the hair mask
        output_path: Path where the output image will be saved
        prompt: Primary text prompt for Redux prior
        prompt_2: Secondary detailed text prompt for Redux prior
        guidance_scale: Classifier-free guidance scale
        num_inference_steps: Number of denoising steps
        seed: Random seed for generation
        save_intermediates: Whether to save intermediate images
    
    Returns:
        Path to the saved output image
    """
    inpainter = FluxFillReduxInpainter(device=0)
    
    result_path = inpainter.inpaint(
        source_image_path=source_image_path,
        reference_image_path=reference_image_path,
        mask_path=mask_path,
        output_path=output_path,
        prompt=prompt,
        prompt_2=prompt_2,
        num_samples=1,
        sample_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        save_intermediates=save_intermediates,
        target_size=(1024, 1024),
        mask_kernel_size=(10, 10),
        mask_iterations=2
    )
    
    return result_path


if __name__ == "__main__":
    # Example usage
    source_image_path = image_list[0]  # Source/target image (where hair will be added)
    reference_image_path = ref_list[0]  # Reference image (with desired hair)
    mask_path = ref_mask_list[0]  # Hair mask
    output_path = "./output_result_flux_fill_redux.png"
    
    print("=" * 80)
    print("Flux Fill + Redux Hair Transfer")
    print("=" * 80)
    print(f"Source image: {source_image_path}")
    print(f"Reference image: {reference_image_path}")
    print(f"Mask: {mask_path}")
    print(f"Output: {output_path}")
    print("=" * 80)
    
    result_path = process_image_flux_fill_redux(
        source_image_path=source_image_path,
        reference_image_path=reference_image_path,
        mask_path=mask_path,
        output_path=output_path,
        prompt="Add hair to the image while preserving the identity, style, and background.",
        prompt_2="The hair is a medium-length, layered brown with subtle lighter highlights that catch the light, falling in soft, wavy strands that cascade down from the temples and crown, framing the face and draping gently over the shoulders and upper torso. The roots show moderate volume, with the hair growing in a slightly forward-swept direction from the scalp, allowing the strands to rest loosely around the ears and neck, creating a natural, relaxed drape without constriction or tension.",
        guidance_scale=4.0,
        num_inference_steps=50,
        seed=6626,
        save_intermediates=True
    )
    
    print(f"\nProcessing complete! Result saved to: {result_path}")
