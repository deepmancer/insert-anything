import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
import math
from utils.utils import get_bbox_from_mask, expand_bbox, pad_to_square, box2squre, crop_back, expand_image_mask


dtype = torch.bfloat16
size = (1024, 1024)

pipe = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev",
    torch_dtype=dtype
).to("cuda")

pipe.load_lora_weights(
    "/workspace/HairPort/Hairdar/insert_anything_lora.safetensors"
)

redux = FluxPriorReduxPipeline.from_pretrained("black-forest-labs/FLUX.1-Redux-dev").to(dtype=dtype).to("cuda")

target_id = "ana_2"
source_id = "taylor_2"
data_dir = "/workspace/outputs"
view_aligned_dir = os.path.join(data_dir, "view_aligned/shape_hi3dgen__texture_mvadapter", f"{target_id}_to_{source_id}")
import os
###   example  #####
ref_dir=os.path.join(view_aligned_dir, "warping/target_image.png")
ref_mask_dir=os.path.join(view_aligned_dir, "warping/target_hair_mask.png")
image_dir=os.path.join(data_dir, "bald/w_seg/image", f"{source_id}.png")
image_mask_dir=ref_mask_dir
# os.path.join(data_dir, "bald/w_seg/mask", f"{source_id}.png")

ref_list=[ref_dir]
ref_list.sort()

ref_mask_list=[ref_mask_dir]
ref_mask_list.sort()

image_list=[image_dir]
image_list.sort()

image_mask_list=[image_mask_dir]
image_mask_list.sort()
###   example  #####


def process_image(base_image_path, base_mask_path, reference_image_path, ref_mask_path, output_path, seed=666, save_intermediates=True):
    """
    Process images to insert reference object into base image.
    
    Args:
        base_image_path: Path to the background/target image
        base_mask_path: Path to the background mask image
        reference_image_path: Path to the reference image
        ref_mask_path: Path to the reference mask image
        output_path: Path where the output image will be saved
        seed: Random seed for generation (default: 666)
        save_intermediates: Whether to save intermediate images and masks (default: True)
    
    Returns:
        Path to the saved output image
    """
    
    # Create intermediate output directory structure
    if save_intermediates:
        output_dir = os.path.dirname(output_path)
        intermediate_dir = os.path.join(output_dir, "intermediates")
        os.makedirs(os.path.join(intermediate_dir, "01_input"), exist_ok=True)
        os.makedirs(os.path.join(intermediate_dir, "02_preprocessing"), exist_ok=True)
        os.makedirs(os.path.join(intermediate_dir, "03_reference_processing"), exist_ok=True)
        os.makedirs(os.path.join(intermediate_dir, "04_target_processing"), exist_ok=True)
        os.makedirs(os.path.join(intermediate_dir, "05_model_input"), exist_ok=True)
        os.makedirs(os.path.join(intermediate_dir, "06_generation"), exist_ok=True)
        os.makedirs(os.path.join(intermediate_dir, "07_bbox_overlays"), exist_ok=True)
    # Load images
    tar_image = Image.open(base_image_path).convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
    tar_mask = Image.open(base_mask_path).convert("L").resize((1024, 1024), Image.Resampling.NEAREST)
    ref_image = Image.open(reference_image_path).convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
    ref_mask = Image.open(ref_mask_path).convert("L").resize((1024, 1024), Image.Resampling.NEAREST)
    
    # Save original inputs
    if save_intermediates:
        tar_image.save(os.path.join(intermediate_dir, "01_input", "target_image_resized.png"))
        tar_mask.save(os.path.join(intermediate_dir, "01_input", "target_mask_resized.png"))
        ref_image.save(os.path.join(intermediate_dir, "01_input", "reference_image_resized.png"))
        ref_mask.save(os.path.join(intermediate_dir, "01_input", "reference_mask_resized.png"))

    # Convert to numpy arrays
    tar_image = np.asarray(tar_image)
    tar_mask = np.asarray(tar_mask)
    tar_mask = np.where(tar_mask > 128, 1, 0).astype(np.uint8)

    ref_image = np.asarray(ref_image)
    ref_mask = np.asarray(ref_mask)
    ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)
    
    # Save binarized masks
    if save_intermediates:
        Image.fromarray((tar_mask * 255).astype(np.uint8)).save(
            os.path.join(intermediate_dir, "02_preprocessing", "target_mask_binary.png"))
        Image.fromarray((ref_mask * 255).astype(np.uint8)).save(
            os.path.join(intermediate_dir, "02_preprocessing", "reference_mask_binary.png"))

    if tar_mask.sum() == 0:
        raise ValueError('No mask for the background image. Please check the mask!')

    if ref_mask.sum() == 0:
        raise ValueError('No mask for the reference image. Please check the mask!')

    ref_box_yyxx = get_bbox_from_mask(ref_mask)
    
    # Visualize reference bounding box on mask
    if save_intermediates:
        ref_mask_vis = np.stack([ref_mask * 255, ref_mask * 255, ref_mask * 255], -1).astype(np.uint8)
        y1_box, y2_box, x1_box, x2_box = ref_box_yyxx
        cv2.rectangle(ref_mask_vis, (x1_box, y1_box), (x2_box, y2_box), (0, 255, 0), 3)
        cv2.putText(ref_mask_vis, 'Ref BBox', (x1_box, y1_box - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        Image.fromarray(ref_mask_vis).save(
            os.path.join(intermediate_dir, "03_reference_processing", "00_bbox_on_mask.png"))
    
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3) 
    
    # Save masked reference image
    if save_intermediates:
        Image.fromarray(masked_ref_image.astype(np.uint8)).save(
            os.path.join(intermediate_dir, "03_reference_processing", "01_masked_reference.png"))
    
    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2] 
    
    # Save cropped reference
    if save_intermediates:
        Image.fromarray(masked_ref_image.astype(np.uint8)).save(
            os.path.join(intermediate_dir, "03_reference_processing", "02_cropped_reference.png"))
        Image.fromarray((ref_mask * 255).astype(np.uint8)).save(
            os.path.join(intermediate_dir, "03_reference_processing", "02_cropped_reference_mask.png"))
    
    ratio = 1.3
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    
    # Save expanded reference
    if save_intermediates:
        Image.fromarray(masked_ref_image.astype(np.uint8)).save(
            os.path.join(intermediate_dir, "03_reference_processing", "03_expanded_reference.png"))
        Image.fromarray((ref_mask * 255).astype(np.uint8)).save(
            os.path.join(intermediate_dir, "03_reference_processing", "03_expanded_reference_mask.png"))


    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False) 
    
    # Save padded reference
    if save_intermediates:
        Image.fromarray(masked_ref_image.astype(np.uint8)).save(
            os.path.join(intermediate_dir, "03_reference_processing", "04_padded_reference.png")) 

    kernel = np.ones((13, 13), np.uint8)
    iterations = 2
    tar_mask = cv2.dilate(tar_mask, kernel, iterations=iterations)
    
    # Save dilated target mask
    if save_intermediates:
        Image.fromarray((tar_mask * 255).astype(np.uint8)).save(
            os.path.join(intermediate_dir, "04_target_processing", "01_dilated_target_mask.png"))

    # zome in
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx_expanded = expand_bbox(tar_mask, tar_box_yyxx, ratio=1.2)

    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx_expanded, ratio=2)    #1.2 1.6
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    
    # Visualize target bounding boxes on mask
    if save_intermediates:
        tar_mask_vis = np.stack([tar_mask * 255, tar_mask * 255, tar_mask * 255], -1).astype(np.uint8)
        # Draw original bbox
        y1_orig, y2_orig, x1_orig, x2_orig = tar_box_yyxx
        cv2.rectangle(tar_mask_vis, (x1_orig, y1_orig), (x2_orig, y2_orig), (255, 0, 0), 2)
        cv2.putText(tar_mask_vis, 'Original', (x1_orig, y1_orig - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        # Draw expanded bbox
        y1_exp, y2_exp, x1_exp, x2_exp = tar_box_yyxx_expanded
        cv2.rectangle(tar_mask_vis, (x1_exp, y1_exp), (x2_exp, y2_exp), (0, 255, 255), 2)
        cv2.putText(tar_mask_vis, 'Expanded 1.2x', (x1_exp, y1_exp - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # Draw crop bbox
        y1_crop, y2_crop, x1_crop, x2_crop = tar_box_yyxx_crop
        cv2.rectangle(tar_mask_vis, (x1_crop, y1_crop), (x2_crop, y2_crop), (0, 255, 0), 3)
        cv2.putText(tar_mask_vis, 'Crop Box (2x, square)', (x1_crop, y1_crop - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        Image.fromarray(tar_mask_vis).save(
            os.path.join(intermediate_dir, "04_target_processing", "00_bbox_on_mask.png"))
    
    y1,y2,x1,x2 = tar_box_yyxx_crop


    old_tar_image = tar_image.copy()
    tar_image = tar_image[y1:y2,x1:x2,:]
    tar_mask = tar_mask[y1:y2,x1:x2]
    
    # Save cropped target
    if save_intermediates:
        Image.fromarray(tar_image.astype(np.uint8)).save(
            os.path.join(intermediate_dir, "04_target_processing", "02_cropped_target.png"))
        Image.fromarray((tar_mask * 255).astype(np.uint8)).save(
            os.path.join(intermediate_dir, "04_target_processing", "02_cropped_target_mask.png"))

    H1, W1 = tar_image.shape[0], tar_image.shape[1]
    # zome in


    tar_mask = pad_to_square(tar_mask, pad_value=0)
    tar_mask = cv2.resize(tar_mask, size)
    
    # Save padded and resized target mask
    if save_intermediates:
        Image.fromarray((tar_mask * 255).astype(np.uint8)).save(
            os.path.join(intermediate_dir, "04_target_processing", "03_padded_resized_target_mask.png"))

    masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), size).astype(np.uint8)
    
    # Save final resized reference
    if save_intermediates:
        Image.fromarray(masked_ref_image).save(
            os.path.join(intermediate_dir, "03_reference_processing", "05_final_resized_reference.png"))
    
    pipe_prior_output = redux(Image.fromarray(masked_ref_image))


    tar_image = pad_to_square(tar_image, pad_value=255)

    H2, W2 = tar_image.shape[0], tar_image.shape[1]

    tar_image = cv2.resize(tar_image, size)
    
    # Save final resized target
    if save_intermediates:
        Image.fromarray(tar_image.astype(np.uint8)).save(
            os.path.join(intermediate_dir, "04_target_processing", "04_final_padded_resized_target.png"))
    
    diptych_ref_tar = np.concatenate([masked_ref_image, tar_image], axis=1)


    tar_mask = np.stack([tar_mask,tar_mask,tar_mask],-1)
    mask_black = np.ones_like(tar_image) * 0
    mask_diptych = np.concatenate([mask_black, tar_mask], axis=1)


    diptych_ref_tar = Image.fromarray(diptych_ref_tar)
    mask_diptych[mask_diptych == 1] = 255
    mask_diptych = Image.fromarray(mask_diptych)
    
    # Save model inputs
    if save_intermediates:
        diptych_ref_tar.save(os.path.join(intermediate_dir, "05_model_input", "diptych_ref_tar.png"))
        mask_diptych.save(os.path.join(intermediate_dir, "05_model_input", "mask_diptych.png"))

    generator = torch.Generator("cuda").manual_seed(seed)
    edited_image = pipe(
        image=diptych_ref_tar,
        mask_image=mask_diptych,
        height=mask_diptych.size[1],
        width=mask_diptych.size[0],
        max_sequence_length=512,
        generator=generator,
        guidance_scale=10.0,
        num_inference_steps=50,
        **pipe_prior_output, 
    ).images[0]

    width, height = edited_image.size
    left = width // 2
    right = width
    top = 0
    bottom = height
    edited_image = edited_image.crop((left, top, right, bottom))
    
    # Save raw model output
    if save_intermediates:
        edited_image.save(os.path.join(intermediate_dir, "06_generation", "01_raw_model_output.png"))


    edited_image = np.array(edited_image)
    edited_image = crop_back(edited_image, old_tar_image, np.array([H1, W1, H2, W2]), np.array(tar_box_yyxx_crop)) 
    
    # Save cropped back image
    if save_intermediates:
        Image.fromarray(edited_image).save(
            os.path.join(intermediate_dir, "06_generation", "02_cropped_back.png"))
    
    edited_image = Image.fromarray(edited_image)

    # Save the output image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    edited_image.save(output_path)
    print(f"Output image saved to: {output_path}")
    
    if save_intermediates:
        # Create final bbox overlay comparison on masks
        ref_mask_rgb = np.stack([ref_mask * 255, ref_mask * 255, ref_mask * 255], -1).astype(np.uint8)
        y1_ref, y2_ref, x1_ref, x2_ref = ref_box_yyxx
        cv2.rectangle(ref_mask_rgb, (x1_ref, y1_ref), (x2_ref, y2_ref), (0, 255, 0), 3)
        cv2.putText(ref_mask_rgb, 'Reference Crop Region', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Get the original tar_mask before cropping (need to recreate from old_tar_image size)
        tar_mask_orig = np.asarray(Image.open(base_mask_path).convert("L").resize((1024, 1024), Image.Resampling.NEAREST))
        tar_mask_orig = np.where(tar_mask_orig > 128, 1, 0).astype(np.uint8)
        kernel = np.ones((10, 10), np.uint8)
        tar_mask_orig = cv2.dilate(tar_mask_orig, kernel, iterations=2)
        tar_mask_rgb = np.stack([tar_mask_orig * 255, tar_mask_orig * 255, tar_mask_orig * 255], -1).astype(np.uint8)
        y1_final, y2_final, x1_final, x2_final = tar_box_yyxx_crop
        cv2.rectangle(tar_mask_rgb, (x1_final, y1_final), (x2_final, y2_final), (0, 255, 0), 3)
        cv2.putText(tar_mask_rgb, 'Target Crop Region', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        Image.fromarray(ref_mask_rgb).save(
            os.path.join(intermediate_dir, "07_bbox_overlays", "reference_bbox_on_mask.png"))
        Image.fromarray(tar_mask_rgb).save(
            os.path.join(intermediate_dir, "07_bbox_overlays", "target_bbox_on_mask.png"))
        
        print(f"Intermediate results saved to: {intermediate_dir}")
        print(f"  - 01_input: Original resized inputs")
        print(f"  - 02_preprocessing: Binarized masks")
        print(f"  - 03_reference_processing: Reference image processing steps")
        print(f"  - 04_target_processing: Target image processing steps")
        print(f"  - 05_model_input: Final inputs to the model")
        print(f"  - 06_generation: Model output and final composition")
        print(f"  - 07_bbox_overlays: Bounding box visualizations on masks")

    return output_path


if __name__ == "__main__":
    # Example usage with the default paths from the original code
    base_image_path = image_list[0]
    base_mask_path = image_mask_list[0]
    reference_image_path = ref_list[0]
    ref_mask_path = ref_mask_list[0]
    output_path = "./output_result.png"
    
    result_path = process_image(
        base_image_path=base_image_path,
        base_mask_path=base_mask_path,
        reference_image_path=reference_image_path,
        ref_mask_path=ref_mask_path,
        output_path=output_path,
        seed=666
    )
    
    print(f"Processing complete! Result saved to: {result_path}")
