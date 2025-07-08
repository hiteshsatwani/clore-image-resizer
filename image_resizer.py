#!/usr/bin/env python3
"""
AI-powered image resizer to 9:16 aspect ratio with realistic extension.
Uses advanced inpainting models for seamless results.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline
from transformers import pipeline


class ImageResizer:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.target_ratio = 9 / 16  # 9:16 aspect ratio
        
        # Load models
        print("Loading AI models...")
        self.inpaint_pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(device)
        
        # Enable memory efficient attention
        self.inpaint_pipeline.enable_model_cpu_offload()
        self.inpaint_pipeline.enable_vae_slicing()
        
        # For content-aware fill detection
        self.depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large", device=device)
        
        print("Models loaded successfully!")
    
    def calculate_target_dimensions(self, width: int, height: int) -> Tuple[int, int]:
        """Calculate target dimensions for 9:16 aspect ratio."""
        current_ratio = width / height
        
        if current_ratio > self.target_ratio:
            # Image is too wide, need to extend height
            new_width = width
            new_height = int(width / self.target_ratio)
        else:
            # Image is too tall, need to extend width
            new_height = height
            new_width = int(height * self.target_ratio)
        
        return new_width, new_height
    
    def create_extension_mask(self, original_img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Create mask for areas that need to be inpainted."""
        target_width, target_height = target_size
        orig_width, orig_height = original_img.size
        
        # Create canvas
        mask = Image.new('L', target_size, 255)  # White = inpaint
        
        # Calculate centering position
        x_offset = (target_width - orig_width) // 2
        y_offset = (target_height - orig_height) // 2
        
        # Black out the original image area (don't inpaint)
        mask.paste(0, (x_offset, y_offset, x_offset + orig_width, y_offset + orig_height))
        
        return mask
    
    def create_base_canvas(self, original_img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Create base canvas with original image centered."""
        target_width, target_height = target_size
        orig_width, orig_height = original_img.size
        
        # Create canvas with edge-extended background
        canvas = Image.new('RGB', target_size, (128, 128, 128))
        
        # Calculate centering position
        x_offset = (target_width - orig_width) // 2
        y_offset = (target_height - orig_height) // 2
        
        # Paste original image
        canvas.paste(original_img, (x_offset, y_offset))
        
        # Create edge extension for better context
        if x_offset > 0:  # Need to extend horizontally
            # Left edge
            left_edge = original_img.crop((0, 0, 1, orig_height))
            left_edge = left_edge.resize((x_offset, orig_height))
            canvas.paste(left_edge, (0, y_offset))
            
            # Right edge
            right_edge = original_img.crop((orig_width-1, 0, orig_width, orig_height))
            right_edge = right_edge.resize((x_offset, orig_height))
            canvas.paste(right_edge, (x_offset + orig_width, y_offset))
        
        if y_offset > 0:  # Need to extend vertically
            # Top edge
            top_edge = original_img.crop((0, 0, orig_width, 1))
            top_edge = top_edge.resize((orig_width, y_offset))
            canvas.paste(top_edge, (x_offset, 0))
            
            # Bottom edge
            bottom_edge = original_img.crop((0, orig_height-1, orig_width, orig_height))
            bottom_edge = bottom_edge.resize((orig_width, y_offset))
            canvas.paste(bottom_edge, (x_offset, y_offset + orig_height))
        
        return canvas
    
    def generate_inpaint_prompt(self) -> str:
        """Generate contextual prompt for inpainting based on image content."""
        # Simple context-aware prompting
        # In production, you might use CLIP or BLIP for better scene understanding
        return "seamless natural extension, photorealistic, high quality, detailed background, consistent lighting and style"
    
    def resize_image(self, image_path: str, output_path: str) -> bool:
        """Resize image to 9:16 aspect ratio using AI inpainting."""
        try:
            # Load image
            original_img = Image.open(image_path).convert('RGB')
            orig_width, orig_height = original_img.size
            
            print(f"Original size: {orig_width}x{orig_height}")
            
            # Calculate target dimensions
            target_width, target_height = self.calculate_target_dimensions(orig_width, orig_height)
            print(f"Target size: {target_width}x{target_height}")
            
            # Check if already correct ratio
            if orig_width == target_width and orig_height == target_height:
                print("Image already has correct aspect ratio")
                original_img.save(output_path, quality=95)
                return True
            
            # Create base canvas and mask
            base_canvas = self.create_base_canvas(original_img, (target_width, target_height))
            mask = self.create_extension_mask(original_img, (target_width, target_height))
            
            # Generate inpainting prompt
            prompt = self.generate_inpaint_prompt()
            
            # Resize for processing (SDXL works best at 1024x1024)
            process_size = 1024
            scale_factor = process_size / max(target_width, target_height)
            
            if scale_factor < 1:
                process_width = int(target_width * scale_factor)
                process_height = int(target_height * scale_factor)
                
                base_resized = base_canvas.resize((process_width, process_height), Image.LANCZOS)
                mask_resized = mask.resize((process_width, process_height), Image.LANCZOS)
            else:
                base_resized = base_canvas
                mask_resized = mask
                process_width, process_height = target_width, target_height
            
            print("Running AI inpainting...")
            
            # Run inpainting
            result = self.inpaint_pipeline(
                prompt=prompt,
                image=base_resized,
                mask_image=mask_resized,
                num_inference_steps=20,  # Reduced for speed
                strength=0.8,
                guidance_scale=7.5,
                height=process_height,
                width=process_width
            ).images[0]
            
            # Resize back to target size if needed
            if scale_factor < 1:
                result = result.resize((target_width, target_height), Image.LANCZOS)
            
            # Save result
            result.save(output_path, quality=95)
            print(f"Saved resized image to: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="AI-powered image resizer to 9:16 aspect ratio")
    parser.add_argument("input", help="Input image path or directory")
    parser.add_argument("-o", "--output", help="Output path or directory")
    parser.add_argument("-d", "--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("-b", "--batch", action="store_true", help="Batch process directory")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output")
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Initialize resizer
    resizer = ImageResizer(device=args.device)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image processing
        output_path = args.output if args.output else f"{input_path.stem}_resized{input_path.suffix}"
        
        start_time = time.time()
        success = resizer.resize_image(str(input_path), output_path)
        end_time = time.time()
        
        if success and not args.quiet:
            print(f"Processing completed in {end_time - start_time:.2f} seconds")
    
    elif input_path.is_dir() and args.batch:
        # Batch processing
        output_dir = Path(args.output) if args.output else input_path / "resized"
        output_dir.mkdir(exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print("No image files found in directory")
            return
        
        successful = 0
        total_time = 0
        
        for i, image_file in enumerate(image_files, 1):
            if not args.quiet:
                print(f"Processing {i}/{len(image_files)}: {image_file.name}")
            
            output_file = output_dir / f"{image_file.stem}_resized{image_file.suffix}"
            
            start_time = time.time()
            success = resizer.resize_image(str(image_file), str(output_file))
            end_time = time.time()
            
            if success:
                successful += 1
                total_time += (end_time - start_time)
            
            if not args.quiet:
                print(f"  -> {'Success' if success else 'Failed'} ({end_time - start_time:.2f}s)")
        
        if not args.quiet:
            avg_time = total_time / successful if successful > 0 else 0
            print(f"\nBatch processing completed:")
            print(f"  Successful: {successful}/{len(image_files)}")
            print(f"  Average time: {avg_time:.2f}s per image")
    
    else:
        print("Invalid input path or missing --batch flag for directory processing")
        sys.exit(1)


if __name__ == "__main__":
    main()