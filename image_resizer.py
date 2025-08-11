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
from PIL import Image, ImageFilter
from diffusers import StableDiffusionXLInpaintPipeline
from transformers import pipeline
import numpy as np
from scipy import ndimage


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
    
    def has_white_background(self, image: Image.Image, white_threshold: float = 0.7, edge_sample_ratio: float = 0.1) -> bool:
        """
        Detect if an image has a white background suitable for product images.
        
        Args:
            image: PIL Image to analyze
            white_threshold: Threshold for considering a pixel "white" (0-1 scale)
            edge_sample_ratio: Ratio of edge pixels to sample for analysis
        
        Returns:
            True if image has a white background, False otherwise
        """
        try:
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            # Sample edge pixels (border regions)
            border_width = max(10, min(width, height) // 20)  # Dynamic border width
            
            # Get edge pixels from all four sides
            edge_pixels = []
            
            # Top and bottom edges
            edge_pixels.extend(img_array[:border_width, :].reshape(-1, 3))
            edge_pixels.extend(img_array[-border_width:, :].reshape(-1, 3))
            
            # Left and right edges
            edge_pixels.extend(img_array[:, :border_width].reshape(-1, 3))
            edge_pixels.extend(img_array[:, -border_width:].reshape(-1, 3))
            
            edge_pixels = np.array(edge_pixels)
            
            # Sample a subset of edge pixels for efficiency
            if len(edge_pixels) > 1000:
                sample_indices = np.random.choice(len(edge_pixels), 1000, replace=False)
                edge_pixels = edge_pixels[sample_indices]
            
            # Convert to 0-1 scale
            edge_pixels_normalized = edge_pixels.astype(np.float32) / 255.0
            
            # Check if pixels are "white" (all RGB values above threshold)
            white_pixels = np.all(edge_pixels_normalized >= white_threshold, axis=1)
            white_percentage = np.mean(white_pixels)
            
            # Also check for near-white pixels (slight gray/off-white backgrounds)
            near_white_threshold = white_threshold - 0.1
            near_white_pixels = np.all(edge_pixels_normalized >= near_white_threshold, axis=1)
            near_white_percentage = np.mean(near_white_pixels)
            
            # Additional check: color uniformity in edges
            color_std = np.std(edge_pixels_normalized, axis=0)
            is_uniform = np.all(color_std < 0.1)  # Low standard deviation indicates uniform color
            
            # Decision logic: high white percentage OR (moderate near-white + uniformity)
            is_white_bg = (white_percentage > 0.8) or (near_white_percentage > 0.9 and is_uniform)
            
            # Log detection details for debugging
            print(f"White background detection - White: {white_percentage:.2f}, Near-white: {near_white_percentage:.2f}, Uniform: {is_uniform}, Result: {is_white_bg}")
            
            return is_white_bg
            
        except Exception as e:
            print(f"Error in white background detection: {e}")
            # If detection fails, assume it's suitable for processing (conservative approach)
            return True
    
    def is_simple_product_image(self, image: Image.Image) -> bool:
        """
        Determine if an image is a simple product image suitable for processing.
        This combines white background detection with additional simplicity checks.
        
        Args:
            image: PIL Image to analyze
        
        Returns:
            True if image is suitable for processing, False otherwise
        """
        try:
            # First check for white background
            has_white_bg = self.has_white_background(image)
            
            if not has_white_bg:
                print("Image rejected: No white background detected")
                return False
            
            # Additional simplicity checks could be added here:
            # - Check for complex scenes (many objects)
            # - Check for text overlay complexity
            # - Check for background patterns
            
            # Convert to array for analysis
            img_array = np.array(image.convert('RGB'))
            
            # Simple complexity check: edge density
            # Simple product images should have clean, defined edges
            gray = np.mean(img_array, axis=2)
            edges = ndimage.sobel(gray)
            edge_density = np.mean(np.abs(edges) > 0.1)
            
            # If edge density is too high, image might be too complex
            if edge_density > 0.3:
                print(f"Image rejected: Too complex (edge density: {edge_density:.2f})")
                return False
            
            print("Image accepted: Simple product image with white background")
            return True
            
        except Exception as e:
            print(f"Error in product image analysis: {e}")
            # If analysis fails, assume it's suitable for processing (conservative approach)
            return True
    
    def scale_to_1080p(self, image: Image.Image) -> Image.Image:
        """Scale image down to 1080p (1920x1080) if larger, maintaining aspect ratio."""
        width, height = image.size
        max_1080p_width = 1920
        max_1080p_height = 1080
        
        # Check if image is already smaller than 1080p
        if width <= max_1080p_width and height <= max_1080p_height:
            return image
        
        # Calculate scale factor to fit within 1080p bounds
        scale_factor = min(max_1080p_width / width, max_1080p_height / height)
        
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        print(f"Scaling from {width}x{height} to {new_width}x{new_height} (1080p max)")
        return image.resize((new_width, new_height), Image.LANCZOS)
    
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
        
        # Ensure dimensions are divisible by 8 (required for SDXL)
        new_width = ((new_width + 7) // 8) * 8
        new_height = ((new_height + 7) // 8) * 8
        
        return new_width, new_height
    
    def create_extension_mask(self, original_img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Create mask for areas that need to be inpainted with protected original area."""
        target_width, target_height = target_size
        orig_width, orig_height = original_img.size
        
        # Create canvas
        mask = Image.new('L', target_size, 255)  # White = inpaint
        
        # Calculate centering position
        x_offset = (target_width - orig_width) // 2
        y_offset = (target_height - orig_height) // 2
        
        # Create a buffer zone around the original image to ensure no AI processing
        buffer_size = 5  # 5 pixel buffer
        protected_x1 = max(0, x_offset - buffer_size)
        protected_y1 = max(0, y_offset - buffer_size)
        protected_x2 = min(target_width, x_offset + orig_width + buffer_size)
        protected_y2 = min(target_height, y_offset + orig_height + buffer_size)
        
        # Black out the protected area (don't inpaint)
        mask.paste(0, (protected_x1, protected_y1, protected_x2, protected_y2))
        
        # Apply feathering only to the outer edges, not the protected area
        feathered_mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
        
        # Restore the hard protected area to ensure original content is never touched
        feathered_mask.paste(0, (protected_x1, protected_y1, protected_x2, protected_y2))
        
        return feathered_mask
    
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
    
    def preserve_original_content(self, ai_result: Image.Image, original_img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Preserve original image content by pasting it back onto the AI-generated result."""
        target_width, target_height = target_size
        orig_width, orig_height = original_img.size
        
        # Calculate centering position
        x_offset = (target_width - orig_width) // 2
        y_offset = (target_height - orig_height) // 2
        
        # Create a copy of the AI result
        final_result = ai_result.copy()
        
        # Paste the original image back to ensure content is preserved
        final_result.paste(original_img, (x_offset, y_offset))
        
        return final_result
    
    def validate_original_preservation(self, final_result: Image.Image, original_img: Image.Image, target_size: Tuple[int, int]) -> bool:
        """Validate that the original image content is perfectly preserved."""
        target_width, target_height = target_size
        orig_width, orig_height = original_img.size
        
        # Calculate centering position
        x_offset = (target_width - orig_width) // 2
        y_offset = (target_height - orig_height) // 2
        
        # Extract the original area from the final result
        extracted_area = final_result.crop((x_offset, y_offset, x_offset + orig_width, y_offset + orig_height))
        
        # Convert both images to numpy arrays for comparison
        original_array = np.array(original_img)
        extracted_array = np.array(extracted_area)
        
        # Check if they are identical
        are_identical = np.array_equal(original_array, extracted_array)
        
        if not are_identical:
            print("WARNING: Original content was modified during processing!")
            # Calculate difference percentage
            diff_pixels = np.sum(original_array != extracted_array)
            total_pixels = original_array.size
            diff_percentage = (diff_pixels / total_pixels) * 100
            print(f"Difference: {diff_percentage:.2f}% of pixels modified")
        
        return are_identical
    
    def create_composite_image(self, original_img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Create a composite image with original content isolated from AI processing."""
        target_width, target_height = target_size
        orig_width, orig_height = original_img.size
        
        # Calculate centering position
        x_offset = (target_width - orig_width) // 2
        y_offset = (target_height - orig_height) // 2
        
        # Create base canvas with neutral background
        canvas = Image.new('RGB', target_size, (128, 128, 128))
        
        # Only add edge extension for context, not the original image
        if x_offset > 0:  # Need to extend horizontally
            # Left edge
            left_edge = original_img.crop((0, 0, min(10, orig_width), orig_height))
            left_edge = left_edge.resize((x_offset, orig_height), Image.LANCZOS)
            canvas.paste(left_edge, (0, y_offset))
            
            # Right edge
            right_edge = original_img.crop((max(0, orig_width-10), 0, orig_width, orig_height))
            right_edge = right_edge.resize((x_offset, orig_height), Image.LANCZOS)
            canvas.paste(right_edge, (x_offset + orig_width, y_offset))
        
        if y_offset > 0:  # Need to extend vertically
            # Top edge
            top_edge = original_img.crop((0, 0, orig_width, min(10, orig_height)))
            top_edge = top_edge.resize((orig_width, y_offset), Image.LANCZOS)
            canvas.paste(top_edge, (x_offset, 0))
            
            # Bottom edge
            bottom_edge = original_img.crop((0, max(0, orig_height-10), orig_width, orig_height))
            bottom_edge = bottom_edge.resize((orig_width, y_offset), Image.LANCZOS)
            canvas.paste(bottom_edge, (x_offset, y_offset + orig_height))
        
        # Do NOT paste the original image here - it will be added after AI processing
        return canvas
    
    def resize_image(self, image_path: str, output_path: str) -> bool:
        """Resize image to 9:16 aspect ratio using AI inpainting."""
        try:
            # Load image
            original_img = Image.open(image_path).convert('RGB')
            orig_width, orig_height = original_img.size
            
            print(f"Original size: {orig_width}x{orig_height}")
            
            # First scale to 1080p if needed
            scaled_img = self.scale_to_1080p(original_img)
            scaled_width, scaled_height = scaled_img.size
            
            # Calculate target dimensions based on scaled image
            target_width, target_height = self.calculate_target_dimensions(scaled_width, scaled_height)
            print(f"Target size: {target_width}x{target_height}")
            
            # Check if already correct ratio
            if scaled_width == target_width and scaled_height == target_height:
                print("Image already has correct aspect ratio")
                scaled_img.save(output_path, quality=95)
                return True
            
            # Create base canvas and mask using scaled image
            base_canvas = self.create_composite_image(scaled_img, (target_width, target_height))
            mask = self.create_extension_mask(scaled_img, (target_width, target_height))
            
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
            
            # Run inpainting with very low strength to only generate extensions
            result = self.inpaint_pipeline(
                prompt=prompt,
                image=base_resized,
                mask_image=mask_resized,
                num_inference_steps=25,  # Slightly increased for better quality
                strength=0.9,  # Higher strength is OK since original area is protected by mask
                guidance_scale=8.0,  # Slightly higher for better adherence to prompt
                height=process_height,
                width=process_width
            ).images[0]
            
            # Resize back to target size if needed
            if scale_factor < 1:
                result = result.resize((target_width, target_height), Image.LANCZOS)
            
            # Preserve original content by pasting it back onto the AI result
            final_result = self.preserve_original_content(result, scaled_img, (target_width, target_height))
            
            # Validate that original content is preserved
            is_preserved = self.validate_original_preservation(final_result, scaled_img, (target_width, target_height))
            if is_preserved:
                print("✓ Original content successfully preserved")
            
            # Save result
            final_result.save(output_path, quality=95)
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