#!/usr/bin/env python3
"""
AI-powered image resizer to 9:16 aspect ratio with realistic extension.
Uses FluxFill (Flux 1.0 Inpainting) for high-quality seamless results.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image, ImageFilter
from diffusers import AutoPipelineForInpainting
import numpy as np
from scipy import ndimage


class ImageResizer:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.target_ratio = 9 / 16  # 9:16 aspect ratio

        # Load models
        print("Loading AI models...")
        print("Initializing FluxFill inpainting pipeline...")
        hf_token = os.getenv("HF_TOKEN")
        self.inpaint_pipeline = AutoPipelineForInpainting.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            token=hf_token
        ).to(device)

        # Enable memory optimization for A100
        self.inpaint_pipeline.enable_attention_slicing()

        print("Models loaded successfully!")
    
    def has_white_background(self, image: Image.Image, white_threshold: float = 0.65, edge_sample_ratio: float = 0.1) -> bool:
        """
        Detect if an image has a white background suitable for product images.
        Uses multiple sampling strategies to handle large products.
        
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
            
            # Strategy 1: Sample from multiple border regions (thin and thick)
            border_samples = []
            
            # Thin border (for smaller products)
            thin_border = max(5, min(width, height) // 40)
            border_samples.extend(img_array[:thin_border, :].reshape(-1, 3))  # Top
            border_samples.extend(img_array[-thin_border:, :].reshape(-1, 3))  # Bottom
            border_samples.extend(img_array[:, :thin_border].reshape(-1, 3))  # Left
            border_samples.extend(img_array[:, -thin_border:].reshape(-1, 3))  # Right
            
            # Thick border (for larger products)
            thick_border = max(20, min(width, height) // 15)
            border_samples.extend(img_array[:thick_border, :].reshape(-1, 3))  # Top
            border_samples.extend(img_array[-thick_border:, :].reshape(-1, 3))  # Bottom
            border_samples.extend(img_array[:, :thick_border].reshape(-1, 3))  # Left
            border_samples.extend(img_array[:, -thick_border:].reshape(-1, 3))  # Right
            
            # Strategy 2: Sample from corners (always background)
            corner_size = min(50, min(width, height) // 8)
            corners = [
                img_array[:corner_size, :corner_size],  # Top-left
                img_array[:corner_size, -corner_size:],  # Top-right
                img_array[-corner_size:, :corner_size],  # Bottom-left
                img_array[-corner_size:, -corner_size:]   # Bottom-right
            ]
            for corner in corners:
                border_samples.extend(corner.reshape(-1, 3))
            
            # Strategy 3: Sample from center strips (for very large products)
            center_strip_width = max(10, min(width, height) // 30)
            # Vertical center strips (left and right of image center)
            center_x = width // 2
            if center_x - center_strip_width > 0:
                border_samples.extend(img_array[:, :center_strip_width].reshape(-1, 3))  # Far left
                border_samples.extend(img_array[:, -center_strip_width:].reshape(-1, 3))  # Far right
            
            # Horizontal center strips (top and bottom of image center)  
            center_y = height // 2
            if center_y - center_strip_width > 0:
                border_samples.extend(img_array[:center_strip_width, :].reshape(-1, 3))  # Far top
                border_samples.extend(img_array[-center_strip_width:, :].reshape(-1, 3))  # Far bottom
            
            border_pixels = np.array(border_samples)
            
            # Sample a subset for efficiency
            if len(border_pixels) > 2000:
                sample_indices = np.random.choice(len(border_pixels), 2000, replace=False)
                border_pixels = border_pixels[sample_indices]
            
            # Convert to 0-1 scale
            border_pixels_normalized = border_pixels.astype(np.float32) / 255.0
            
            # Multiple white detection strategies
            # Strategy A: Pure white pixels
            pure_white_pixels = np.all(border_pixels_normalized >= white_threshold, axis=1)
            pure_white_percentage = np.mean(pure_white_pixels)
            
            # Strategy B: Near-white pixels (off-white, light gray backgrounds)
            near_white_threshold = white_threshold - 0.15
            near_white_pixels = np.all(border_pixels_normalized >= near_white_threshold, axis=1)
            near_white_percentage = np.mean(near_white_pixels)
            
            # Strategy C: Brightness-based detection (very bright backgrounds)
            brightness = np.mean(border_pixels_normalized, axis=1)
            bright_pixels = brightness >= (white_threshold - 0.1)
            bright_percentage = np.mean(bright_pixels)
            
            # Strategy D: Color uniformity check
            color_std = np.std(border_pixels_normalized, axis=0)
            is_uniform = np.all(color_std < 0.15)  # More lenient uniformity
            
            # Strategy E: Dominant color analysis
            # Check if the most common color in borders is white-ish
            from collections import Counter
            # Quantize colors to reduce noise
            quantized_colors = (border_pixels_normalized * 4).astype(int)  # 0-4 scale
            color_counts = Counter([tuple(color) for color in quantized_colors])
            if color_counts:
                most_common_color = color_counts.most_common(1)[0]
                dominant_color_ratio = most_common_color[1] / len(quantized_colors)
                dominant_color_brightness = np.mean(most_common_color[0]) / 4.0  # Back to 0-1 scale
                is_dominant_white = dominant_color_brightness >= (white_threshold - 0.1) and dominant_color_ratio > 0.3
            else:
                is_dominant_white = False
            
            # Comprehensive decision logic (more lenient for large products)
            decision_factors = [
                pure_white_percentage > 0.6,  # Lowered from 0.8
                near_white_percentage > 0.75,  # Lowered from 0.9
                bright_percentage > 0.7 and is_uniform,
                is_dominant_white,
                pure_white_percentage > 0.4 and is_uniform,  # New: moderate white + uniformity
            ]
            
            is_white_bg = any(decision_factors)
            
            # Log detection details for debugging
            print(f"White BG Detection - Pure: {pure_white_percentage:.2f}, Near: {near_white_percentage:.2f}, "
                  f"Bright: {bright_percentage:.2f}, Uniform: {is_uniform}, Dominant: {is_dominant_white}, "
                  f"Result: {is_white_bg}")
            
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
        
        # Ensure dimensions are divisible by 8 (required for Flux)
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
            
            # Resize for processing (FluxFill works best at native resolution)
            # Keep original resolution for better quality with FluxFill
            process_width = target_width
            process_height = target_height
            base_resized = base_canvas
            mask_resized = mask

            print("Running AI inpainting with FluxFill...")

            # Run inpainting with FluxFill for high-quality results
            # FluxFill uses a different API - it doesn't have strength parameter
            result = self.inpaint_pipeline(
                prompt=prompt,
                image=base_resized,
                mask_image=mask_resized,
                num_inference_steps=50,  # FluxFill benefits from more steps
                guidance_scale=30.0,  # FluxFill uses higher guidance scale
                height=process_height,
                width=process_width
            ).images[0]
            
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