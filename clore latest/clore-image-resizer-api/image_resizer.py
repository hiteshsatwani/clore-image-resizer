#!/usr/bin/env python3
"""
AI-powered image resizer to 9:16 aspect ratio with ControlNet guidance.
Uses FluxControlNetInpaintingPipeline for intelligent, edge-aware extension.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from scipy import ndimage

from diffusers.models import FluxTransformer2DModel

try:
    from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
    from controlnet_flux import FluxControlNetModel
    print("✓ ControlNet pipeline imported successfully")
except Exception as e:
    print(f"⚠️  ControlNet import warning: {e}")


class ImageResizer:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.target_ratio = 9 / 16

        print("Loading AI models...")
        print("Initializing FluxControlNetInpaintingPipeline...")

        try:
            hf_token = os.getenv("HF_TOKEN")

            # Load base transformer
            transformer = FluxTransformer2DModel.from_pretrained(
                "black-forest-labs/FLUX.1-Fill-dev",
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
                token=hf_token
            )

            # Load ControlNet
            controlnet = FluxControlNetModel.from_pretrained(
                "black-forest-labs/FLUX.1-Fill-dev",
                subfolder="controlnet",
                torch_dtype=torch.bfloat16,
                token=hf_token
            )

            # Load full pipeline
            self.pipe = FluxControlNetInpaintingPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Fill-dev",
                transformer=transformer,
                controlnet=controlnet,
                torch_dtype=torch.bfloat16,
                token=hf_token
            ).to(device)

            self.pipe.enable_attention_slicing()

            print("✓ ControlNet pipeline loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading ControlNet pipeline: {e}")
            print("Falling back to standard FluxFillPipeline...")
            from diffusers import FluxFillPipeline

            hf_token = os.getenv("HF_TOKEN")
            self.pipe = FluxFillPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Fill-dev",
                torch_dtype=torch.bfloat16,
                token=hf_token
            ).to(device)
            self.pipe.enable_attention_slicing()
            self.use_controlnet = False
            return

        self.use_controlnet = True

    def scale_to_1080p(self, image: Image.Image) -> Image.Image:
        """Scale image down to 1080p if larger."""
        width, height = image.size
        max_1080p_width = 1920
        max_1080p_height = 1080

        if width <= max_1080p_width and height <= max_1080p_height:
            return image

        scale_factor = min(max_1080p_width / width, max_1080p_height / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        print(f"Scaling from {width}x{height} to {new_width}x{new_height} (1080p max)")
        return image.resize((new_width, new_height), Image.LANCZOS)

    def calculate_target_dimensions(self, width: int, height: int) -> Tuple[int, int]:
        """Calculate target dimensions for 9:16 aspect ratio."""
        current_ratio = width / height

        if current_ratio > self.target_ratio:
            new_width = width
            new_height = int(width / self.target_ratio)
        else:
            new_height = height
            new_width = int(height * self.target_ratio)

        new_width = ((new_width + 7) // 8) * 8
        new_height = ((new_height + 7) // 8) * 8

        return new_width, new_height

    def create_extension_mask(self, original_img: Image.Image, target_size: Tuple[int, int], overlap_percentage: int = 10) -> Image.Image:
        """Create mask for areas that need to be inpainted with configurable overlap."""
        target_width, target_height = target_size
        orig_width, orig_height = original_img.size

        mask = Image.new('L', target_size, 255)

        x_offset = (target_width - orig_width) // 2
        y_offset = (target_height - orig_height) // 2

        overlap_x = max(1, int(orig_width * (overlap_percentage / 100)))
        overlap_y = max(1, int(orig_height * (overlap_percentage / 100)))

        protected_x1 = x_offset + overlap_x
        protected_y1 = y_offset + overlap_y
        protected_x2 = x_offset + orig_width - overlap_x
        protected_y2 = y_offset + orig_height - overlap_y

        mask.paste(0, (protected_x1, protected_y1, protected_x2, protected_y2))
        feathered_mask = mask.filter(ImageFilter.GaussianBlur(radius=16))

        return feathered_mask

    def create_base_canvas(self, original_img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Create base canvas with original image centered on white background."""
        target_width, target_height = target_size
        orig_width, orig_height = original_img.size

        canvas = Image.new('RGB', target_size, (255, 255, 255))
        x_offset = (target_width - orig_width) // 2
        y_offset = (target_height - orig_height) // 2

        canvas.paste(original_img, (x_offset, y_offset))

        return canvas

    def generate_inpaint_prompt(self) -> str:
        """Generate high-quality contextual prompt for inpainting."""
        return (
            "professional product photography, seamless extension, "
            "photorealistic, high quality, sharp details, "
            "consistent lighting, natural colors, smooth transition, "
            "perfect blend, professional background, no visible seams"
        )

    def resize_image(self, image_path: str, output_path: str) -> bool:
        """Resize image to 9:16 aspect ratio using AI inpainting with ControlNet."""
        try:
            original_img = Image.open(image_path).convert('RGB')
            orig_width, orig_height = original_img.size

            print(f"Original size: {orig_width}x{orig_height}")

            scaled_img = self.scale_to_1080p(original_img)
            scaled_width, scaled_height = scaled_img.size

            target_width, target_height = self.calculate_target_dimensions(scaled_width, scaled_height)
            print(f"Target size: {target_width}x{target_height}")

            if scaled_width == target_width and scaled_height == target_height:
                print("Image already has correct aspect ratio")
                scaled_img.save(output_path, quality=95)
                return True

            base_canvas = self.create_base_canvas(scaled_img, (target_width, target_height))
            mask = self.create_extension_mask(scaled_img, (target_width, target_height), overlap_percentage=10)

            prompt = self.generate_inpaint_prompt()

            print("Running AI inpainting with ControlNet guidance...")

            if self.use_controlnet:
                # Use ControlNet pipeline with dual guidance
                result = self.pipe(
                    prompt=prompt,
                    prompt_2=prompt,
                    height=target_height,
                    width=target_width,
                    num_inference_steps=28,
                    guidance_scale=7.0,
                    true_guidance_scale=3.5,
                    control_image=base_canvas,
                    control_mask=mask,
                    controlnet_conditioning_scale=1.0,
                    max_sequence_length=512,
                    generator=torch.Generator(device=self.device).manual_seed(42),
                ).images[0]
            else:
                # Fallback to standard FluxFill
                result = self.pipe(
                    prompt=prompt,
                    image=base_canvas,
                    mask_image=mask,
                    num_inference_steps=8,
                    guidance_scale=30.0,
                    height=target_height,
                    width=target_width,
                    max_sequence_length=512,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                ).images[0]

                # RGBA compositing for smooth blending
                result_rgba = result.convert('RGBA')
                base_rgba = base_canvas.convert('RGBA')
                result_rgba.putalpha(mask)
                base_rgba.paste(result_rgba, (0, 0), mask)
                result = base_rgba.convert('RGB')

            print("✓ Inpainting completed with edge-aware guidance")

            result.save(output_path, quality=95)
            print(f"Saved resized image to: {output_path}")

            return True

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description="AI-powered image resizer to 9:16 aspect ratio with ControlNet")
    parser.add_argument("input", help="Input image path or directory")
    parser.add_argument("-o", "--output", help="Output path or directory")
    parser.add_argument("-d", "--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("-b", "--batch", action="store_true", help="Batch process directory")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    resizer = ImageResizer(device=args.device)

    input_path = Path(args.input)

    if input_path.is_file():
        output_path = args.output if args.output else f"{input_path.stem}_resized{input_path.suffix}"

        start_time = time.time()
        success = resizer.resize_image(str(input_path), output_path)
        end_time = time.time()

        if success and not args.quiet:
            print(f"Processing completed in {end_time - start_time:.2f} seconds")

    elif input_path.is_dir() and args.batch:
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
