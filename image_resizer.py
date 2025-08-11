#!/usr/bin/env python3
"""
Advanced AI-powered image resizer to 9:16 aspect ratio.
Multi-stage pipeline with scene understanding and adaptive processing.
Optimized for g5.2xlarge instances with 10-second processing target.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageStat
from diffusers import StableDiffusionXLInpaintPipeline, AutoPipelineForInpainting
from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    CLIPProcessor, CLIPModel,
    pipeline as hf_pipeline
)
import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage import segmentation, color, measure
from sklearn.cluster import KMeans
import open_clip


class BackgroundType(Enum):
    """Background classification types for strategy selection."""
    STUDIO_SOLID = "studio_solid"
    STUDIO_GRADIENT = "studio_gradient" 
    TEXTURED_WALL = "textured_wall"
    COMPLEX_SCENE = "complex_scene"
    LIFESTYLE_NATURAL = "lifestyle_natural"
    PRODUCT_ONLY = "product_only"


@dataclass
class SceneAnalysis:
    """Comprehensive scene understanding results."""
    caption: str
    background_type: BackgroundType
    dominant_colors: List[Tuple[int, int, int]]
    has_person: bool
    confidence: float
    depth_map: Optional[np.ndarray] = None
    subject_mask: Optional[np.ndarray] = None
    complexity_score: float = 0.0


@dataclass  
class ProcessingStrategy:
    """Adaptive processing strategy based on scene analysis."""
    extension_method: str
    inpaint_strength: float
    prompt_template: str
    use_depth: bool
    edge_feather: int
    inference_steps: int
    guidance_scale: float


class AdvancedSceneAnalyzer:
    """Multi-model scene understanding system."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        print("🔍 Loading scene analysis models...")
        
        try:
            # BLIP-2 for detailed image captioning
            self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("✓ BLIP-2 captioning model loaded")
        except Exception as e:
            print(f"⚠️  BLIP-2 failed to load: {e}")
            self.blip_model = None
            self.blip_processor = None
        
        try:
            # CLIP for classification
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai'
            )
            self.clip_model = self.clip_model.to(device)
            print("✓ CLIP classification model loaded")
        except Exception as e:
            print(f"⚠️  CLIP failed to load: {e}")
            self.clip_model = None
        
        try:
            # Lightweight depth estimation
            self.depth_estimator = hf_pipeline(
                "depth-estimation",
                model="Intel/dpt-large", 
                device=device
            )
            print("✓ Depth estimation model loaded")
        except Exception as e:
            print(f"⚠️  Depth estimator failed to load: {e}")
            self.depth_estimator = None
        
        print("🔍 Scene analysis models ready!")
    
    def analyze_scene(self, image: Image.Image) -> SceneAnalysis:
        """Comprehensive scene analysis with fallback strategies."""
        print("🔍 Analyzing scene...")
        
        # Generate caption (with fallback)
        caption = self._generate_caption(image)
        
        # Classify background type  
        background_type, confidence = self._classify_background(image, caption)
        
        # Extract dominant colors
        dominant_colors = self._extract_background_colors(image)
        
        # Detect person presence
        has_person = self._detect_person(caption)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity(image)
        
        # Generate depth map for complex scenes
        depth_map = None
        if background_type in [BackgroundType.COMPLEX_SCENE, BackgroundType.LIFESTYLE_NATURAL]:
            depth_map = self._estimate_depth(image)
        
        # Generate subject mask
        subject_mask = self._generate_subject_mask(image)
        
        analysis = SceneAnalysis(
            caption=caption,
            background_type=background_type,
            dominant_colors=dominant_colors,
            has_person=has_person,
            confidence=confidence,
            depth_map=depth_map,
            subject_mask=subject_mask,
            complexity_score=complexity_score
        )
        
        print(f"📝 Caption: {caption}")
        print(f"🎨 Background: {background_type.value} (confidence: {confidence:.2f})")
        print(f"👤 Has person: {has_person}")
        print(f"📊 Complexity: {complexity_score:.2f}")
        
        return analysis
    
    def _generate_caption(self, image: Image.Image) -> str:
        """Generate image caption with fallback."""
        if self.blip_model is None:
            return self._fallback_caption_analysis(image)
        
        try:
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device, torch.float16)
            generated_ids = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return caption.strip()
        except Exception as e:
            print(f"⚠️  BLIP-2 caption failed: {e}")
            return self._fallback_caption_analysis(image)
    
    def _fallback_caption_analysis(self, image: Image.Image) -> str:
        """Fallback caption based on basic image analysis."""
        # Simple heuristics based on image properties
        stat = ImageStat.Stat(image)
        mean_brightness = sum(stat.mean) / 3
        
        if mean_brightness > 200:
            return "product photo with bright white background"
        elif mean_brightness < 80:
            return "product photo with dark background"
        else:
            return "product photo with neutral background"
    
    def _classify_background(self, image: Image.Image, caption: str) -> Tuple[BackgroundType, float]:
        """Classify background type using CLIP and heuristics."""
        if self.clip_model is None:
            return self._fallback_background_classification(image, caption)
        
        try:
            background_prompts = [
                "a product photo with solid white studio background",
                "a product photo with gradient studio background", 
                "a product photo with textured wall background",
                "a lifestyle photo with complex natural scene",
                "a person wearing clothing in natural environment",
                "a isolated product on transparent background"
            ]
            
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            text_inputs = open_clip.tokenize(background_prompts).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_inputs)
                
                similarities = F.cosine_similarity(image_features, text_features)
                best_idx = similarities.argmax().item()
                confidence = similarities[best_idx].item()
            
            background_types = [
                BackgroundType.STUDIO_SOLID,
                BackgroundType.STUDIO_GRADIENT,
                BackgroundType.TEXTURED_WALL,
                BackgroundType.COMPLEX_SCENE,
                BackgroundType.LIFESTYLE_NATURAL,
                BackgroundType.PRODUCT_ONLY
            ]
            
            return background_types[best_idx], confidence
            
        except Exception as e:
            print(f"⚠️  CLIP classification failed: {e}")
            return self._fallback_background_classification(image, caption)
    
    def _fallback_background_classification(self, image: Image.Image, caption: str) -> Tuple[BackgroundType, float]:
        """Fallback background classification using simple heuristics."""
        caption_lower = caption.lower()
        
        # Simple keyword-based classification
        if 'person' in caption_lower or 'wearing' in caption_lower:
            return BackgroundType.LIFESTYLE_NATURAL, 0.7
        elif 'white' in caption_lower:
            return BackgroundType.STUDIO_SOLID, 0.8
        elif 'gradient' in caption_lower:
            return BackgroundType.STUDIO_GRADIENT, 0.7
        else:
            return BackgroundType.COMPLEX_SCENE, 0.6
    
    def _extract_background_colors(self, image: Image.Image) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image edges."""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Sample from edges (likely background)
        edge_pixels = []
        edge_size = min(30, min(w, h) // 6)
        
        # Corners and edges
        edge_pixels.extend(img_array[:edge_size, :edge_size].reshape(-1, 3))
        edge_pixels.extend(img_array[:edge_size, -edge_size:].reshape(-1, 3))
        edge_pixels.extend(img_array[-edge_size:, :edge_size].reshape(-1, 3))
        edge_pixels.extend(img_array[-edge_size:, -edge_size:].reshape(-1, 3))
        
        edge_pixels = np.array(edge_pixels)
        
        try:
            # Use KMeans to find dominant colors
            n_colors = min(3, len(edge_pixels) // 10)
            if n_colors > 0:
                kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
                kmeans.fit(edge_pixels)
                colors = kmeans.cluster_centers_.astype(int)
                return [tuple(color) for color in colors]
        except:
            pass
        
        # Fallback to mean color
        mean_color = edge_pixels.mean(axis=0).astype(int)
        return [tuple(mean_color)]
    
    def _detect_person(self, caption: str) -> bool:
        """Detect person presence from caption."""
        person_keywords = ['person', 'people', 'man', 'woman', 'model', 'wearing', 'portrait', 'human']
        caption_lower = caption.lower()
        return any(keyword in caption_lower for keyword in person_keywords)
    
    def _calculate_complexity(self, image: Image.Image) -> float:
        """Calculate scene complexity score."""
        img_array = np.array(image.convert('L'))
        
        # Edge detection for texture complexity
        edges = cv2.Canny(img_array, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Color variance
        color_array = np.array(image)
        color_variance = np.var(color_array) / 255.0
        
        # Combine metrics
        complexity = (edge_density * 0.7 + color_variance * 0.3)
        return min(1.0, complexity)
    
    def _estimate_depth(self, image: Image.Image) -> Optional[np.ndarray]:
        """Estimate depth map."""
        if self.depth_estimator is None:
            return None
        
        try:
            depth = self.depth_estimator(image)
            return np.array(depth['depth'])
        except Exception as e:
            print(f"⚠️  Depth estimation failed: {e}")
            return None
    
    def _generate_subject_mask(self, image: Image.Image) -> Optional[np.ndarray]:
        """Generate subject mask using simple techniques."""
        try:
            # Convert to HSV for better segmentation
            img_array = np.array(image)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Simple center-weighted mask as fallback
            h, w = img_array.shape[:2]
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            
            # Create elliptical mask
            mask = ((x - center_x) / (w * 0.35))**2 + ((y - center_y) / (h * 0.45))**2 <= 1
            return mask.astype(np.uint8) * 255
            
        except Exception as e:
            print(f"⚠️  Subject mask generation failed: {e}")
            return None


class ImageResizer:
    """Advanced multi-stage image resizer optimized for product photography."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.target_ratio = 9 / 16
        
        print("🚀 Initializing Advanced Image Resizer...")
        
        # Stage 1: Scene Analysis
        self.scene_analyzer = AdvancedSceneAnalyzer(device)
        
        # Stage 3: Inpainting Models
        print("🎨 Loading inpainting models...")
        
        try:
            # Primary SDXL Inpainting
            self.primary_inpainter = StableDiffusionXLInpaintPipeline.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            ).to(device)
            
            # Memory optimizations
            self.primary_inpainter.enable_model_cpu_offload()
            self.primary_inpainter.enable_vae_slicing()
            print("✓ Primary SDXL inpainter loaded")
            
        except Exception as e:
            print(f"❌ Failed to load primary inpainter: {e}")
            raise
        
        # Try to load fallback inpainter
        try:
            self.fallback_inpainter = AutoPipelineForInpainting.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16
            ).to(device)
            print("✓ Fallback inpainter loaded")
        except Exception as e:
            print(f"⚠️  Fallback inpainter not available: {e}")
            self.fallback_inpainter = None
        
        print("🚀 Advanced Image Resizer ready!")
    
    def get_processing_strategy(self, analysis: SceneAnalysis) -> ProcessingStrategy:
        """Select optimal processing strategy based on scene analysis."""
        strategies = {
            BackgroundType.STUDIO_SOLID: ProcessingStrategy(
                extension_method="gradient_fill",
                inpaint_strength=0.6,
                prompt_template="seamless solid {color} studio background, professional product photography, clean, minimal",
                use_depth=False,
                edge_feather=3,
                inference_steps=15,
                guidance_scale=6.0
            ),
            BackgroundType.STUDIO_GRADIENT: ProcessingStrategy(
                extension_method="gradient_extend", 
                inpaint_strength=0.7,
                prompt_template="smooth gradient studio background, professional lighting, seamless transition",
                use_depth=False,
                edge_feather=5,
                inference_steps=18,
                guidance_scale=6.5
            ),
            BackgroundType.TEXTURED_WALL: ProcessingStrategy(
                extension_method="texture_aware",
                inpaint_strength=0.75,
                prompt_template="textured wall background, consistent pattern and lighting, seamless extension",
                use_depth=False,
                edge_feather=4,
                inference_steps=20,
                guidance_scale=7.0
            ),
            BackgroundType.COMPLEX_SCENE: ProcessingStrategy(
                extension_method="ai_generate",
                inpaint_strength=0.85,
                prompt_template="{caption}, photorealistic scene extension, natural lighting, high detail",
                use_depth=True,
                edge_feather=6,
                inference_steps=25,
                guidance_scale=7.5
            ),
            BackgroundType.LIFESTYLE_NATURAL: ProcessingStrategy(
                extension_method="ai_generate",
                inpaint_strength=0.8,
                prompt_template="lifestyle photography, {caption}, natural environment, realistic extension",
                use_depth=True,
                edge_feather=5,
                inference_steps=22,
                guidance_scale=7.0
            ),
            BackgroundType.PRODUCT_ONLY: ProcessingStrategy(
                extension_method="edge_extend",
                inpaint_strength=0.5,
                prompt_template="clean product background extension, minimal, professional",
                use_depth=False,
                edge_feather=2,
                inference_steps=12,
                guidance_scale=5.5
            )
        }
        
        return strategies.get(analysis.background_type, strategies[BackgroundType.COMPLEX_SCENE])
    
    def calculate_target_dimensions(self, width: int, height: int) -> Tuple[int, int]:
        """Calculate target dimensions for 9:16 aspect ratio."""
        current_ratio = width / height
        
        if current_ratio > self.target_ratio:
            new_width = width
            new_height = int(width / self.target_ratio)
        else:
            new_height = height  
            new_width = int(height * self.target_ratio)
        
        # Ensure dimensions are divisible by 8 for AI models
        new_width = ((new_width + 7) // 8) * 8
        new_height = ((new_height + 7) // 8) * 8
        
        return new_width, new_height
    
    def create_adaptive_canvas(self, image: Image.Image, target_size: Tuple[int, int], 
                              strategy: ProcessingStrategy, analysis: SceneAnalysis) -> Image.Image:
        """Create intelligent canvas based on background analysis."""
        target_width, target_height = target_size
        orig_width, orig_height = image.size
        
        x_offset = (target_width - orig_width) // 2
        y_offset = (target_height - orig_height) // 2
        
        if strategy.extension_method == "gradient_fill":
            canvas = self._create_gradient_canvas(target_size, analysis.dominant_colors)
        elif strategy.extension_method == "edge_extend":
            canvas = self._create_edge_extended_canvas(image, target_size)
        else:
            # AI-based methods need intelligent context
            canvas = self._create_context_aware_canvas(image, target_size, analysis)
        
        return canvas
    
    def _create_gradient_canvas(self, target_size: Tuple[int, int], 
                               colors: List[Tuple[int, int, int]]) -> Image.Image:
        """Create gradient background for solid studio backgrounds."""
        target_width, target_height = target_size
        primary_color = colors[0] if colors else (255, 255, 255)
        
        if len(colors) > 1:
            # Subtle gradient between dominant colors
            gradient = np.linspace(0, 1, target_height)[:, None]
            color1 = np.array(colors[0])
            color2 = np.array(colors[1]) 
            
            gradient_array = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            for i in range(target_height):
                blend_color = color1 * (1 - gradient[i]) + color2 * gradient[i]
                gradient_array[i, :] = blend_color.astype(np.uint8)
            
            return Image.fromarray(gradient_array)
        else:
            return Image.new('RGB', target_size, primary_color)
    
    def _create_edge_extended_canvas(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Create canvas with intelligent edge extension."""
        target_width, target_height = target_size
        orig_width, orig_height = image.size
        
        # Use dominant color as base
        stat = ImageStat.Stat(image)
        mean_color = tuple(int(c) for c in stat.mean)
        canvas = Image.new('RGB', target_size, mean_color)
        
        x_offset = (target_width - orig_width) // 2
        y_offset = (target_height - orig_height) // 2
        
        # Smart edge extension with larger samples
        if x_offset > 0:
            edge_width = min(30, orig_width // 3)
            
            # Left edge with blur for seamless transition
            left_edge = image.crop((0, 0, edge_width, orig_height))
            left_extended = left_edge.resize((x_offset, orig_height), Image.LANCZOS)
            left_extended = left_extended.filter(ImageFilter.GaussianBlur(radius=1))
            canvas.paste(left_extended, (0, y_offset))
            
            # Right edge
            right_edge = image.crop((orig_width - edge_width, 0, orig_width, orig_height))
            right_extended = right_edge.resize((x_offset, orig_height), Image.LANCZOS)
            right_extended = right_extended.filter(ImageFilter.GaussianBlur(radius=1))
            canvas.paste(right_extended, (x_offset + orig_width, y_offset))
        
        if y_offset > 0:
            edge_height = min(30, orig_height // 3)
            
            # Top edge
            top_edge = image.crop((0, 0, orig_width, edge_height))
            top_extended = top_edge.resize((orig_width, y_offset), Image.LANCZOS)
            top_extended = top_extended.filter(ImageFilter.GaussianBlur(radius=1))
            canvas.paste(top_extended, (x_offset, 0))
            
            # Bottom edge
            bottom_edge = image.crop((0, orig_height - edge_height, orig_width, orig_height))
            bottom_extended = bottom_edge.resize((orig_width, y_offset), Image.LANCZOS)
            bottom_extended = bottom_extended.filter(ImageFilter.GaussianBlur(radius=1))
            canvas.paste(bottom_extended, (x_offset, y_offset + orig_height))
        
        return canvas
    
    def _create_context_aware_canvas(self, image: Image.Image, target_size: Tuple[int, int], 
                                    analysis: SceneAnalysis) -> Image.Image:
        """Create context-aware canvas for AI processing."""
        primary_color = analysis.dominant_colors[0] if analysis.dominant_colors else (128, 128, 128)
        canvas = Image.new('RGB', target_size, primary_color)
        
        # Add subtle noise for better AI context
        canvas_array = np.array(canvas, dtype=np.float32)
        noise = np.random.normal(0, 3, canvas_array.shape)
        canvas_array = np.clip(canvas_array + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(canvas_array)
    
    def create_intelligent_mask(self, image: Image.Image, target_size: Tuple[int, int],
                               analysis: SceneAnalysis, feather_radius: int = 4) -> Image.Image:
        """Create intelligent mask using subject segmentation."""
        target_width, target_height = target_size
        orig_width, orig_height = image.size
        
        # Base mask
        mask = Image.new('L', target_size, 255)  # White = inpaint
        
        x_offset = (target_width - orig_width) // 2
        y_offset = (target_height - orig_height) // 2
        
        if analysis.subject_mask is not None:
            # Use subject mask for precise protection
            subject_mask = Image.fromarray(analysis.subject_mask).convert('L')
            subject_mask = subject_mask.resize((orig_width, orig_height), Image.LANCZOS)
            
            # Create protection zone
            protection_mask = Image.new('L', target_size, 0)
            protection_mask.paste(subject_mask, (x_offset, y_offset))
            
            # Apply protection
            mask_array = np.array(mask)
            protection_array = np.array(protection_mask)
            mask_array[protection_array > 128] = 0
            
            mask = Image.fromarray(mask_array)
        else:
            # Fallback protection with buffer
            buffer_size = 10
            protected_x1 = max(0, x_offset - buffer_size)
            protected_y1 = max(0, y_offset - buffer_size)
            protected_x2 = min(target_width, x_offset + orig_width + buffer_size)
            protected_y2 = min(target_height, y_offset + orig_height + buffer_size)
            
            mask.paste(0, (protected_x1, protected_y1, protected_x2, protected_y2))
        
        # Apply feathering
        if feather_radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
            
            # Restore hard protection in center
            center_protection = Image.new('L', target_size, 255)
            center_protection.paste(0, (x_offset + 5, y_offset + 5, x_offset + orig_width - 5, y_offset + orig_height - 5))
            
            mask_array = np.array(mask)
            center_array = np.array(center_protection)
            mask_array = np.minimum(mask_array, center_array)
            mask = Image.fromarray(mask_array)
        
        return mask
    
    def generate_dynamic_prompt(self, strategy: ProcessingStrategy, analysis: SceneAnalysis) -> str:
        """Generate contextual prompt based on scene understanding."""
        prompt = strategy.prompt_template
        
        # Replace template variables
        if "{caption}" in prompt:
            # Clean up caption for prompting
            clean_caption = analysis.caption.lower().replace("a photo of", "").strip()
            prompt = prompt.replace("{caption}", clean_caption)
        
        if "{color}" in prompt and analysis.dominant_colors:
            color_name = self._get_color_name(analysis.dominant_colors[0])
            prompt = prompt.replace("{color}", color_name)
        
        # Add quality enhancers
        quality_terms = "high quality, professional, detailed, photorealistic, consistent lighting"
        final_prompt = f"{prompt}, {quality_terms}"
        
        return final_prompt
    
    def _get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to approximate color name."""
        r, g, b = rgb
        
        if r > 240 and g > 240 and b > 240:
            return "white"
        elif r < 30 and g < 30 and b < 30:
            return "black"  
        elif r > 200 and g > 200 and b < 80:
            return "yellow"
        elif r > g + 30 and r > b + 30:
            return "red"
        elif g > r + 30 and g > b + 30:
            return "green"
        elif b > r + 30 and b > g + 30:
            return "blue"
        elif (r + g + b) / 3 < 100:
            return "dark"
        else:
            return "neutral"
    
    def run_optimized_inpainting(self, canvas: Image.Image, mask: Image.Image,
                                prompt: str, strategy: ProcessingStrategy) -> Image.Image:
        """Run inpainting with optimized settings."""
        print(f"🎨 Running inpainting: {strategy.extension_method}")
        print(f"📝 Prompt: {prompt[:80]}...")
        
        try:
            # Optimize processing size for g5.2xlarge
            max_dimension = 768  # Reduced for speed
            scale_factor = min(1.0, max_dimension / max(canvas.width, canvas.height))
            
            if scale_factor < 1.0:
                process_width = int(canvas.width * scale_factor)
                process_height = int(canvas.height * scale_factor)
                process_canvas = canvas.resize((process_width, process_height), Image.LANCZOS)
                process_mask = mask.resize((process_width, process_height), Image.LANCZOS)
            else:
                process_canvas = canvas
                process_mask = mask
                process_width, process_height = canvas.width, canvas.height
            
            # Run primary inpainting
            result = self.primary_inpainter(
                prompt=prompt,
                image=process_canvas,
                mask_image=process_mask,
                num_inference_steps=strategy.inference_steps,
                strength=strategy.inpaint_strength,
                guidance_scale=strategy.guidance_scale,
                height=process_height,
                width=process_width
            ).images[0]
            
            # Scale back if needed
            if scale_factor < 1.0:
                result = result.resize((canvas.width, canvas.height), Image.LANCZOS)
            
            return result
            
        except Exception as e:
            print(f"❌ Primary inpainting failed: {e}")
            
            # Try fallback if available
            if self.fallback_inpainter:
                print("🔄 Trying fallback inpainter...")
                try:
                    result = self.fallback_inpainter(
                        prompt=prompt,
                        image=canvas,
                        mask_image=mask,
                        num_inference_steps=15,
                        height=canvas.height,
                        width=canvas.width
                    ).images[0]
                    return result
                except Exception as e2:
                    print(f"❌ Fallback inpainting failed: {e2}")
            
            raise Exception("All inpainting models failed")
    
    def apply_post_processing(self, result: Image.Image, original: Image.Image, 
                             target_size: Tuple[int, int], analysis: SceneAnalysis) -> Image.Image:
        """Apply advanced post-processing and quality enhancement."""
        print("✨ Applying post-processing...")
        
        target_width, target_height = target_size
        orig_width, orig_height = original.size
        
        x_offset = (target_width - orig_width) // 2
        y_offset = (target_height - orig_height) // 2
        
        # Color matching
        result = self._match_extension_colors(result, original, (x_offset, y_offset, x_offset + orig_width, y_offset + orig_height))
        
        # Seamless blending at edges
        result = self._apply_edge_blending(result, original, target_size)
        
        # Preserve original content perfectly
        result.paste(original, (x_offset, y_offset))
        
        # Final enhancement
        result = self._enhance_final_quality(result, original)
        
        return result
    
    def _match_extension_colors(self, result: Image.Image, original: Image.Image, 
                               original_bbox: Tuple[int, int, int, int]) -> Image.Image:
        """Match colors in extended areas to original image."""
        result_array = np.array(result, dtype=np.float32)
        orig_array = np.array(original, dtype=np.float32)
        
        # Get original color statistics
        orig_mean = orig_array.mean(axis=(0, 1))
        orig_std = orig_array.std(axis=(0, 1))
        
        # Create mask for extended areas
        x1, y1, x2, y2 = original_bbox
        mask = np.ones(result_array.shape[:2], dtype=bool)
        mask[y1:y2, x1:x2] = False
        
        # Apply color matching to extended areas only
        for c in range(3):
            extended_pixels = result_array[:, :, c][mask]
            if len(extended_pixels) > 0:
                ext_mean = extended_pixels.mean()
                ext_std = extended_pixels.std() + 1e-8
                
                # Normalize and rescale
                normalized = (extended_pixels - ext_mean) / ext_std
                matched = normalized * orig_std[c] * 0.7 + orig_mean[c]  # Partial matching
                
                result_array[:, :, c][mask] = np.clip(matched, 0, 255)
        
        return Image.fromarray(result_array.astype(np.uint8))
    
    def _apply_edge_blending(self, result: Image.Image, original: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Apply seamless edge blending."""
        # Simplified blending for performance
        target_width, target_height = target_size
        orig_width, orig_height = original.size
        
        x_offset = (target_width - orig_width) // 2
        y_offset = (target_height - orig_height) // 2
        
        # Create soft blend masks at edges
        blend_width = 15
        result_array = np.array(result, dtype=np.float32)
        
        # Apply subtle gaussian blur to extension areas only
        mask = np.zeros(result_array.shape[:2], dtype=bool)
        
        # Mark extension areas for subtle blurring
        if x_offset > 0:
            mask[:, :x_offset] = True
            mask[:, x_offset + orig_width:] = True
        if y_offset > 0:
            mask[:y_offset, :] = True
            mask[y_offset + orig_height:, :] = True
        
        # Apply very light blur to extension areas
        for c in range(3):
            channel = result_array[:, :, c]
            blurred_channel = gaussian_filter(channel, sigma=0.5)
            channel[mask] = blurred_channel[mask]
        
        return Image.fromarray(result_array.astype(np.uint8))
    
    def _enhance_final_quality(self, result: Image.Image, original: Image.Image) -> Image.Image:
        """Apply final quality enhancements."""
        # Subtle sharpening
        enhancer = ImageEnhance.Sharpness(result)
        result = enhancer.enhance(1.02)
        
        # Brightness matching
        orig_stat = ImageStat.Stat(original)
        result_stat = ImageStat.Stat(result)
        
        orig_brightness = sum(orig_stat.mean) / 3
        result_brightness = sum(result_stat.mean) / 3
        
        if abs(orig_brightness - result_brightness) > 5:
            brightness_factor = orig_brightness / result_brightness
            brightness_factor = np.clip(brightness_factor, 0.95, 1.05)
            
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(brightness_factor)
        
        return result
    
    def calculate_quality_metrics(self, result: Image.Image, original: Image.Image, 
                                 target_size: Tuple[int, int]) -> Dict[str, float]:
        """Calculate comprehensive quality metrics."""
        metrics = {}
        
        target_width, target_height = target_size
        orig_width, orig_height = original.size
        
        x_offset = (target_width - orig_width) // 2
        y_offset = (target_height - orig_height) // 2
        
        # Original preservation check
        extracted = result.crop((x_offset, y_offset, x_offset + orig_width, y_offset + orig_height))
        orig_array = np.array(original, dtype=np.float32)
        ext_array = np.array(extracted, dtype=np.float32)
        
        preservation_mse = np.mean((orig_array - ext_array) ** 2)
        metrics['preservation_score'] = max(0, 1 - preservation_mse / 1000)
        
        # Color consistency  
        orig_mean = orig_array.mean(axis=(0, 1))
        result_array = np.array(result, dtype=np.float32)
        
        # Sample extended areas
        extended_pixels = []
        if x_offset > 0:
            extended_pixels.extend(result_array[:, :x_offset].reshape(-1, 3))
            extended_pixels.extend(result_array[:, x_offset + orig_width:].reshape(-1, 3))
        if y_offset > 0:
            extended_pixels.extend(result_array[:y_offset, :].reshape(-1, 3))
            extended_pixels.extend(result_array[y_offset + orig_height:, :].reshape(-1, 3))
        
        if extended_pixels:
            extended_pixels = np.array(extended_pixels)
            ext_mean = extended_pixels.mean(axis=0)
            color_diff = np.abs(orig_mean - ext_mean).mean()
            metrics['color_consistency'] = max(0, 1 - color_diff / 50)
        else:
            metrics['color_consistency'] = 1.0
        
        # Overall quality score
        metrics['overall_score'] = (
            0.6 * metrics['preservation_score'] + 
            0.4 * metrics['color_consistency']
        )
        
        return metrics
    
    def resize_image(self, image_path: str, output_path: str) -> bool:
        """Main processing pipeline with comprehensive error handling."""
        start_time = time.time()
        
        try:
            print(f"\n🚀 === Processing: {Path(image_path).name} ===")
            
            # Load image
            original_img = Image.open(image_path).convert('RGB')
            orig_width, orig_height = original_img.size
            print(f"📐 Original size: {orig_width}x{orig_height}")
            
            # Calculate target dimensions
            target_width, target_height = self.calculate_target_dimensions(orig_width, orig_height)
            print(f"🎯 Target size: {target_width}x{target_height}")
            
            # Check if already correct ratio
            if orig_width == target_width and orig_height == target_height:
                print("✅ Image already has correct aspect ratio")
                original_img.save(output_path, quality=95, optimize=True)
                return True
            
            # STAGE 1: Scene Understanding (~2s)
            stage_start = time.time()
            print(f"\n📊 [Stage 1/4] Scene Understanding...")
            analysis = self.scene_analyzer.analyze_scene(original_img)
            print(f"⏱️  Stage 1 completed in {time.time() - stage_start:.1f}s")
            
            # STAGE 2: Strategy Selection (~0.1s)
            stage_start = time.time()
            print(f"\n🎛️  [Stage 2/4] Strategy Selection...")
            strategy = self.get_processing_strategy(analysis)
            print(f"🔧 Method: {strategy.extension_method}")
            print(f"💪 Strength: {strategy.inpaint_strength}")
            print(f"⚡ Steps: {strategy.inference_steps}")
            print(f"⏱️  Stage 2 completed in {time.time() - stage_start:.1f}s")
            
            # STAGE 3: AI Processing (~5-6s)
            stage_start = time.time()
            print(f"\n🎨 [Stage 3/4] AI Processing...")
            
            # Create adaptive canvas and mask
            canvas = self.create_adaptive_canvas(original_img, (target_width, target_height), strategy, analysis)
            mask = self.create_intelligent_mask(original_img, (target_width, target_height), analysis, strategy.edge_feather)
            
            # Generate dynamic prompt
            prompt = self.generate_dynamic_prompt(strategy, analysis)
            
            # Run optimized inpainting
            result = self.run_optimized_inpainting(canvas, mask, prompt, strategy)
            print(f"⏱️  Stage 3 completed in {time.time() - stage_start:.1f}s")
            
            # STAGE 4: Post-Processing (~1s)
            stage_start = time.time()
            print(f"\n✨ [Stage 4/4] Post-Processing...")
            
            final_result = self.apply_post_processing(result, original_img, (target_width, target_height), analysis)
            
            # Calculate quality metrics
            metrics = self.calculate_quality_metrics(final_result, original_img, (target_width, target_height))
            print(f"📊 Quality Score: {metrics['overall_score']:.2f}")
            print(f"🔒 Preservation: {metrics['preservation_score']:.2f}")
            print(f"🎨 Color Consistency: {metrics['color_consistency']:.2f}")
            print(f"⏱️  Stage 4 completed in {time.time() - stage_start:.1f}s")
            
            # Save result
            final_result.save(output_path, quality=95, optimize=True)
            
            total_time = time.time() - start_time
            print(f"\n✅ Processing completed in {total_time:.1f}s")
            print(f"💾 Saved to: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description="Advanced AI-powered image resizer to 9:16 aspect ratio")
    parser.add_argument("input", help="Input image path or directory")
    parser.add_argument("-o", "--output", help="Output path or directory")
    parser.add_argument("-d", "--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("-b", "--batch", action="store_true", help="Batch process directory")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress detailed output")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Initialize resizer
    try:
        resizer = ImageResizer(device=args.device)
    except Exception as e:
        print(f"❌ Failed to initialize resizer: {e}")
        return
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image processing
        output_path = args.output if args.output else f"{input_path.stem}_resized{input_path.suffix}"
        
        start_time = time.time()
        success = resizer.resize_image(str(input_path), output_path)
        end_time = time.time()
        
        if success and not args.quiet:
            print(f"\n🎉 Processing completed in {end_time - start_time:.1f} seconds")
    
    elif input_path.is_dir() and args.batch:
        # Batch processing
        output_dir = Path(args.output) if args.output else input_path / "resized"
        output_dir.mkdir(exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print("❌ No image files found in directory")
            return
        
        print(f"🔄 Processing {len(image_files)} images...")
        successful = 0
        total_time = 0
        
        for i, image_file in enumerate(image_files, 1):
            if not args.quiet:
                print(f"\n📁 Processing {i}/{len(image_files)}: {image_file.name}")
            
            output_file = output_dir / f"{image_file.stem}_resized{image_file.suffix}"
            
            start_time = time.time()
            success = resizer.resize_image(str(image_file), str(output_file))
            end_time = time.time()
            
            if success:
                successful += 1
                total_time += (end_time - start_time)
            
            if not args.quiet:
                status = "✅ Success" if success else "❌ Failed"
                print(f"  {status} ({end_time - start_time:.1f}s)")
        
        if not args.quiet:
            avg_time = total_time / successful if successful > 0 else 0
            print(f"\n🎉 Batch processing completed:")
            print(f"  ✅ Successful: {successful}/{len(image_files)}")
            print(f"  ⏱️  Average time: {avg_time:.1f}s per image")
    
    else:
        print("❌ Invalid input path or missing --batch flag for directory processing")
        sys.exit(1)


if __name__ == "__main__":
    main()