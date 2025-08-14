#!/usr/bin/env python3
"""
GPT Fashion Tagger
Simple, accurate tagging using GPT-5 nano with image analysis
"""

import json
import base64
import io
import os
from typing import List, Dict, Optional
from PIL import Image
from openai import OpenAI
from dataclasses import dataclass
import time

from logger import logger

@dataclass
class TaggingResult:
    """Result from GPT tagging"""
    tags: List[str]
    gender: str
    category: str
    aesthetics: List[str]
    processing_time: float = 0.0
    error: Optional[str] = None

class GPTFashionTagger:
    """GPT-5 nano powered fashion tagging system"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5-nano"):
        """Initialize with OpenAI API key"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        logger.info("GPT Fashion Tagger initialized", model=self.model)

    def encode_image_to_base64(self, image: Image.Image, max_size: int = 1024) -> str:
        """Convert PIL Image to base64 string for API"""
        try:
            # Resize if too large to control costs and API limits
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{image_b64}"
        except Exception as e:
            logger.error("Error encoding image to base64", error=str(e))
            raise

    def tag_product(self, 
                   images: List[Image.Image], 
                   product_name: str,
                   product_id: str) -> TaggingResult:
        """
        Tag a product using GPT-5 nano vision analysis
        
        Args:
            images: List of PIL Images of the product
            product_name: Name/title of the product  
            product_id: Product ID for logging
            
        Returns:
            TaggingResult with tags, gender, and category
        """
        start_time = time.time()
        
        try:
            # Limit to 4 images to control costs
            images_to_process = images[:4]
            logger.info("Starting GPT tagging", product_id=product_id, image_count=len(images_to_process))
            
            # Prepare image data
            image_data = []
            for i, img in enumerate(images_to_process):
                try:
                    # For the new API, we might need URLs instead of base64
                    # This is a placeholder - adjust based on actual GPT-5 nano API
                    image_b64 = self.encode_image_to_base64(img)
                    image_data.append(image_b64)
                except Exception as e:
                    logger.warning(f"Error processing image {i}", product_id=product_id, error=str(e))
                    continue
            
            if not image_data:
                raise ValueError("No valid images to process")
            
            # Build the prompt
            prompt_text = f"""Analyze this fashion product: "{product_name}"

You must respond with ONLY valid JSON in this EXACT format (no extra text):

{{
    "tags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7"],
    "gender": "Male|Female|Unisex",
    "category": "Main Category Name",
    "aesthetics": ["aesthetic1", "aesthetic2", "aesthetic3"]
}}

Rules:
- tags: exactly 7 descriptive tags for product discovery (lowercase, relevant for search)
- gender: exactly one of: "Male", "Female", "Unisex"
- category: main category like "Hoodie", "Jeans", "Sneakers", "Dress", etc. (Title Case)
- aesthetics: exactly 3 style aesthetics that describe this product (lowercase)

Gender Classification Guidelines:
- Female: baby tee, crop top, mini skirt, dress, women's blouse, feminine cuts, typically women's sizing
- Male: men's shirt, masculine cuts, boxier fits, typically men's sizing  
- Unisex: hoodies, basic t-shirts, jeans, sneakers, items that work for any gender

Aesthetic Examples:
- y2k, grunge, minimalist, vintage, retro, gothic, kawaii, preppy, bohemian, industrial
- streetwear, cottagecore, dark academia, light academia, fairycore, indie, alt
- coquette, soft girl, clean girl, baddie, e-girl, academia, normcore, gorpcore
- cyberpunk, steampunk, punk, emo, scene, hipster, sporty, elegant, edgy

Focus on: style, fit, color, material, occasion, gender-specific design cues, and visual aesthetics
Response must be valid JSON only"""

            # Call GPT-5 nano API (using the new format you showed)
            response = self.client.responses.create(
                model=self.model,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_text},
                        *[{"type": "input_image", "image_url": img_url} for img_url in image_data]
                    ]
                }]
            )
            
            # Parse response
            result_text = response.output_text.strip()
            logger.debug("GPT response received", product_id=product_id, response_length=len(result_text))
            
            try:
                result_data = json.loads(result_text)
            except json.JSONDecodeError as e:
                logger.error("JSON parsing failed", product_id=product_id, response=result_text[:200], error=str(e))
                raise ValueError(f"Invalid JSON response from GPT: {str(e)}")
            
            # Validate and clean the response
            tags = result_data.get("tags", [])
            gender = result_data.get("gender", "Unisex")
            category = result_data.get("category", "Unknown")
            aesthetics = result_data.get("aesthetics", [])
            
            # Ensure exactly 7 tags
            if len(tags) < 7:
                tags.extend(["fashion", "clothing", "apparel", "style", "trendy", "casual", "wear"][:7-len(tags)])
            elif len(tags) > 7:
                tags = tags[:7]
            
            # Ensure exactly 3 aesthetics
            if len(aesthetics) < 3:
                aesthetics.extend(["minimalist", "casual", "modern"][:3-len(aesthetics)])
            elif len(aesthetics) > 3:
                aesthetics = aesthetics[:3]
            
            # Validate gender
            if gender not in ["Male", "Female", "Unisex"]:
                logger.warning("Invalid gender from GPT, defaulting to Unisex", product_id=product_id, received_gender=gender)
                gender = "Unisex"
            
            # Ensure category is not empty
            if not category or category.lower() == "unknown":
                category = "Apparel"
            
            processing_time = time.time() - start_time
            
            logger.info(
                "GPT tagging completed successfully", 
                product_id=product_id,
                category=category,
                gender=gender,
                tag_count=len(tags),
                aesthetic_count=len(aesthetics),
                processing_time=round(processing_time, 2)
            )
            
            return TaggingResult(
                tags=tags,
                gender=gender,
                category=category,
                aesthetics=aesthetics,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            logger.error(
                "GPT tagging failed", 
                product_id=product_id,
                error=error_msg,
                processing_time=round(processing_time, 2)
            )
            
            # Return fallback result
            return TaggingResult(
                tags=["fashion", "clothing", "apparel", "style", "trendy", "casual", "wear"],
                gender="Unisex",
                category="Apparel",
                aesthetics=["minimalist", "casual", "modern"],
                processing_time=processing_time,
                error=error_msg
            )

    def estimate_cost(self, num_products: int, images_per_product: int = 3) -> Dict[str, float]:
        """Estimate API costs for tagging (placeholder - update with actual GPT-5 nano pricing)"""
        
        # Placeholder pricing - update when you get actual GPT-5 nano costs
        estimated_cost_per_product = 0.002  # $0.002 per product (estimated)
        
        total_cost = num_products * estimated_cost_per_product
        
        return {
            "total_cost": round(total_cost, 4),
            "cost_per_product": estimated_cost_per_product,
            "estimated_monthly_cost": round(total_cost * 30, 2) if num_products <= 1000 else round(total_cost, 2)
        }

def main():
    """Test the GPT Fashion Tagger"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python gpt_fashion_tagger.py <product_name> <image_path1> [image_path2] ...")
        return
    
    product_name = sys.argv[1]
    image_paths = sys.argv[2:]
    
    try:
        # Load images
        images = []
        for path in image_paths:
            if os.path.exists(path):
                images.append(Image.open(path).convert("RGB"))
            else:
                print(f"Image not found: {path}")
        
        if not images:
            print("No valid images found")
            return
        
        # Initialize tagger
        tagger = GPTFashionTagger()
        
        # Tag product
        result = tagger.tag_product(images, product_name, "test_product")
        
        # Display results
        print(f"\nProduct: {product_name}")
        print(f"Category: {result.category}")
        print(f"Gender: {result.gender}")
        print(f"Tags: {', '.join(result.tags)}")
        print(f"Processing time: {result.processing_time:.2f}s")
        
        if result.error:
            print(f"Error: {result.error}")
        
        # Cost estimation
        costs = tagger.estimate_cost(1000)
        print(f"\nCost estimation for 1000 products:")
        print(f"Total: ${costs['total_cost']}")
        print(f"Per product: ${costs['cost_per_product']}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()