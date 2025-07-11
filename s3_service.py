"""S3 service for uploading and managing processed images."""

import boto3
import io
import hashlib
from typing import List, Optional
from urllib.parse import urlparse
from tenacity import retry, stop_after_attempt, wait_exponential
from PIL import Image
import requests

from config import config
from logger import logger

class S3Service:
    """Service for S3 operations."""
    
    def __init__(self):
        self.s3_client = boto3.client('s3', region_name=config.aws_region)
        self.bucket = config.s3_bucket
    
    def _generate_s3_key(self, product_id: str, image_index: int, original_url: str) -> str:
        """Generate S3 key for processed image."""
        # Extract file extension from original URL
        parsed_url = urlparse(original_url)
        path = parsed_url.path
        extension = path.split('.')[-1] if '.' in path else 'jpg'
        
        # Create unique key
        url_hash = hashlib.md5(original_url.encode()).hexdigest()[:8]
        return f"{config.s3_key_prefix}/{product_id}/image_{image_index}_{url_hash}.{extension}"
    
    @retry(
        stop=stop_after_attempt(config.s3_max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def upload_image(self, image: Image.Image, product_id: str, image_index: int, original_url: str) -> str:
        """Upload processed image to S3."""
        try:
            # Generate S3 key
            s3_key = self._generate_s3_key(product_id, image_index, original_url)
            
            # Convert PIL Image to bytes
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG', quality=95, optimize=True)
            img_buffer.seek(0)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=img_buffer.getvalue(),
                ContentType='image/jpeg',
                Metadata={
                    'product_id': product_id,
                    'image_index': str(image_index),
                    'original_url': original_url,
                    'processed_by': 'clore-image-resizer'
                }
            )
            
            # Generate public URL
            s3_url = f"https://{self.bucket}.s3.{config.aws_region}.amazonaws.com/{s3_key}"
            
            logger.info(
                "Uploaded image to S3",
                product_id=product_id,
                image_index=image_index,
                s3_key=s3_key,
                s3_url=s3_url
            )
            
            return s3_url
            
        except Exception as e:
            logger.error(
                "Failed to upload image to S3",
                product_id=product_id,
                image_index=image_index,
                error=str(e)
            )
            raise
    
    @retry(
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def download_image(self, url: str) -> Optional[Image.Image]:
        """Download image from URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Open image with PIL
            image = Image.open(io.BytesIO(response.content))
            image = image.convert('RGB')  # Ensure RGB format
            
            logger.info("Downloaded image", url=url, size=image.size)
            return image
            
        except Exception as e:
            logger.error("Failed to download image", url=url, error=str(e))
            return None
    
    def upload_batch_images(self, images: List[Image.Image], product_id: str, original_urls: List[str]) -> List[str]:
        """Upload multiple images and return S3 URLs in the same order."""
        s3_urls = []
        
        for index, (image, original_url) in enumerate(zip(images, original_urls)):
            try:
                s3_url = self.upload_image(image, product_id, index, original_url)
                s3_urls.append(s3_url)
            except Exception as e:
                logger.error(
                    "Failed to upload image in batch",
                    product_id=product_id,
                    image_index=index,
                    error=str(e)
                )
                # Keep the original URL if upload fails
                s3_urls.append(original_url)
        
        return s3_urls
    
    def download_batch_images(self, urls: List[str]) -> List[Optional[Image.Image]]:
        """Download multiple images from URLs."""
        images = []
        
        for url in urls:
            image = self.download_image(url)
            images.append(image)
        
        return images
    
    @retry(
        stop=stop_after_attempt(config.s3_max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def delete_image(self, s3_url: str) -> bool:
        """Delete image from S3."""
        try:
            # Extract key from URL
            parsed_url = urlparse(s3_url)
            if parsed_url.netloc != f"{self.bucket}.s3.{config.aws_region}.amazonaws.com":
                logger.warning("URL is not from our S3 bucket", url=s3_url)
                return False
            
            s3_key = parsed_url.path.lstrip('/')
            
            # Delete from S3
            self.s3_client.delete_object(
                Bucket=self.bucket,
                Key=s3_key
            )
            
            logger.info("Deleted image from S3", s3_key=s3_key)
            return True
            
        except Exception as e:
            logger.error("Failed to delete image from S3", url=s3_url, error=str(e))
            return False
    
    def cleanup_product_images(self, product_id: str) -> bool:
        """Delete all images for a product."""
        try:
            # List all objects with the product prefix
            prefix = f"{config.s3_key_prefix}/{product_id}/"
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                # Delete all objects
                objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
                
                self.s3_client.delete_objects(
                    Bucket=self.bucket,
                    Delete={'Objects': objects_to_delete}
                )
                
                logger.info(
                    "Cleaned up product images",
                    product_id=product_id,
                    deleted_count=len(objects_to_delete)
                )
            
            return True
            
        except Exception as e:
            logger.error("Failed to cleanup product images", product_id=product_id, error=str(e))
            return False
    
    def get_image_metadata(self, s3_url: str) -> Optional[dict]:
        """Get metadata for an S3 image."""
        try:
            parsed_url = urlparse(s3_url)
            if parsed_url.netloc != f"{self.bucket}.s3.{config.aws_region}.amazonaws.com":
                return None
            
            s3_key = parsed_url.path.lstrip('/')
            
            response = self.s3_client.head_object(
                Bucket=self.bucket,
                Key=s3_key
            )
            
            return response.get('Metadata', {})
            
        except Exception as e:
            logger.error("Failed to get image metadata", url=s3_url, error=str(e))
            return None