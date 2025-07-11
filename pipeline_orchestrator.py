"""Main orchestrator for the image processing pipeline."""

import time
import signal
import sys
import os
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import gc

from config import config
from logger import logger
from sqs_service import SQSService, ProcessingMessage
from database_service import DatabaseService, Product
from s3_service import S3Service
from image_resizer import ImageResizer
from multi_image_streetwear_tagger import MultiImageStreetwearTagger
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

@dataclass
class ProcessingResult:
    """Result of processing a single product."""
    product_id: str
    success: bool
    processed_images: int
    error: Optional[str] = None
    processing_time: float = 0.0

class PipelineOrchestrator:
    """Main orchestrator for the image processing pipeline."""
    
    def __init__(self):
        self.sqs_service = SQSService()
        self.db_service = DatabaseService()
        self.s3_service = S3Service()
        self.image_resizer = ImageResizer(device=config.device)
        self.image_tagger = MultiImageStreetwearTagger(memory_efficient=True)
        
        self.running = True
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = time.time()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Pipeline orchestrator initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def process_product(self, message: ProcessingMessage) -> ProcessingResult:
        """Process a single product."""
        start_time = time.time()
        
        try:
            logger.info("Processing product", product_id=message.product_id)
            
            # Update processing status
            self.db_service.update_processing_status(message.product_id, "processing")
            
            # Get product from database
            product = self.db_service.get_product_images(message.product_id)
            if not product:
                logger.warning("Product not found in database", product_id=message.product_id)
                return ProcessingResult(
                    product_id=message.product_id,
                    success=False,
                    processed_images=0,
                    error="Product not found in database"
                )
            
            if not product.images:
                logger.warning("No images found for product", product_id=message.product_id)
                return ProcessingResult(
                    product_id=message.product_id,
                    success=False,
                    processed_images=0,
                    error="No images found for product"
                )
            
            # Backup original images
            self.db_service.backup_original_images(message.product_id, product.images)
            
            # Download images
            logger.info("Downloading images", product_id=message.product_id, image_count=len(product.images))
            downloaded_images = self.s3_service.download_batch_images(product.images)
            
            # Filter out failed downloads
            valid_images = []
            valid_urls = []
            for img, url in zip(downloaded_images, product.images):
                if img is not None:
                    valid_images.append(img)
                    valid_urls.append(url)
                else:
                    logger.warning("Failed to download image", url=url)
            
            if not valid_images:
                logger.error("No images could be downloaded", product_id=message.product_id)
                return ProcessingResult(
                    product_id=message.product_id,
                    success=False,
                    processed_images=0,
                    error="No images could be downloaded"
                )
            
            # Process images with AI resizer
            logger.info("Processing images with AI", product_id=message.product_id, image_count=len(valid_images))
            processed_images = []
            
            for i, image in enumerate(valid_images):
                try:
                    # Scale to 1080p first
                    scaled_image = self.image_resizer.scale_to_1080p(image)
                    
                    # Calculate target dimensions
                    width, height = scaled_image.size
                    target_width, target_height = self.image_resizer.calculate_target_dimensions(width, height)
                    
                    # Skip if already correct ratio
                    if width == target_width and height == target_height:
                        processed_images.append(scaled_image)
                        continue
                    
                    # Create base canvas and mask
                    base_canvas = self.image_resizer.create_base_canvas(scaled_image, (target_width, target_height))
                    mask = self.image_resizer.create_extension_mask(scaled_image, (target_width, target_height))
                    
                    # Generate prompt
                    prompt = self.image_resizer.generate_inpaint_prompt()
                    
                    # Resize for processing
                    process_size = 1024
                    scale_factor = process_size / max(target_width, target_height)
                    
                    if scale_factor < 1:
                        process_width = int(target_width * scale_factor)
                        process_height = int(target_height * scale_factor)
                        
                        # Ensure dimensions are divisible by 8
                        process_width = ((process_width + 7) // 8) * 8
                        process_height = ((process_height + 7) // 8) * 8
                        
                        base_resized = base_canvas.resize((process_width, process_height), Image.LANCZOS)
                        mask_resized = mask.resize((process_width, process_height), Image.LANCZOS)
                    else:
                        base_resized = base_canvas
                        mask_resized = mask
                        process_width, process_height = target_width, target_height
                    
                    # Run AI inpainting
                    logger.info("Running AI inpainting", product_id=message.product_id, image_index=i)
                    
                    result = self.image_resizer.inpaint_pipeline(
                        prompt=prompt,
                        image=base_resized,
                        mask_image=mask_resized,
                        num_inference_steps=20,
                        strength=0.8,
                        guidance_scale=7.5,
                        height=process_height,
                        width=process_width
                    ).images[0]
                    
                    # Resize back to target size if needed
                    if scale_factor < 1:
                        result = result.resize((target_width, target_height), Image.LANCZOS)
                    
                    processed_images.append(result)
                    
                    # Clear GPU memory
                    if hasattr(self.image_resizer, 'device') and 'cuda' in self.image_resizer.device and torch is not None:
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    logger.error("Failed to process image", product_id=message.product_id, image_index=i, error=str(e))
                    # Use original image if processing fails
                    processed_images.append(valid_images[i])
            
            # Upload processed images to S3
            logger.info("Uploading processed images", product_id=message.product_id, image_count=len(processed_images))
            s3_urls = self.s3_service.upload_batch_images(processed_images, message.product_id, valid_urls)
            
            # Update database with new URLs
            logger.info("Updating database with new URLs", product_id=message.product_id)
            self.db_service.update_product_images(message.product_id, s3_urls)
            
            # Tag the processed images
            try:
                logger.info("Tagging processed images", product_id=message.product_id)
                
                # Download processed images for tagging
                processed_image_files = []
                for i, processed_img in enumerate(processed_images):
                    # Save image temporarily for tagging
                    temp_path = f"/tmp/{message.product_id}_processed_{i}.jpg"
                    processed_img.save(temp_path, "JPEG", quality=95)
                    processed_image_files.append(temp_path)
                
                # Run image tagger on processed images
                tag_result = self.image_tagger.analyze_product(processed_image_files, message.product_id)
                
                if "error" not in tag_result:
                    consensus = tag_result["consensus"]
                    tags = consensus.get("comprehensive_tags", [])
                    gender = consensus.get("consensus_gender", "unisex")
                    category = consensus.get("consensus_category", "unknown")
                    
                    # Update database with tags
                    self.db_service.update_product_tags(message.product_id, tags, gender, category)
                    logger.info(
                        "Product tagged successfully",
                        product_id=message.product_id,
                        tags=tags,
                        gender=gender,
                        category=category
                    )
                else:
                    logger.warning("Tagging failed", product_id=message.product_id, error=tag_result["error"])
                
                # Clean up temporary files
                for temp_file in processed_image_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        
            except Exception as e:
                logger.error("Error during tagging", product_id=message.product_id, error=str(e))
            
            self.db_service.update_processing_status(message.product_id, "completed")
            
            processing_time = time.time() - start_time
            
            logger.info(
                "Product processing completed",
                product_id=message.product_id,
                processed_images=len(processed_images),
                processing_time=processing_time
            )
            
            return ProcessingResult(
                product_id=message.product_id,
                success=True,
                processed_images=len(processed_images),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error("Product processing failed", product_id=message.product_id, error=str(e))
            self.db_service.update_processing_status(message.product_id, "failed")
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                product_id=message.product_id,
                success=False,
                processed_images=0,
                error=str(e),
                processing_time=processing_time
            )
    
    def process_batch(self, messages: List[ProcessingMessage]) -> List[ProcessingResult]:
        """Process a batch of messages."""
        results = []
        
        for message in messages:
            if not self.running:
                break
                
            result = self.process_product(message)
            results.append(result)
            
            if result.success:
                self.processed_count += 1
                # Delete message from SQS
                self.sqs_service.delete_message(message.receipt_handle)
            else:
                self.failed_count += 1
                # Keep message in queue for retry (will be retried after visibility timeout)
        
        return results
    
    def run(self):
        """Main processing loop."""
        logger.info("Starting pipeline orchestrator")
        
        while self.running:
            try:
                # Check queue depth
                queue_depth = self.sqs_service.get_queue_depth()
                if queue_depth == 0:
                    logger.info("Queue is empty, waiting...")
                    time.sleep(30)
                    continue
                
                logger.info("Processing batch", queue_depth=queue_depth)
                
                # Receive messages
                messages = self.sqs_service.poll_messages(config.batch_size)
                
                if not messages:
                    logger.info("No messages received, waiting...")
                    time.sleep(10)
                    continue
                
                # Process messages
                results = self.process_batch(messages)
                
                # Log batch results
                successful = sum(1 for r in results if r.success)
                failed = len(results) - successful
                
                logger.info(
                    "Batch processing completed",
                    batch_size=len(results),
                    successful=successful,
                    failed=failed,
                    total_processed=self.processed_count,
                    total_failed=self.failed_count
                )
                
            except Exception as e:
                logger.error("Error in main processing loop", error=str(e))
                time.sleep(10)
        
        self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down pipeline orchestrator")
        
        # Calculate final statistics
        total_time = time.time() - self.start_time
        total_processed = self.processed_count + self.failed_count
        
        logger.info(
            "Pipeline orchestrator shutdown complete",
            total_processed=total_processed,
            successful=self.processed_count,
            failed=self.failed_count,
            total_time=total_time,
            avg_time_per_product=total_time / max(total_processed, 1)
        )
        
        # Close database connections
        self.db_service.close()
        
        logger.info("Goodbye!")
    
    def get_status(self) -> dict:
        """Get current processing status."""
        return {
            "running": self.running,
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
            "queue_depth": self.sqs_service.get_queue_depth(),
            "in_flight_messages": self.sqs_service.get_in_flight_messages(),
            "uptime": time.time() - self.start_time
        }

def main():
    """Main entry point."""
    orchestrator = PipelineOrchestrator()
    
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error("Fatal error in orchestrator", error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()