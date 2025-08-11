"""Main orchestrator for the image processing pipeline."""

import time
import signal
import sys
import os
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import gc
import threading
from datetime import datetime, timedelta

from config import config
from logger import logger, save_shutdown_logs
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
        self.skipped_count = 0
        self.total_images_processed = 0
        self.current_product_id = None
        self.current_stage = "Initializing"
        self.current_batch_size = 0
        self.start_time = datetime.now()
        self.current_batch_progress = 0
        self.recent_processing_times = []
        self.last_activity = time.time()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Start UI update thread
        self.ui_thread = threading.Thread(target=self._update_ui, daemon=True)
        self.ui_thread.start()
        
        logger.info("Pipeline orchestrator initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def _clear_screen(self):
        """Clear the terminal screen."""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes:.0f}m {secs:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def _get_progress_bar(self, current: int, total: int, width: int = 50) -> str:
        """Generate a progress bar."""
        if total == 0:
            return "█" * width
        
        filled = int(width * current / total)
        bar = "█" * filled + "░" * (width - filled)
        percentage = (current / total) * 100 if total > 0 else 0
        return f"{bar} {percentage:.1f}%"
    
    def _get_speed_stats(self) -> tuple:
        """Get current processing speed statistics."""
        if not self.recent_processing_times:
            return 0.0, 0.0, "0s"
        
        avg_time = sum(self.recent_processing_times) / len(self.recent_processing_times)
        current_speed = 1 / avg_time if avg_time > 0 else 0
        
        total_processed = self.processed_count + self.failed_count
        if total_processed > 0:
            total_time = time.time() - self.start_time
            overall_speed = total_processed / total_time
        else:
            overall_speed = 0
        
        # Estimate time remaining
        queue_depth = self.sqs_service.get_queue_depth()
        if current_speed > 0 and queue_depth > 0:
            eta_seconds = queue_depth / current_speed
            eta = self._format_duration(eta_seconds)
        else:
            eta = "Unknown"
        
        return current_speed, overall_speed, eta
    
    def _update_ui(self):
        """Update the terminal UI continuously."""
        while self.running:
            try:
                self._clear_screen()
                
                # Header
                print("🚀 CLORE IMAGE PROCESSING PIPELINE")
                print("=" * 80)
                
                # Status section
                uptime = time.time() - self.start_time
                queue_depth = self.sqs_service.get_queue_depth()
                in_flight = self.sqs_service.get_in_flight_messages()
                
                status_color = "🟢" if self.running else "🔴"
                print(f"\n{status_color} Status: {'RUNNING' if self.running else 'STOPPED'}")
                print(f"⏱️  Uptime: {self._format_duration(uptime)}")
                print(f"📊 Queue Depth: {queue_depth}")
                print(f"✈️  In Flight: {in_flight}")
                
                # Processing statistics
                print(f"\n📈 PROCESSING STATISTICS")
                print("-" * 40)
                
                total_processed = self.processed_count + self.failed_count + self.skipped_count
                success_rate = (self.processed_count / max(total_processed, 1)) * 100
                
                print(f"✅ Successful: {self.processed_count}")
                print(f"❌ Failed: {self.failed_count}")
                print(f"⏭️  Skipped: {self.skipped_count}")
                print(f"🖼️  Images Processed: {self.total_images_processed}")
                print(f"📊 Success Rate: {success_rate:.1f}%")
                
                # Speed statistics
                current_speed, overall_speed, eta = self._get_speed_stats()
                print(f"\n⚡ PERFORMANCE")
                print("-" * 40)
                print(f"Current Speed: {current_speed:.2f} products/min")
                print(f"Overall Speed: {overall_speed:.2f} products/min")
                print(f"ETA: {eta}")
                
                # Current activity
                print(f"\n🔄 CURRENT ACTIVITY")
                print("-" * 40)
                
                if self.current_product_id:
                    print(f"Product: {self.current_product_id[:8]}...")
                    print(f"Stage: {self.current_stage}")
                    
                    if self.current_batch_size > 0:
                        progress_bar = self._get_progress_bar(
                            self.current_batch_progress, 
                            self.current_batch_size
                        )
                        print(f"Batch Progress: {progress_bar}")
                        print(f"({self.current_batch_progress}/{self.current_batch_size})")
                else:
                    if queue_depth == 0:
                        print("⏳ Waiting for new messages...")
                    else:
                        print("🔍 Checking queue...")
                
                # Recent activity
                time_since_activity = time.time() - self.last_activity
                if time_since_activity < 60:
                    print(f"Last activity: {time_since_activity:.0f}s ago")
                else:
                    print(f"Last activity: {self._format_duration(time_since_activity)} ago")
                
                # Bottom status bar
                print(f"\n{'=' * 80}")
                print(f"💡 Press Ctrl+C to stop gracefully")
                
                time.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                # Don't let UI errors crash the main process
                time.sleep(5)
    
    def _update_current_stage(self, stage: str, product_id: str = None):
        """Update current processing stage."""
        self.current_stage = stage
        if product_id:
            self.current_product_id = product_id
        self.last_activity = time.time()
    
    def _format_category(self, category: str) -> str:
        """Format category for frontend display."""
        if not category or category == "unknown":
            return "Unknown"
        
        # Replace underscores with spaces and title case
        formatted = category.replace("_", " ").title()
        
        # Handle special cases for better formatting
        special_cases = {
            "Oversized Tee": "Oversized Tee",
            "Graphic Tee": "Graphic Tee",
            "Cropped Hoodie": "Cropped Hoodie",
            "Oversized Hoodie": "Oversized Hoodie",
            "Zip Hoodie": "Zip Hoodie",
            "Baggy Jeans": "Baggy Jeans",
            "Skinny Jeans": "Skinny Jeans",
            "Cargo Pants": "Cargo Pants",
            "Wide Leg Pants": "Wide Leg Pants",
            "Bomber Jacket": "Bomber Jacket",
            "Denim Jacket": "Denim Jacket",
            "Puffer Jacket": "Puffer Jacket",
            "Coach Jacket": "Coach Jacket",
            "Chunky Sneakers": "Chunky Sneakers",
            "High Tops": "High Tops",
            "Skate Shoes": "Skate Shoes",
            "Bucket Hat": "Bucket Hat",
            "Crossbody Bag": "Crossbody Bag",
            "Chain Necklace": "Chain Necklace"
        }
        
        return special_cases.get(formatted, formatted)
    
    def _format_gender(self, gender: str) -> str:
        """Format gender for frontend display."""
        if not gender:
            return "Unisex"
        
        gender_mapping = {
            "male": "Male",
            "female": "Female", 
            "unisex": "Unisex",
            "men": "Male",
            "women": "Female",
            "mens": "Male",
            "womens": "Female"
        }
        
        return gender_mapping.get(gender.lower(), "Unisex")
    
    def process_product(self, message: ProcessingMessage) -> ProcessingResult:
        """Process a single product."""
        start_time = time.time()
        
        try:
            self._update_current_stage("Checking product status", message.product_id)
            logger.info("Processing product", product_id=message.product_id)
            
            # Check if product is already finished
            current_status = self.db_service.get_processing_status(message.product_id)
            if current_status == "completed":
                logger.info("Product already completed, skipping", product_id=message.product_id)
                self.skipped_count += 1
                return ProcessingResult(
                    product_id=message.product_id,
                    success=True,
                    processed_images=0,
                    processing_time=time.time() - start_time
                )
            
            # Update processing status
            self._update_current_stage("Updating database status", message.product_id)
            self.db_service.update_processing_status(message.product_id, "processing")
            
            # Get product from database
            self._update_current_stage("Fetching product data", message.product_id)
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
            self._update_current_stage("Backing up original images", message.product_id)
            self.db_service.backup_original_images(message.product_id, product.images)
            
            # Download images
            self._update_current_stage(f"Downloading {len(product.images)} images", message.product_id)
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
            self._update_current_stage(f"AI processing {len(valid_images)} images", message.product_id)
            logger.info("Processing images with AI", product_id=message.product_id, image_count=len(valid_images))
            processed_images = []
            
            for i, image in enumerate(valid_images):
                try:
                    self._update_current_stage(f"AI processing image {i+1}/{len(valid_images)}", message.product_id)
                    
                    # Save image to temporary file for processing
                    import tempfile
                    import os
                    
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_input:
                        image.save(temp_input.name, format='JPEG', quality=95)
                        temp_input_path = temp_input.name
                    
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_output:
                        temp_output_path = temp_output.name
                    
                    try:
                        # Use the advanced pipeline
                        logger.info("Running advanced AI pipeline", product_id=message.product_id, image_index=i)
                        success = self.image_resizer.resize_image(temp_input_path, temp_output_path)
                        
                        if success:
                            # Load the processed result
                            result = Image.open(temp_output_path).convert('RGB')
                            processed_images.append(result)
                        else:
                            logger.warning("Image processing failed, using original", product_id=message.product_id, image_index=i)
                            processed_images.append(image)
                        
                    finally:
                        # Clean up temporary files
                        try:
                            os.unlink(temp_input_path)
                            os.unlink(temp_output_path)
                        except:
                            pass
                    
                    # Clear GPU memory
                    if hasattr(self.image_resizer, 'device') and 'cuda' in self.image_resizer.device and torch is not None:
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    logger.error("Failed to process image", product_id=message.product_id, image_index=i, error=str(e))
                    # Use original image if processing fails
                    processed_images.append(valid_images[i])
            
            # Upload processed images to S3
            self._update_current_stage(f"Uploading {len(processed_images)} images to S3", message.product_id)
            logger.info("Uploading processed images", product_id=message.product_id, image_count=len(processed_images))
            s3_urls = self.s3_service.upload_batch_images(processed_images, message.product_id, valid_urls)
            
            # Update database with new URLs
            self._update_current_stage("Updating database with new URLs", message.product_id)
            logger.info("Updating database with new URLs", product_id=message.product_id)
            self.db_service.update_product_images(message.product_id, s3_urls)
            
            # Tag the processed images
            try:
                self._update_current_stage("AI tagging processed images", message.product_id)
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
                    raw_gender = consensus.get("consensus_gender", "unisex")
                    raw_category = consensus.get("consensus_category", "unknown")
                    
                    # Format for frontend display
                    gender = self._format_gender(raw_gender)
                    category = self._format_category(raw_category)
                    
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
            
            self._update_current_stage("Finalizing product", message.product_id)
            self.db_service.update_processing_status(message.product_id, "completed")
            
            processing_time = time.time() - start_time
            
            # Update tracking stats
            self.total_images_processed += len(processed_images)
            self.recent_processing_times.append(processing_time)
            
            # Keep only recent times for speed calculation
            if len(self.recent_processing_times) > 20:
                self.recent_processing_times = self.recent_processing_times[-20:]
            
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
        
        # Set up batch tracking
        self.current_batch_size = len(messages)
        self.current_batch_progress = 0
        
        for i, message in enumerate(messages):
            if not self.running:
                break
            
            self.current_batch_progress = i + 1
            result = self.process_product(message)
            results.append(result)
            
            if result.success:
                self.processed_count += 1
                # Delete message from SQS
                self.sqs_service.delete_message(message.receipt_handle)
            else:
                self.failed_count += 1
                # Keep message in queue for retry (will be retried after visibility timeout)
        
        # Reset batch tracking
        self.current_batch_size = 0
        self.current_batch_progress = 0
        self.current_product_id = None
        
        return results
    
    def run(self):
        """Main processing loop."""
        logger.info("Starting pipeline orchestrator")
        self._update_current_stage("Starting up")
        
        while self.running:
            try:
                # Check queue depth
                queue_depth = self.sqs_service.get_queue_depth()
                if queue_depth == 0:
                    self._update_current_stage("Queue empty, waiting...")
                    logger.info("Queue is empty, waiting...")
                    time.sleep(30)
                    continue
                
                self._update_current_stage(f"Processing batch (queue: {queue_depth})")
                logger.info("Processing batch", queue_depth=queue_depth)
                
                # Receive messages
                messages = self.sqs_service.poll_messages(config.batch_size)
                
                if not messages:
                    self._update_current_stage("No messages received, waiting...")
                    logger.info("No messages received, waiting...")
                    time.sleep(10)
                    continue
                
                # Process messages
                results = self.process_batch(messages)
                
                # Log batch results
                successful = sum(1 for r in results if r.success)
                failed = len(results) - successful
                skipped = sum(1 for r in results if r.success and r.processed_images == 0)
                actually_processed = sum(1 for r in results if r.success and r.processed_images > 0)
                
                logger.info(
                    "Batch processing completed",
                    batch_size=len(results),
                    successful=successful,
                    failed=failed,
                    skipped=skipped,
                    actually_processed=actually_processed,
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
    finally:
        # Save shutdown logs with stats
        shutdown_stats = {
            'processed_count': getattr(orchestrator, 'processed_count', 0),
            'failed_count': getattr(orchestrator, 'failed_count', 0),
            'start_time': getattr(orchestrator, 'start_time', None),
            'shutdown_reason': 'normal' if not hasattr(orchestrator, '_shutdown_error') else 'error'
        }
        
        if hasattr(orchestrator, 'start_time') and orchestrator.start_time:
            shutdown_stats['runtime_seconds'] = (datetime.now() - orchestrator.start_time).total_seconds()
        
        logger.info("Pipeline shutting down", **shutdown_stats)
        log_file = save_shutdown_logs(shutdown_stats)
        
        if log_file:
            print(f"📊 Final stats: {orchestrator.processed_count} processed, {orchestrator.failed_count} failed")
            print(f"📝 Detailed logs saved to: {log_file}")
        
        if hasattr(orchestrator, '_shutdown_error'):
            sys.exit(1)

if __name__ == "__main__":
    main()