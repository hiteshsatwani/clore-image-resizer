"""Main orchestrator for the image processing pipeline."""

import time
import signal
import sys
import os
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import gc
import threading
from datetime import datetime, timedelta
from collections import deque
import shutil

from config import config
from logger import logger
from sqs_service import SQSService, ProcessingMessage
from database_service import DatabaseService, Product
from s3_service import S3Service
from image_resizer import ImageResizer
from gpt_fashion_tagger import GPTFashionTagger
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

@dataclass
class LogEntry:
    """A log entry for the UI display."""
    timestamp: datetime
    level: str
    product_id: Optional[str]
    message: str
    details: Optional[Dict[str, Any]] = None

class PipelineOrchestrator:
    """Main orchestrator for the image processing pipeline."""
    
    def __init__(self):
        self.sqs_service = SQSService()
        self.db_service = DatabaseService()
        self.s3_service = S3Service()
        self.image_resizer = ImageResizer(device=config.device)
        self.image_tagger = GPTFashionTagger()
        
        self.running = True
        self.processed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.total_images_processed = 0
        self.current_product_id = None
        self.current_stage = "Initializing"
        self.current_batch_size = 0
        self.current_batch_progress = 0
        self.recent_processing_times = []
        self.last_activity = time.time()
        self.start_time = time.time()
        
        # UI and logging
        self.log_buffer = deque(maxlen=100)  # Store last 100 log entries
        self.ui_lock = threading.Lock()
        self.terminal_width = shutil.get_terminal_size().columns
        self.terminal_height = shutil.get_terminal_size().lines
        
        # Detailed progress tracking
        self.current_image_index = 0
        self.current_image_total = 0
        self.current_operation = ""
        self.detailed_progress = {}
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Start UI update thread
        self.ui_thread = threading.Thread(target=self._update_ui, daemon=True)
        self.ui_thread.start()
        
        # Add initial log entry
        self._add_log_entry("INFO", None, "Pipeline orchestrator initialized")
        logger.info("Pipeline orchestrator initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self._add_log_entry("WARN", None, f"Received signal {signum}, shutting down gracefully...")
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def _add_log_entry(self, level: str, product_id: Optional[str], message: str, details: Optional[Dict[str, Any]] = None):
        """Add a log entry to the buffer for UI display."""
        with self.ui_lock:
            entry = LogEntry(
                timestamp=datetime.now(),
                level=level,
                product_id=product_id,
                message=message,
                details=details
            )
            self.log_buffer.append(entry)
    
    def _get_level_color(self, level: str) -> str:
        """Get color code for log level."""
        colors = {
            "INFO": "🔵",
            "WARN": "🟡", 
            "ERROR": "🔴",
            "SUCCESS": "🟢",
            "DEBUG": "⚪"
        }
        return colors.get(level, "⚪")
    
    def _format_log_entry(self, entry: LogEntry, max_width: int) -> str:
        """Format a log entry for display."""
        timestamp = entry.timestamp.strftime("%H:%M:%S")
        color = self._get_level_color(entry.level)
        
        # Format product ID
        product_part = ""
        if entry.product_id:
            product_part = f" [{entry.product_id[:8]}...]"
        
        # Truncate message if too long
        available_width = max_width - len(timestamp) - len(product_part) - 6  # Account for color and spacing
        if len(entry.message) > available_width:
            message = entry.message[:available_width-3] + "..."
        else:
            message = entry.message
        
        return f"{color} {timestamp}{product_part} {message}"
    
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
        """Update the terminal UI continuously with improved layout and live logs."""
        while self.running:
            try:
                # Update terminal dimensions
                self.terminal_width = shutil.get_terminal_size().columns
                self.terminal_height = shutil.get_terminal_size().lines
                
                self._clear_screen()
                
                # Calculate layout
                width = min(self.terminal_width, 120)  # Max width for readability
                log_height = max(10, self.terminal_height - 25)  # Reserve space for other sections
                
                # Header with enhanced styling
                header = "🚀 CLORE IMAGE PROCESSING PIPELINE"
                padding = (width - len(header)) // 2
                print("═" * width)
                print(" " * padding + header)
                print("═" * width)
                
                # Status section - compact layout
                uptime = time.time() - self.start_time
                queue_depth = self.sqs_service.get_queue_depth()
                in_flight = self.sqs_service.get_in_flight_messages()
                
                status_color = "🟢" if self.running else "🔴"
                current_time = datetime.now().strftime("%H:%M:%S")
                
                # Top status bar
                print(f"\n{status_color} RUNNING │ ⏱️  {self._format_duration(uptime)} │ 📊 Queue: {queue_depth} │ ✈️  Flight: {in_flight} │ 🕒 {current_time}")
                
                # Statistics in a compact grid
                total_processed = self.processed_count + self.failed_count + self.skipped_count
                success_rate = (self.processed_count / max(total_processed, 1)) * 100
                current_speed, overall_speed, eta = self._get_speed_stats()
                
                print(f"✅ Success: {self.processed_count:4d} │ ❌ Failed: {self.failed_count:3d} │ ⏭️  Skipped: {self.skipped_count:3d} │ 🖼️  Images: {self.total_images_processed:5d} │ 📊 Rate: {success_rate:5.1f}%")
                print(f"⚡ Speed: {overall_speed:.1f}/min │ 🎯 Current: {current_speed:.1f}/min │ ⏰ ETA: {eta}")
                
                # Current activity with detailed progress
                print(f"\n🔄 CURRENT ACTIVITY")
                print("─" * width)
                
                if self.current_product_id:
                    print(f"📦 Product: {self.current_product_id[:12]}... │ Stage: {self.current_stage}")
                    
                    if self.current_batch_size > 0:
                        batch_progress_bar = self._get_progress_bar(
                            self.current_batch_progress, 
                            self.current_batch_size,
                            width=30
                        )
                        print(f"📊 Batch: {batch_progress_bar} ({self.current_batch_progress}/{self.current_batch_size})")
                    
                    if self.current_image_total > 0:
                        image_progress_bar = self._get_progress_bar(
                            self.current_image_index,
                            self.current_image_total,
                            width=30
                        )
                        print(f"🖼️  Images: {image_progress_bar} ({self.current_image_index}/{self.current_image_total})")
                        if self.current_operation:
                            print(f"🔧 Operation: {self.current_operation}")
                else:
                    if queue_depth == 0:
                        print("⏳ Waiting for new messages...")
                    else:
                        print("🔍 Checking queue...")
                
                # Recent activity
                time_since_activity = time.time() - self.last_activity
                activity_text = f"{time_since_activity:.0f}s ago" if time_since_activity < 60 else f"{self._format_duration(time_since_activity)} ago"
                print(f"⏱️  Last activity: {activity_text}")
                
                # Live logs section
                print(f"\n📝 LIVE LOGS")
                print("─" * width)
                
                with self.ui_lock:
                    # Get recent log entries
                    recent_logs = list(self.log_buffer)[-log_height:]
                    
                    if recent_logs:
                        for entry in recent_logs:
                            log_line = self._format_log_entry(entry, width - 2)
                            print(f" {log_line}")
                    else:
                        print(" No logs yet...")
                
                # Fill remaining space if needed
                current_line_count = 15 + len(recent_logs)  # Approximate line count
                remaining_lines = max(0, self.terminal_height - current_line_count - 3)
                for _ in range(remaining_lines):
                    print()
                
                # Bottom status bar
                print("═" * width)
                print(f"💡 Press Ctrl+C to stop gracefully │ Logs: {len(self.log_buffer)}/100 │ Terminal: {self.terminal_width}x{self.terminal_height}")
                
                time.sleep(1)  # Update every second for more responsive logs
                
            except Exception as e:
                # Don't let UI errors crash the main process
                time.sleep(5)
    
    def _update_current_stage(self, stage: str, product_id: str = None):
        """Update current processing stage and add log entry."""
        self.current_stage = stage
        if product_id:
            self.current_product_id = product_id
        self.last_activity = time.time()
        
        # Add to log buffer
        self._add_log_entry("INFO", product_id, stage)
    
    def _update_detailed_progress(self, operation: str, current: int = 0, total: int = 0):
        """Update detailed progress for current operation."""
        self.current_operation = operation
        self.current_image_index = current
        self.current_image_total = total
        
        if total > 0:
            progress_msg = f"{operation} ({current}/{total})"
        else:
            progress_msg = operation
            
        self._add_log_entry("DEBUG", self.current_product_id, progress_msg)
    
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
                self._add_log_entry("INFO", message.product_id, "Already completed, skipping")
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
                self._add_log_entry("ERROR", message.product_id, "Product not found in database")
                logger.warning("Product not found in database", product_id=message.product_id)
                return ProcessingResult(
                    product_id=message.product_id,
                    success=False,
                    processed_images=0,
                    error="Product not found in database"
                )
            
            if not product.images:
                self._add_log_entry("ERROR", message.product_id, "No images found for product")
                logger.warning("No images found for product", product_id=message.product_id)
                return ProcessingResult(
                    product_id=message.product_id,
                    success=False,
                    processed_images=0,
                    error="No images found for product"
                )
            
            self._add_log_entry("INFO", message.product_id, f"Found {len(product.images)} images")
            
            # Backup original images
            self._update_current_stage("Backing up original images", message.product_id)
            self.db_service.backup_original_images(message.product_id, product.images)
            
            # Download images
            self._update_current_stage(f"Downloading {len(product.images)} images", message.product_id)
            logger.info("Downloading images", product_id=message.product_id, image_count=len(product.images))
            downloaded_images = self.s3_service.download_batch_images(product.images)
            
            # Tag ALL products first (before filtering) using original downloaded images
            try:
                self._update_current_stage("GPT tagging all product images", message.product_id)
                self._add_log_entry("INFO", message.product_id, "Starting GPT tagging process for all images")
                logger.info("Tagging all product images with GPT", product_id=message.product_id)
                
                # Get all successfully downloaded images for tagging
                all_downloaded_images = [img for img in downloaded_images if img is not None]
                
                if all_downloaded_images:
                    # Use the product title we already fetched from the database
                    product_name = product.title or f"Product {message.product_id}"
                    
                    # Run GPT tagger on ALL downloaded images
                    tag_result = self.image_tagger.tag_product(
                        images=all_downloaded_images,
                        product_name=product_name,
                        product_id=message.product_id
                    )
                    
                    if not tag_result.error:
                        # GPT tagger returns the exact format we need for the database
                        tags = tag_result.tags
                        gender = tag_result.gender
                        category = tag_result.category
                        aesthetics = tag_result.aesthetics
                        suitable_indices = tag_result.suitable_image_indices
                        
                        # Update database with tags and aesthetics
                        self.db_service.update_product_tags(message.product_id, tags, gender, category, aesthetics)
                        self._add_log_entry("SUCCESS", message.product_id, f"Tagged: {category} | {gender} | {len(tags)} tags | {len(aesthetics)} aesthetics | {len(suitable_indices)}/{len(all_downloaded_images)} suitable")
                        logger.info(
                            "Product tagged successfully with GPT",
                            product_id=message.product_id,
                            tags=tags,
                            gender=gender,
                            category=category,
                            aesthetics=aesthetics,
                            suitable_images=f"{len(suitable_indices)}/{len(all_downloaded_images)}",
                            processing_time=tag_result.processing_time
                        )
                    else:
                        self._add_log_entry("ERROR", message.product_id, f"GPT tagging failed: {tag_result.error}")
                        logger.warning("GPT tagging failed", product_id=message.product_id, error=tag_result.error)
                else:
                    self._add_log_entry("WARN", message.product_id, "No images available for tagging")
                    logger.warning("No images available for tagging", product_id=message.product_id)
                        
            except Exception as e:
                self._add_log_entry("ERROR", message.product_id, f"Tagging error: {str(e)}")
                logger.error("Error during tagging", product_id=message.product_id, error=str(e))
            
            # Now filter images for AI processing based on GPT's per-image analysis
            self._update_current_stage("Selecting images for AI processing", message.product_id)
            valid_images = []
            valid_urls = []
            
            # Use GPT's suitability decisions if available, otherwise use all downloaded images
            if 'suitable_indices' in locals() and not tag_result.error:
                # Use GPT's per-image decisions
                for i, (img, url) in enumerate(zip(downloaded_images, product.images)):
                    if img is not None and i in suitable_indices:
                        valid_images.append(img)
                        valid_urls.append(url)
                        self._add_log_entry("SUCCESS", message.product_id, f"Image {i+1} selected by GPT for AI processing")
                        logger.info("Image selected by GPT for AI processing", url=url[:100], product_id=message.product_id)
                    elif img is not None:
                        self._add_log_entry("INFO", message.product_id, f"Image {i+1} not suitable for AI processing (per GPT)")
                        logger.info("Image not suitable for AI processing per GPT", url=url[:100], product_id=message.product_id)
                    else:
                        self._add_log_entry("ERROR", message.product_id, f"Image {i+1} download failed")
                        logger.warning("Failed to download image", url=url)
            else:
                # Fallback: use all downloaded images if GPT analysis failed
                self._add_log_entry("WARN", message.product_id, "Using all images as fallback (GPT analysis failed)")
                for i, (img, url) in enumerate(zip(downloaded_images, product.images)):
                    if img is not None:
                        valid_images.append(img)
                        valid_urls.append(url)
                    else:
                        self._add_log_entry("ERROR", message.product_id, f"Image {i+1} download failed")
                        logger.warning("Failed to download image", url=url)
            
            if not valid_images:
                downloaded_count = len([img for img in downloaded_images if img is not None])
                error_msg = f"No suitable images for AI processing (downloaded: {downloaded_count}, suitable: {len(valid_images)})"
                self._add_log_entry("WARN", message.product_id, f"Product skipped - {error_msg}")
                logger.warning(error_msg, product_id=message.product_id)
                
                # Mark as completed but with a note that it was skipped due to no suitable images
                self.db_service.update_processing_status(message.product_id, "completed")
                self.skipped_count += 1
                
                return ProcessingResult(
                    product_id=message.product_id,
                    success=True,  # Consider it successful since it was intentionally skipped
                    processed_images=0,
                    error=error_msg
                )
            
            self._add_log_entry("SUCCESS", message.product_id, f"Filtering complete: {len(valid_images)} suitable images")
            logger.info(
                "Image filtering completed", 
                product_id=message.product_id, 
                downloaded=len([img for img in downloaded_images if img is not None]),
                suitable=len(valid_images)
            )
            
            # Process images with AI resizer
            self._update_current_stage(f"AI processing {len(valid_images)} images", message.product_id)
            logger.info("Processing images with AI", product_id=message.product_id, image_count=len(valid_images))
            processed_images = []
            
            for i, image in enumerate(valid_images):
                try:
                    self._update_current_stage(f"AI processing image {i+1}/{len(valid_images)}", message.product_id)
                    self._update_detailed_progress(f"Processing image", i + 1, len(valid_images))
                    
                    # Scale to 1080p first
                    self._add_log_entry("INFO", message.product_id, f"Scaling image {i+1} to 1080p")
                    scaled_image = self.image_resizer.scale_to_1080p(image)
                    
                    # Calculate target dimensions
                    width, height = scaled_image.size
                    target_width, target_height = self.image_resizer.calculate_target_dimensions(width, height)
                    
                    # Skip if already correct ratio
                    if width == target_width and height == target_height:
                        self._add_log_entry("INFO", message.product_id, f"Image {i+1} already has correct aspect ratio")
                        processed_images.append(scaled_image)
                        continue
                    
                    # Create base canvas and mask
                    self._add_log_entry("INFO", message.product_id, f"Creating canvas and mask for image {i+1}")
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
                        
                        self._add_log_entry("INFO", message.product_id, f"Resizing image {i+1} for processing: {process_width}x{process_height}")
                        base_resized = base_canvas.resize((process_width, process_height), Image.LANCZOS)
                        mask_resized = mask.resize((process_width, process_height), Image.LANCZOS)
                    else:
                        base_resized = base_canvas
                        mask_resized = mask
                        process_width, process_height = target_width, target_height
                    
                    # Run AI inpainting
                    self._add_log_entry("INFO", message.product_id, f"Running AI inpainting for image {i+1}...")
                    self._update_detailed_progress(f"AI inpainting image", i + 1, len(valid_images))
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
                    
                    self._add_log_entry("SUCCESS", message.product_id, f"Image {i+1} AI processing completed")
                    processed_images.append(result)
                    
                    # Clear GPU memory
                    if hasattr(self.image_resizer, 'device') and 'cuda' in self.image_resizer.device and torch is not None:
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    self._add_log_entry("ERROR", message.product_id, f"Image {i+1} processing failed: {str(e)}")
                    logger.error("Failed to process image", product_id=message.product_id, image_index=i, error=str(e))
                    # Use original image if processing fails
                    processed_images.append(valid_images[i])
            
            # Upload processed images to S3
            self._update_current_stage(f"Uploading {len(processed_images)} images to S3", message.product_id)
            self._add_log_entry("INFO", message.product_id, f"Uploading {len(processed_images)} processed images to S3")
            logger.info("Uploading processed images", product_id=message.product_id, image_count=len(processed_images))
            s3_urls = self.s3_service.upload_batch_images(processed_images, message.product_id, valid_urls)
            
            # Update database with new URLs
            self._update_current_stage("Updating database with new URLs", message.product_id)
            self._add_log_entry("INFO", message.product_id, "Updating database with new image URLs")
            logger.info("Updating database with new URLs", product_id=message.product_id)
            self.db_service.update_product_images(message.product_id, s3_urls)
            
            self._update_current_stage("Finalizing product", message.product_id)
            self._add_log_entry("SUCCESS", message.product_id, f"Product processing completed - {len(processed_images)} images processed")
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
            self._add_log_entry("ERROR", message.product_id, f"Processing failed: {str(e)}")
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
            self._add_log_entry("INFO", None, f"Starting batch item {i+1}/{len(messages)}: {message.product_id[:8]}...")
            
            result = self.process_product(message)
            results.append(result)
            
            if result.success:
                self.processed_count += 1
                # Delete message from SQS
                self.sqs_service.delete_message(message.receipt_handle)
                if result.processed_images > 0:
                    self._add_log_entry("SUCCESS", message.product_id, f"Batch item {i+1} completed successfully")
                else:
                    self._add_log_entry("INFO", message.product_id, f"Batch item {i+1} skipped (no suitable images)")
            else:
                self.failed_count += 1
                self._add_log_entry("ERROR", message.product_id, f"Batch item {i+1} failed: {result.error}")
                # Keep message in queue for retry (will be retried after visibility timeout)
        
        # Reset batch tracking
        self.current_batch_size = 0
        self.current_batch_progress = 0
        self.current_product_id = None
        
        return results
    
    def run(self):
        """Main processing loop."""
        self._add_log_entry("SUCCESS", None, "Pipeline orchestrator starting up...")
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
                self._add_log_entry("INFO", None, f"Found {queue_depth} items in queue, starting batch processing...")
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
        self._add_log_entry("WARN", None, "Shutting down pipeline orchestrator...")
        logger.info("Shutting down pipeline orchestrator")
        
        # Calculate final statistics
        total_time = time.time() - self.start_time
        total_processed = self.processed_count + self.failed_count
        
        self._add_log_entry("INFO", None, f"Final stats: {self.processed_count} successful, {self.failed_count} failed, {self.skipped_count} skipped")
        self._add_log_entry("INFO", None, f"Total runtime: {self._format_duration(total_time)}")
        
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
        
        self._add_log_entry("SUCCESS", None, "Goodbye! Pipeline orchestrator shutdown complete.")
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