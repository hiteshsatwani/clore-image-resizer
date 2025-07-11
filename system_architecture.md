# Image Processing System Architecture

## Overview
Automated system for processing product images when stores onboard to the platform. Converts images to 9:16 aspect ratio and uploads to S3.

## Components

### 1. Message Queue (SQS)
- **Purpose**: Decouples image processing from store onboarding
- **Message Format**: `{"product_id": "12345", "store_id": "store_abc", "timestamp": "2024-01-01T10:00:00Z"}`
- **Visibility Timeout**: 900 seconds (15 minutes)
- **Dead Letter Queue**: For failed processing attempts

### 2. Image Processor Service (EC2 g5.4xlarge)
- **Purpose**: Processes SQS messages and resizes images
- **Instance Type**: g5.4xlarge (16 vCPUs, 64GB RAM, 24GB GPU)
- **Scaling**: Auto Scaling Group (1-5 instances based on queue depth)
- **Processing**: SDXL inpainting for high-quality 9:16 conversion

### 3. Database Layer
- **Read**: Product images array from products table
- **Write**: Updated S3 URLs in same order as original images
- **Connection**: Connection pooling for high throughput

### 4. S3 Storage
- **Bucket Structure**: `processed-images/{store_id}/{product_id}/image_{index}.jpg`
- **Lifecycle**: Delete original temp files after 30 days
- **CDN**: CloudFront for fast global delivery

## Data Flow

1. **Trigger**: Store onboarding → SQS message
2. **Consumer**: Poll SQS → Get product_id
3. **Fetch**: Query DB → Get image URLs array
4. **Process**: Download → Resize → Upload to S3
5. **Update**: Replace URLs in DB (maintain order)
6. **Complete**: Delete SQS message

## Scalability Considerations

- **Queue Depth Monitoring**: Auto-scale based on messages
- **GPU Utilization**: Monitor VRAM usage for optimal batching
- **Database Connections**: Connection pooling to prevent exhaustion
- **S3 Rate Limits**: Implement exponential backoff for uploads

## Error Handling

- **Transient Failures**: Retry with exponential backoff
- **Permanent Failures**: Send to Dead Letter Queue
- **Partial Success**: Track progress per image for resume capability
- **Database Rollback**: Revert URL updates on S3 upload failure