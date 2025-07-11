"""SQS service for consuming product processing messages."""

import json
import boto3
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

from config import config
from logger import logger

@dataclass
class ProcessingMessage:
    """SQS message structure for product processing."""
    product_id: str
    store_id: Optional[str] = None
    priority: str = "normal"
    timestamp: Optional[str] = None
    retry_count: int = 0
    receipt_handle: Optional[str] = None
    message_id: Optional[str] = None
    
    @classmethod
    def from_sqs_message(cls, sqs_message: Dict[str, Any]) -> 'ProcessingMessage':
        """Create ProcessingMessage from SQS message."""
        try:
            body = json.loads(sqs_message['Body'])
            return cls(
                product_id=body['product_id'],
                store_id=body.get('store_id'),
                priority=body.get('priority', 'normal'),
                timestamp=body.get('timestamp'),
                retry_count=body.get('retry_count', 0),
                receipt_handle=sqs_message['ReceiptHandle'],
                message_id=sqs_message['MessageId']
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse SQS message", error=str(e), message=sqs_message)
            raise ValueError(f"Invalid message format: {e}")

class SQSService:
    """Service for SQS operations."""
    
    def __init__(self):
        self.sqs_client = boto3.client('sqs', region_name=config.aws_region)
        self.queue_url = config.sqs_queue_url
        self.visibility_timeout = config.visibility_timeout
    
    @retry(
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def receive_messages(self, max_messages: int = 10) -> List[ProcessingMessage]:
        """Receive messages from SQS queue."""
        try:
            response = self.sqs_client.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=min(max_messages, 10),  # SQS limit is 10
                WaitTimeSeconds=20,  # Long polling
                VisibilityTimeoutSeconds=self.visibility_timeout,
                MessageAttributeNames=['All']
            )
            
            messages = response.get('Messages', [])
            processing_messages = []
            
            for message in messages:
                try:
                    processing_message = ProcessingMessage.from_sqs_message(message)
                    processing_messages.append(processing_message)
                except ValueError as e:
                    logger.error("Skipping invalid message", error=str(e))
                    # Delete invalid message
                    self.delete_message(message['ReceiptHandle'])
            
            logger.info(
                "Received messages from SQS",
                message_count=len(processing_messages),
                queue_url=self.queue_url
            )
            
            return processing_messages
            
        except Exception as e:
            logger.error("Failed to receive messages from SQS", error=str(e))
            raise
    
    @retry(
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def delete_message(self, receipt_handle: str) -> bool:
        """Delete message from SQS queue."""
        try:
            self.sqs_client.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            
            logger.info("Deleted message from SQS", receipt_handle=receipt_handle)
            return True
            
        except Exception as e:
            logger.error("Failed to delete message from SQS", error=str(e))
            return False
    
    @retry(
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def change_message_visibility(self, receipt_handle: str, visibility_timeout: int) -> bool:
        """Change message visibility timeout."""
        try:
            self.sqs_client.change_message_visibility(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=visibility_timeout
            )
            
            logger.info(
                "Changed message visibility",
                receipt_handle=receipt_handle,
                visibility_timeout=visibility_timeout
            )
            return True
            
        except Exception as e:
            logger.error("Failed to change message visibility", error=str(e))
            return False
    
    def extend_message_visibility(self, receipt_handle: str, additional_seconds: int = 300) -> bool:
        """Extend message visibility timeout."""
        return self.change_message_visibility(receipt_handle, additional_seconds)
    
    @retry(
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def get_queue_attributes(self) -> Dict[str, Any]:
        """Get queue attributes."""
        try:
            response = self.sqs_client.get_queue_attributes(
                QueueUrl=self.queue_url,
                AttributeNames=['All']
            )
            
            return response.get('Attributes', {})
            
        except Exception as e:
            logger.error("Failed to get queue attributes", error=str(e))
            return {}
    
    def get_queue_depth(self) -> int:
        """Get approximate number of messages in queue."""
        attributes = self.get_queue_attributes()
        return int(attributes.get('ApproximateNumberOfMessages', 0))
    
    def get_in_flight_messages(self) -> int:
        """Get number of messages currently being processed."""
        attributes = self.get_queue_attributes()
        return int(attributes.get('ApproximateNumberOfMessagesNotVisible', 0))
    
    @retry(
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def send_message(self, product_id: str, store_id: Optional[str] = None, priority: str = "normal") -> bool:
        """Send message to SQS queue (for testing purposes)."""
        try:
            message_body = {
                "product_id": product_id,
                "store_id": store_id,
                "priority": priority,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "retry_count": 0
            }
            
            self.sqs_client.send_message(
                QueueUrl=self.queue_url,
                MessageBody=json.dumps(message_body)
            )
            
            logger.info("Sent message to SQS", product_id=product_id)
            return True
            
        except Exception as e:
            logger.error("Failed to send message to SQS", error=str(e))
            return False
    
    def poll_messages(self, batch_size: int = 10) -> List[ProcessingMessage]:
        """Poll for messages with error handling."""
        try:
            messages = self.receive_messages(batch_size)
            
            if messages:
                logger.info(
                    "Polling received messages",
                    message_count=len(messages),
                    queue_depth=self.get_queue_depth()
                )
            
            return messages
            
        except Exception as e:
            logger.error("Error polling messages", error=str(e))
            return []
    
    def process_message_batch(self, messages: List[ProcessingMessage]) -> List[ProcessingMessage]:
        """Process a batch of messages and return successfully processed ones."""
        successful_messages = []
        
        for message in messages:
            try:
                # Extend visibility timeout for processing
                self.extend_message_visibility(message.receipt_handle, 900)
                successful_messages.append(message)
                
            except Exception as e:
                logger.error(
                    "Failed to prepare message for processing",
                    product_id=message.product_id,
                    error=str(e)
                )
        
        return successful_messages
    
    def cleanup_processed_messages(self, messages: List[ProcessingMessage]) -> int:
        """Delete successfully processed messages."""
        deleted_count = 0
        
        for message in messages:
            if self.delete_message(message.receipt_handle):
                deleted_count += 1
        
        logger.info("Cleaned up processed messages", deleted_count=deleted_count)
        return deleted_count