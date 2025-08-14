"""Database service for managing product image URLs."""

import psycopg2
import psycopg2.pool
from typing import List, Optional, Tuple
from contextlib import contextmanager
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential

from config import config
from logger import logger

@dataclass
class Product:
    """Product data structure."""
    product_id: str
    images: List[str]
    title: Optional[str] = None
    
class DatabaseService:
    """Service for database operations."""
    
    def __init__(self):
        self.connection_pool = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize database connection pool."""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=5,
                dsn=config.db_connection_string
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error("Failed to initialize database connection pool", error=str(e))
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error("Database connection error", error=str(e))
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    @retry(
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def get_product_images(self, product_id: str) -> Optional[Product]:
        """Retrieve product images from database."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT productid, images, title FROM products WHERE productid = %s",
                    (product_id,)
                )
                result = cursor.fetchone()
                
                if result:
                    return Product(
                        product_id=result[0],
                        images=result[1] if result[1] else [],
                        title=result[2] if result[2] else None
                    )
                return None
    
    @retry(
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def update_product_images(self, product_id: str, new_images: List[str]) -> bool:
        """Update product images in database."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE products SET images = %s WHERE productid = %s",
                    (new_images, product_id)
                )
                
                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(
                        "Updated product images",
                        product_id=product_id,
                        image_count=len(new_images)
                    )
                    return True
                else:
                    logger.warning("No product found to update", product_id=product_id)
                    return False
    
    @retry(
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def backup_original_images(self, product_id: str, original_images: List[str]) -> bool:
        """Backup original images before processing."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Check if column exists, if not create it
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'products' AND column_name = 'original_images'
                """)
                
                if not cursor.fetchone():
                    cursor.execute("""
                        ALTER TABLE products 
                        ADD COLUMN original_images text[]
                    """)
                    conn.commit()
                
                # Update original images
                cursor.execute(
                    "UPDATE products SET original_images = %s WHERE productid = %s",
                    (original_images, product_id)
                )
                
                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(
                        "Backed up original images",
                        product_id=product_id,
                        image_count=len(original_images)
                    )
                    return True
                return False
    
    @retry(
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def get_processing_status(self, product_id: str) -> Optional[str]:
        """Get processing status for a product."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Check if status column exists, if not create it
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'products' AND column_name = 'processing_status'
                """)
                
                if not cursor.fetchone():
                    cursor.execute("""
                        ALTER TABLE products 
                        ADD COLUMN processing_status varchar(50) DEFAULT 'pending'
                    """)
                    conn.commit()
                
                cursor.execute(
                    "SELECT processing_status FROM products WHERE productid = %s",
                    (product_id,)
                )
                result = cursor.fetchone()
                return result[0] if result else None
    
    @retry(
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def update_processing_status(self, product_id: str, status: str) -> bool:
        """Update processing status for a product."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE products SET processing_status = %s WHERE productid = %s",
                    (status, product_id)
                )
                
                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(
                        "Updated processing status",
                        product_id=product_id,
                        status=status
                    )
                    return True
                return False
    
    @retry(
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def update_product_tags(self, product_id: str, tags: List[str], gender: str, category: str) -> bool:
        """Update product tags, gender, and category in database."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Check if columns exist, if not create them
                columns_to_check = [
                    ("tags", "text[]"),
                    ("gender", "varchar(20)"),
                    ("category", "varchar(100)")
                ]
                
                for column_name, column_type in columns_to_check:
                    cursor.execute("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = 'products' AND column_name = %s
                    """, (column_name,))
                    
                    if not cursor.fetchone():
                        cursor.execute(f"""
                            ALTER TABLE products 
                            ADD COLUMN {column_name} {column_type}
                        """)
                        conn.commit()
                
                # Update tags, gender, and category
                cursor.execute("""
                    UPDATE products 
                    SET tags = %s, gender = %s, category = %s 
                    WHERE productid = %s
                """, (tags, gender, category, product_id))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(
                        "Updated product tags",
                        product_id=product_id,
                        tags=tags,
                        gender=gender,
                        category=category
                    )
                    return True
                else:
                    logger.warning("No product found to update tags", product_id=product_id)
                    return False
    
    def close(self):
        """Close database connection pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Database connection pool closed")