"""FastAPI backend for GPT-powered image tagging and recommendations."""

import io
import os
import sys
import json
from pathlib import Path
from typing import Optional, List

import requests
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, WebSocket, Body
from fastapi.responses import HTMLResponse
from PIL import Image

# Import batch processing services
try:
    from services.gpt_tagger_service import get_tagger
    from services.graphql_client_service import get_graphql_client
    print("✓ Batch processing services imported successfully")
except Exception as e:
    print(f"⚠️  Batch processing import warning: {e}")

print("🚀 Initializing Clore Image Tagging & Recommendation API...")

# Initialize FastAPI app
app = FastAPI(
    title="Clore Image Tagging & Recommendations",
    description="GPT-powered image tagging and product recommendations",
    version="1.0.0"
)
print("✓ FastAPI app created")

# Pydantic models
class ProcessBatchRequest(BaseModel):
    product_ids: List[str]

# WebSocket connection manager for dashboard
active_connections: set = set()


async def broadcast_message(level: str, product_id: str, message: str):
    """Broadcast a message to all connected dashboard clients."""
    msg_data = {
        "level": level,
        "product_id": product_id,
        "message": message
    }
    disconnected = set()

    for connection in active_connections:
        try:
            await connection.send_json(msg_data)
        except Exception as e:
            print(f"⚠️  Failed to send websocket message: {e}")
            disconnected.add(connection)

    # Remove disconnected clients
    active_connections.difference_update(disconnected)




@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    """Serve the homepage."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        with open(html_path, "r") as f:
            return f.read()
    else:
        return """
        <html>
            <body>
                <h1>🚀 Clore Image Resizer</h1>
                <p>Please create templates/index.html</p>
            </body>
        </html>
        """




@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "GPT Image Tagging & Recommendations",
        "gpu_required": False
    }




@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates."""
    await websocket.accept()
    active_connections.add(websocket)
    print(f"📊 Dashboard client connected. Active connections: {len(active_connections)}")

    try:
        # Keep the connection alive
        while True:
            data = await websocket.receive_text()
            # Clients can send ping messages to stay connected
            if data == "ping":
                await websocket.send_text("pong")
    except Exception as e:
        print(f"⚠️  WebSocket error: {e}")
    finally:
        active_connections.discard(websocket)
        print(f"📊 Dashboard client disconnected. Active connections: {len(active_connections)}")


@app.post("/process-batch")
async def process_batch(request: ProcessBatchRequest):
    """
    Process a batch of products for image tagging and recommendations.

    Fetches product images, runs GPT tagging, and generates AI recommendations.
    No GPU required - uses OpenAI and GPT services for analysis.

    Args:
        product_ids: List of product IDs to process

    Returns:
        Tagging and recommendation results for each product
    """
    if not request.product_ids:
        raise HTTPException(status_code=400, detail="No product IDs provided")

    try:
        tagger = get_tagger()
        graphql_client = get_graphql_client()

        results = {
            "total": len(request.product_ids),
            "successful": 0,
            "failed": 0,
            "products": {}
        }

        for product_id in request.product_ids:
            try:
                print(f"🔄 Processing product: {product_id}")
                await broadcast_message("info", product_id, f"Starting to process product {product_id}")

                # Get product images via GraphQL query
                try:
                    print(f"  🔍 Fetching product images from GraphQL...")
                    images = await graphql_client.get_product_images(product_id)
                    if not images:
                        print(f"  ⚠️  No images found for product")
                        results["products"][product_id] = {
                            "status": "failed",
                            "error": "No images found"
                        }
                        results["failed"] += 1
                        continue
                except Exception as e:
                    print(f"  ⚠️  Failed to fetch product images: {e}")
                    results["products"][product_id] = {
                        "status": "failed",
                        "error": f"Failed to fetch images: {str(e)}"
                    }
                    results["failed"] += 1
                    continue

                print(f"  📥 Downloaded {len(images)} images")

                # Download images
                downloaded_images = []
                for i, img_url in enumerate(images):
                    try:
                        response = requests.get(img_url, timeout=10)
                        if response.status_code == 200:
                            img = Image.open(io.BytesIO(response.content))
                            downloaded_images.append(img.convert("RGB"))
                    except Exception as e:
                        print(f"  ⚠️  Failed to download image {i+1}: {e}")

                if not downloaded_images:
                    results["products"][product_id] = {
                        "status": "failed",
                        "error": "Failed to download images"
                    }
                    results["failed"] += 1
                    continue

                # Run GPT tagging
                print(f"  🤖 Running GPT tagger...")
                await broadcast_message("info", product_id, "Running GPT tagger...")
                tag_result = tagger.tag_product(
                    images=downloaded_images,
                    product_name=f"Product {product_id}",
                    product_id=product_id
                )

                if not tag_result.error:
                    # Update database with tags via GraphQL
                    tag_update_success = await graphql_client.update_product_metadata(
                        product_id=product_id,
                        tags=tag_result.tags,
                        gender=tag_result.gender,
                        category=tag_result.category,
                        aesthetics=tag_result.aesthetics
                    )
                    if tag_update_success:
                        category_name = tag_result.category.get('specific')
                        await broadcast_message("success", product_id, f"Tagged: {category_name}")
                        print(f"  ✓ Tagged: {category_name}")
                    else:
                        print(f"  ⚠️  Failed to update product tags")

                results["products"][product_id] = {
                    "status": "success",
                    "category": tag_result.category.get("specific"),
                    "gender": tag_result.gender,
                    "aesthetics": tag_result.aesthetics,
                    "tags": tag_result.tags,
                    "total_images": len(downloaded_images)
                }
                results["successful"] += 1

                # Broadcast final product stats
                await broadcast_message("stats", product_id, {
                    "status": "completed",
                    "category": tag_result.category.get("specific"),
                    "gender": tag_result.gender,
                    "total_images": len(downloaded_images),
                    "progress_percent": 100
                })

                await broadcast_message("success", product_id, f"✅ Completed: Tagged with {tag_result.category.get('specific')}")
                print(f"✅ Completed product: {product_id} - Tagged successfully")

            except Exception as e:
                print(f"❌ Error processing {product_id}: {e}")
                await broadcast_message("error", product_id, f"Error: {str(e)}")
                results["products"][product_id] = {
                    "status": "failed",
                    "error": str(e)
                }
                results["failed"] += 1

        return results

    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@app.get("/status")
async def get_status():
    """Get current service status."""
    return {
        "status": "ready",
        "service": "GPT Image Tagging & Recommendations",
        "gpu_required": False,
        "capabilities": [
            "Image tagging with GPT vision",
            "Product category classification",
            "Gender and aesthetic analysis",
            "AI-powered recommendations"
        ]
    }


if __name__ == "__main__":
    import uvicorn

    # Determine port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))

    # Determine host - bind to 0.0.0.0 for Azure deployments
    host = "0.0.0.0"

    print(f"🚀 Starting Clore Image Tagging & Recommendation API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
