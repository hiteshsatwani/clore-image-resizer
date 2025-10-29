"""FastAPI backend for AI-powered 9:16 image resizer."""

import io
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from image_resizer import ImageResizer

# Initialize FastAPI app
app = FastAPI(
    title="Clore Image Resizer",
    description="AI-powered image resizer to 9:16 aspect ratio",
    version="1.0.0"
)

# Initialize image resizer on startup
resizer: Optional[ImageResizer] = None

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Using device: {DEVICE}")


def initialize_resizer():
    """Lazy load the image resizer (called on first request)."""
    global resizer
    if resizer is None:
        try:
            print("📦 Loading AI model (this may take 2-5 minutes on first run)...")
            resizer = ImageResizer(device=DEVICE)
            print("✅ AI model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load AI model: {e}")
            raise


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
        "device": DEVICE,
        "model_loaded": resizer is not None,
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/resize")
async def resize_image(file: UploadFile = File(...)):
    """
    Resize image to 9:16 aspect ratio.

    Accepts image files (jpg, png, webp, etc.)
    Returns resized image as PNG.
    """
    # Lazy load model on first request
    initialize_resizer()

    # Validate file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file (jpg, png, webp, etc.)"
        )

    try:
        # Read uploaded file
        contents = await file.read()

        # Validate file size (max 50MB)
        if len(contents) > 50 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="File too large. Maximum 50MB allowed."
            )

        # Open image
        input_image = Image.open(io.BytesIO(contents))

        # Validate image dimensions
        width, height = input_image.size
        if width < 100 or height < 100:
            raise HTTPException(
                status_code=400,
                detail="Image too small. Minimum 100x100 pixels required."
            )

        if width > 8000 or height > 8000:
            raise HTTPException(
                status_code=400,
                detail="Image too large. Maximum 8000x8000 pixels."
            )

        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_input:
            input_path = tmp_input.name
            input_image.save(input_path, "PNG")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_output:
            output_path = tmp_output.name

        try:
            # Process image
            print(f"🔄 Processing image: {file.filename} ({width}x{height})")
            success = resizer.resize_image(input_path, output_path)

            if not success:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to process image. The image may be too complex."
                )

            # Return resized image
            print(f"✅ Successfully resized: {file.filename}")
            return FileResponse(
                output_path,
                media_type="image/png",
                filename=f"{Path(file.filename).stem}_resized.png"
            )

        finally:
            # Clean up input file (output will be cleaned up by FastAPI)
            try:
                os.unlink(input_path)
            except:
                pass

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.get("/status")
async def get_status():
    """Get current service status."""
    try:
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            return {
                "status": "ready",
                "device": DEVICE,
                "gpu_memory_gb": gpu_memory_gb,
                "model_loaded": resizer is not None
            }
        else:
            return {
                "status": "ready",
                "device": "cpu",
                "model_loaded": resizer is not None
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    import uvicorn

    # Determine port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))

    # Determine host - bind to 0.0.0.0 for Azure deployments
    host = "0.0.0.0"

    print(f"🚀 Starting Clore Image Resizer API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
