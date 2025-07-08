# AI Image Resizer

High-performance CLI tool for resizing images to 9:16 aspect ratio using advanced AI inpainting techniques.

## Features

- **AI-powered extension**: Uses Stable Diffusion XL for realistic image extension
- **Smart aspect ratio handling**: Automatically calculates optimal dimensions
- **Context-aware inpainting**: Generates seamless extensions based on image content
- **Batch processing**: Process multiple images at once
- **GPU optimized**: Designed for A100 40GB VRAM with memory efficiency
- **Fast processing**: 1-5 second processing time per image

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Single Image
```bash
python image_resizer.py input.jpg -o output.jpg
```

### Batch Processing
```bash
python image_resizer.py /path/to/images/ -b -o /path/to/output/
```

### Options
- `-o, --output`: Output path or directory
- `-d, --device`: Device to use (cuda/cpu, default: cuda)
- `-b, --batch`: Enable batch processing for directories
- `-q, --quiet`: Suppress output messages

## Performance

- **Target processing time**: 1-5 seconds per image on A100 40GB
- **Memory efficient**: Uses model CPU offloading and VAE slicing
- **Optimized inference**: 20 inference steps for speed/quality balance

## Technical Details

- **Primary model**: Stable Diffusion XL Inpainting
- **Depth estimation**: Intel DPT-Large for content analysis
- **Image processing**: PIL with LANCZOS resampling
- **Target aspect ratio**: 9:16 (vertical)
- **Processing resolution**: 1024x1024 for optimal SDXL performance

## Requirements

- Python 3.8+
- CUDA-compatible GPU (A100 recommended)
- 8GB+ VRAM minimum (40GB recommended)
- PyTorch 2.0+