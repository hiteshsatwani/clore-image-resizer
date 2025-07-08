#!/usr/bin/env python3
"""Setup script for AI Image Resizer."""

from setuptools import setup, find_packages

setup(
    name="clore-image-resizer",
    version="1.0.0",
    description="AI-powered image resizer to 9:16 aspect ratio",
    author="Clore AI",
    python_requires=">=3.8",
    py_modules=["image_resizer"],
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "diffusers>=0.25.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "safetensors>=0.4.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
    ],
    entry_points={
        "console_scripts": [
            "clore-resize=image_resizer:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)