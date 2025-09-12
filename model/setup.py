#!/usr/bin/env python3
"""Setup script for development convenience"""

from setuptools import setup, find_packages

setup(
    name="crescendai-model",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "jax[cpu]>=0.4.13,<0.4.20",
        "flax>=0.7.2,<0.8.0",
        "optax>=0.1.7,<0.2.0",
        "numpy>=1.24.0,<2.0.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "modal>=1.1.4",
        "fastapi>=0.104.0",
        "pydantic>=2.5.0",
        "requests>=2.31.0"
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "pytest>=7.4.0", 
            "ruff>=0.1.0",
        ]
    }
)