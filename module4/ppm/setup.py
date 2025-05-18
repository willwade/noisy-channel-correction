"""
Setup script for the enhanced PPM package.
"""

from setuptools import setup, find_packages

setup(
    name="enhanced-ppm",
    version="0.1.0",
    description="Enhanced PPM (Prediction by Partial Matching) for AAC",
    author="Will Wade",
    author_email="wwade@acecentre.org.uk",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "gradio",
        "requests",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
