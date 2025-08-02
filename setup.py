#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ahot",
    version="1.0.0",
    author="Adrian Paredez",
    author_email="adrian.paredez@example.com",
    description="Adaptive Hardware-Oriented Tokenizer (AHOT) - A tokenizer that adapts to hardware constraints",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/paredezadrian/ahot",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipykernel>=6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ahot-benchmark=src.benchmark.benchmarker:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="tokenizer, hardware, optimization, nlp, machine learning",
    project_urls={
        "Bug Reports": "https://github.com/paredezadrian/ahot/issues",
        "Source": "https://github.com/paredezadrian/ahot",
        "Documentation": "https://ahot.readthedocs.io/",
    },
) 