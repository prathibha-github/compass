"""Setup configuration for compass."""
from setuptools import find_packages, setup

with open("compass/_version.py") as f:
    version_line = [line for line in f if line.startswith("__version__")][0]
    version = version_line.split("=")[1].strip().strip('"')

with open("README.md") as f:
    long_description = f.read()

setup(
    name="compass-eval",
    version=version,
    description="Evaluation framework for subjective model behavior",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Prathibha Alam",
    url="https://github.com/prathibha-github/compass",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=2.0",
    ],
    extras_require={
        "openai": ["openai>=1.0"],
        "anthropic": ["anthropic>=0.7"],
        "google": ["google-genai>=0.1.0"],
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "isort>=5.0",
            "mypy>=1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
