from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="torchzq",
    python_requires=">=3.9.0",
    version="1.0.9",
    description="TorchZQ: A PyTorch experiment runner.",
    author="enhuiz",
    author_email="niuzhe.nz@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "zouqi==1.0.10",
        "pyyaml",
        "natsort",
        "tqdm",
        "pandas",
        "deprecated",
        "wandb",
    ],
    url="https://github.com/enhuiz/torchzq",
    entry_points={
        "console_scripts": [
            "tzq=torchzq.cli:main",
        ],
    },
)
