from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="torchzq",
    python_requires=">=3.9.0",
    version="1.0.9.dev1",
    description="TorchZQ: A simple PyTorch experiment runner.",
    author="enhuiz",
    author_email="niuzhe.nz@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    scripts=["tzq"],
    install_requires=[
        "torch",
        "zouqi==1.0.9.dev0",
        "pyyaml",
        "tqdm",
        "pandas",
        "torchvision",
        "natsort",
        "deprecated",
        "wandb",
    ],
    url="https://github.com/enhuiz/torchzq",
)
