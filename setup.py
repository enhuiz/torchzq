from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="torchzq",
    python_requires=">=3.6.0",
    version="1.0.3",
    description="TorchZQ: A simple PyTorch experiment runner.",
    author="enhuiz",
    author_email="niuzhe.nz@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["torchzq", "torchzq/runners"],
    scripts=["zouqi", "zqboard"],
    install_requires=["torch", "zouqi==1.0.4", "pyyaml"],
    url="https://github.com/enhuiz/torchzq",
)
