from setuptools import setup

setup(
    name="torchzq",
    python_requires=">=3.6.0",
    version="1.1",
    description="TorchZQ: A simple PyTorch experiment runner.",
    author="enhuiz",
    author_email="niuzhe.nz@outlook.com",
    packages=["torchzq", "torchzq/runners"],
    scripts=["zouqi", "zqboard"],
    install_requires=["torch", "tensorboardX", "tensorboard"],
)
