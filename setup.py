from setuptools import setup

setup(
    name="torchzq",
    python_requires=">=3.6.0",
    version="1.0",
    description="TorchZQ: A simple PyTorch experiment runner.",
    author="enhuiz",
    author_email="niuzhe.nz@outlook.com",
    packages=["torchzq"],
    scripts=["zouqi", "zqboard"],
    install_requires=["torch", "tensorboardX", "tensorboard"],
)
