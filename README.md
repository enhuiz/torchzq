# TorchZQ: A simple PyTorch experiment runner.

> Zouqi (『走起』 in Chinese) means "let's go". When you zouqi your experiment, the experiment will go with you.

## Installation

```
pip install git+https://github.com/enhuiz/torchzq
```

or

```
$ git clone https://github.com/enhuiz/torchzq
$ cd torchzq
$ pip install .
```

## Run an Example

**Training**

```
$ zouqi example/config/mnist.yml train
```


**Testing**

```
$ zouqi example/config/mnist.yml test
```

![](example/animation.gif)


**TensorBoard**

```
$ zqboard --logdir .
```

## Supported Features

- [x] Model checkpoints
- [x] Logging
- [x] Gradient accumulation
- [x] Configuration file
- [x] Configuration file inheritance
- [x] TensorBoard
- [x] (c)GAN training (WGAN-GP)
- [x] Automatic mixed precision 
