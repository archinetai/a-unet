# A-UNet

A library that provides building blocks to customize UNets, in PyTorch.

## Install
```bash
pip install a-unet
```

[![PyPI - Python Version](https://img.shields.io/pypi/v/a-unet?style=flat&colorA=black&colorB=black)](https://pypi.org/project/a-unet/)


## Usage

### Basic UNet
This UNet [`examples/unet_basic.py`](examples/unet_basic.py) shows how build a convolutional only UNet, using A-UNet blocks.
```python:examples/unet_basic.py

```

#### Example
```
unet = UNet(dim=2, in_channels=8, channels=[256, 512], factors=[2, 2], blocks=[2, 2])
x = torch.randn(1, 8, 16, 16)
y = unet(x)
print(y.shape)
```
