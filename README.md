# A-UNet

A library that provides building blocks to customize UNets, in PyTorch.

## Install
```bash
pip install a-unet
```

[![PyPI - Python Version](https://img.shields.io/pypi/v/a-unet?style=flat&colorA=black&colorB=black)](https://pypi.org/project/a-unet/)


## Usage

### Basic UNet

<details> <summary> A convolutional only UNet generic to any dimension, using A-UNet blocks. </summary>

```py
from typing import List
from a_unet import T, Downsample, Repeat, ResnetBlock, Skip, Upsample
from torch import nn

def UNet(
    dim: int,
    in_channels: int,
    channels: List[int],
    factors: List[int],
    blocks: List[int],
) -> nn.Module:
    # Check lengths
    n_layers = len(channels)
    assert n_layers == len(factors) and n_layers == len(blocks), "lengths must match"

    # Resnet stack
    def Stack(channels: int, n_blocks: int) -> nn.Module:
        # The T function is used create a type template that pre-initializes paramters if called
        Block = T(ResnetBlock)(dim=dim, in_channels=channels, out_channels=channels)
        resnet = Repeat(Block, times=n_blocks)
        return resnet

    # Build UNet recursively
    def Net(i: int) -> nn.Module:
        if i == n_layers: return nn.Identity()
        in_ch, out_ch = (channels[i - 1] if i > 0 else in_channels), channels[i]
        factor = factors[i]
        # Wraps modules with skip connection that merges paths with torch.add
        return Skip(torch.add)(
            Downsample(dim=dim, factor=factor, in_channels=in_ch, out_channels=out_ch),
            Stack(channels=out_ch, n_blocks=blocks[i]),
            Net(i + 1),
            Stack(channels=out_ch, n_blocks=blocks[i]),
            Upsample(dim=dim, factor=factor, in_channels=out_ch, out_channels=in_ch),
        )
    return Net(0)
```

</details>

```py
unet = UNet(dim=2, in_channels=8, channels=[256, 512], factors=[2, 2], blocks=[2, 2])
x = torch.randn(1, 8, 16, 16)
y = unet(x) # [1, 8, 16, 16]
```
