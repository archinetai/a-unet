# A-UNet

A toolbox that provides hackable building blocks for generic 1D/2D/3D UNets, in PyTorch.

## Install
```bash
pip install a-unet
```

[![PyPI - Python Version](https://img.shields.io/pypi/v/a-unet?style=flat&colorA=black&colorB=black)](https://pypi.org/project/a-unet/)


## Usage

### Basic UNet

<details> <summary> (Code): A convolutional only UNet generic to any dimension. </summary>

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
unet = UNet(
  dim=2,
  in_channels=8,
  channels=[256, 512],
  factors=[2, 2],
  blocks=[2, 2]
)
x = torch.randn(1, 8, 16, 16)
y = unet(x) # [1, 8, 16, 16]
```


### ApeX UNet

<details> <summary> (Code): ApeX is a UNet template complete with tools for easy customizability. The following example UNet includes multiple features: (1) custom item arrangement for resnets, modulation, attention, and cross attention, (2) custom skip connection with concatenation, (3) time conditioning (usually used for diffusion), (4) classifier free guidance. </summary>

```py
from typing import Sequence, Optional, Callable

from a_unet import TimeConditioningPlugin, ClassifierFreeGuidancePlugin
from a_unet.apex import (
    XUNet,
    XBlock,
    ResnetItem as R,
    AttentionItem as A,
    CrossAttentionItem as C,
    ModulationItem as M,
    SkipCat
)

def UNet(
    dim: int,
    in_channels: int,
    channels: Sequence[int],
    factors: Sequence[int],
    items: Sequence[int],
    attentions: Sequence[int],
    cross_attentions: Sequence[int],
    attention_features: int,
    attention_heads: int,
    embedding_features: Optional[int] = None,
    skip_t: Callable = SkipCat,
    resnet_groups: int = 8,
    modulation_features: int = 1024,
    embedding_max_length: int = 0,
    use_classifier_free_guidance: bool = False,
    out_channels: Optional[int] = None,
):
    # Check lengths
    num_layers = len(channels)
    sequences = (channels, factors, items, attentions, cross_attentions)
    assert all(len(sequence) == num_layers for sequence in sequences)

    # Define UNet type with time conditioning and CFG plugins
    UNet = TimeConditioningPlugin(XUNet)
    if use_classifier_free_guidance:
        UNet = ClassifierFreeGuidancePlugin(UNet, embedding_max_length)

    return UNet(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        blocks=[
            XBlock(
                channels=channels,
                factor=factor,
                items=([R, M] + [A] * n_att + [C] * n_cross) * n_items,
            ) for channels, factor, n_items, n_att, n_cross in zip(*sequences)
        ],
        skip_t=skip_t,
        attention_features=attention_features,
        attention_heads=attention_heads,
        embedding_features=embedding_features,
        modulation_features=modulation_features,
        resnet_groups=resnet_groups
    )
```

</details>

```py
unet = UNet(
    dim=2,
    in_channels=2,
    channels=[128, 256, 512, 1024],
    factors=[2, 2, 2, 2],
    items=[2, 2, 2, 2],
    attentions=[0, 0, 0, 1],
    cross_attentions=[1, 1, 1, 1],
    attention_features=64,
    attention_heads=8,
    embedding_features=768,
    use_classifier_free_guidance=False
)
x = torch.randn(2, 2, 64, 64)
time = [0.2, 0.5]
embedding = torch.randn(2, 512, 768)
y = unet(x, time=time, embedding=embedding) # [2, 2, 64, 64]
```
