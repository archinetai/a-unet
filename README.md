# A-UNet

A library that provides building blocks to customize UNets, in PyTorch.

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

### Attention UNet

<details> <summary> (Code): A UNet generic to any dimension augmented with attention and cross attention for conditioning. </summary>

```py
from typing import List
from torch import nn
from a_unet import T, Ts, Downsample, Upsample, ResnetBlock, Attention, FeedForward, Select, Sequential, Repeat, Packed, Skip

def UNet(
    dim: int,
    in_channels: int,
    context_features: int,
    channels: List[int],
    factors: List[int],
    blocks: List[int],
    attentions: List[int],
    attention_heads: int,
    attention_features: int,
    attention_multiplier: int,
):
    # Check that all lists have matching lengths
    n_layers = len(channels)
    assert all(len(xs) == n_layers for xs in (factors, blocks, attentions))

    # Selects only first module input, ignores context
    S = Select(lambda x, context: x)

    # Pre-initalize attention, cross-attention, and feed-forward types with parameters
    A = T(Attention)(head_features=attention_features, num_heads=attention_heads)
    C = T(A)(context_features=context_features) # Same as A but with context features
    F = T(FeedForward)(multiplier=attention_multiplier)

    def Stack(channels: int, n_blocks: int, n_attentions: int):
        # Build resnet stack type
        Block = T(ResnetBlock)(dim=dim, in_channels=channels, out_channels=channels)
        ResnetStack = S(Repeat(Block, times=n_blocks))
        # Build attention, cross att, and feed forward types (ignoring context in A & F)
        Attention = T(S(A))(features=channels)
        CrossAttention = T(C)(features=channels)
        FeedForward = T(S(F))(features=channels)
        # Build transformer type
        Transformer = Ts(Sequential)(Attention, CrossAttention, FeedForward)
        TransformerStack = Repeat(Transformer, times=n_attentions)
        # Instantiate sequential resnet stack and transformer stack
        return Sequential(ResnetStack(), Packed(TransformerStack()))

    # Downsample and upsample types that ignore context
    Down = T(S(Downsample))(dim=dim)
    Up = T(S(Upsample))(dim=dim)

    def Net(i: int):
        if i == n_layers: return S(nn.Identity)()
        n_channels = channels[i-1] if i > 0 else in_channels
        factor = factors[i]

        return Skip(torch.add)(
            Down(factor=factor, in_channels=n_channels, out_channels=channels[i]),
            Stack(channels=channels[i], n_blocks=blocks[i], n_attentions=attentions[i]),
            Net(i+1),
            Stack(channels=channels[i], n_blocks=blocks[i], n_attentions=attentions[i]),
            Up(factor=factor, in_channels=channels[i], out_channels=n_channels)
        )

    return Net(0)
```

</details>

```py
unet = UNet(
    dim=2,
    in_channels=8,
    context_features=512,
    channels=[256, 512],
    factors=[2, 2],
    blocks=[2, 2],
    attentions=[2, 2],
    attention_heads=8,
    attention_features=64,
    attention_multiplier=4,
)
x = torch.randn(1, 8, 16, 16)
context = torch.randn(1, 256, 512)
y = unet(x, context) # [1, 8, 16, 16]
```
