from typing import List

from a_unet.blocks import DownsampleT, Repeat, ResnetBlockT, Skip, UpsampleT
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

    # Define convolutional blocks types with provided dimensions
    Downsample = DownsampleT(dim=dim)
    Upsample = UpsampleT(dim=dim)
    ResnetBlock = ResnetBlockT(dim=dim)

    # Resnet stack
    def Block(channels: int, n_blocks: int) -> nn.Module:
        resnet_block = ResnetBlock(in_channels=channels, out_channels=channels)
        resnet_stack = Repeat(resnet_block, times=n_blocks)
        return resnet_stack

    # Build UNet recursively
    def build(i: int) -> nn.Module:
        if i == n_layers:
            return nn.Identity()
        n_channels = channels[i - 1] if i > 0 else in_channels
        factor = factors[i]

        return Skip(
            Downsample(factor=factor, in_channels=n_channels, out_channels=channels[i]),
            Block(channels=channels[i], n_blocks=blocks[i]),
            build(i + 1),
            Block(channels=channels[i], n_blocks=blocks[i]),
            Upsample(factor=factor, in_channels=channels[i], out_channels=n_channels),
        )

    return build(0)
