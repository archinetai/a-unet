from typing import Callable, List, Optional, Sequence, no_type_check

import torch
from torch import Tensor, nn

from .blocks import (
    Attention,
    Conv,
    ConvBlock,
    CrossAttention,
    Downsample,
    FeedForward,
    LinearAttentionBase,
    MergeAdd,
    MergeCat,
    MergeModulate,
    Modulation,
    Module,
    Packed,
    ResnetBlock,
    Select,
    Sequential,
    T,
    Upsample,
    default,
    exists,
)

"""
Items
"""

# Selections for item forward parameters
SelectX = Select(lambda x, *_: (x,))
SelectXF = Select(lambda x, f, *_: (x, f))
SelectXE = Select(lambda x, f, e, *_: (x, e))
SelectXC = Select(lambda x, f, e, c, *_: (x, c))


""" Downsample / Upsample """


def DownsampleItem(
    dim: Optional[int] = None,
    factor: Optional[int] = None,
    in_channels: Optional[int] = None,
    channels: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    msg = "DownsampleItem requires dim, factor, in_channels, channels"
    assert (
        exists(dim) and exists(factor) and exists(in_channels) and exists(channels)
    ), msg
    Item = SelectX(Downsample)
    return Item(  # type: ignore
        dim=dim, factor=factor, in_channels=in_channels, out_channels=channels
    )


def UpsampleItem(
    dim: Optional[int] = None,
    factor: Optional[int] = None,
    channels: Optional[int] = None,
    out_channels: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    msg = "UpsampleItem requires dim, factor, channels, out_channels"
    assert (
        exists(dim) and exists(factor) and exists(channels) and exists(out_channels)
    ), msg
    Item = SelectX(Upsample)
    return Item(  # type: ignore
        dim=dim, factor=factor, in_channels=channels, out_channels=out_channels
    )


""" Main """


def ResnetItem(
    dim: Optional[int] = None,
    channels: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    msg = "ResnetItem requires dim, channels, and resnet_groups"
    assert exists(dim) and exists(channels) and exists(resnet_groups), msg
    Item = SelectX(ResnetBlock)
    conv_block_t = T(ConvBlock)(norm_t=T(nn.GroupNorm)(num_groups=resnet_groups))
    return Item(
        dim=dim, in_channels=channels, out_channels=channels, conv_block_t=conv_block_t
    )  # type: ignore


def AttentionItem(
    channels: Optional[int] = None,
    attention_features: Optional[int] = None,
    attention_heads: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    msg = "AttentionItem requires channels, attention_features, attention_heads"
    assert (
        exists(channels) and exists(attention_features) and exists(attention_heads)
    ), msg
    Item = SelectX(Attention)
    return Packed(
        Item(  # type: ignore
            features=channels,
            head_features=attention_features,
            num_heads=attention_heads,
        )
    )


def CrossAttentionItem(
    channels: Optional[int] = None,
    attention_features: Optional[int] = None,
    attention_heads: Optional[int] = None,
    embedding_features: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    msg = "CrossAttentionItem requires channels, embedding_features, attention_*"
    assert (
        exists(channels)
        and exists(embedding_features)
        and exists(attention_features)
        and exists(attention_heads)
    ), msg
    Item = SelectXE(CrossAttention)
    return Packed(
        Item(  # type: ignore
            features=channels,
            head_features=attention_features,
            num_heads=attention_heads,
            context_features=embedding_features,
        )
    )


def ModulationItem(
    channels: Optional[int] = None, modulation_features: Optional[int] = None, **kwargs
) -> nn.Module:
    msg = "ModulationItem requires channels, modulation_features"
    assert exists(channels) and exists(modulation_features), msg
    Item = SelectXF(Modulation)
    return Packed(
        Item(in_features=channels, num_features=modulation_features)  # type: ignore
    )


def LinearAttentionItem(
    channels: Optional[int] = None,
    attention_features: Optional[int] = None,
    attention_heads: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    msg = "LinearAttentionItem requires attention_features and attention_heads"
    assert (
        exists(channels) and exists(attention_features) and exists(attention_heads)
    ), msg
    Item = SelectX(T(Attention)(attention_base_t=LinearAttentionBase))
    return Packed(
        Item(  # type: ignore
            features=channels,
            head_features=attention_features,
            num_heads=attention_heads,
        )
    )


def LinearCrossAttentionItem(
    channels: Optional[int] = None,
    attention_features: Optional[int] = None,
    attention_heads: Optional[int] = None,
    embedding_features: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    msg = "LinearCrossAttentionItem requires channels, embedding_features, attention_*"
    assert (
        exists(channels)
        and exists(embedding_features)
        and exists(attention_features)
        and exists(attention_heads)
    ), msg
    Item = SelectXE(T(CrossAttention)(attention_base_t=LinearAttentionBase))
    return Packed(
        Item(  # type: ignore
            features=channels,
            head_features=attention_features,
            num_heads=attention_heads,
            context_features=embedding_features,
        )
    )


def FeedForwardItem(
    channels: Optional[int] = None, attention_multiplier: Optional[int] = None, **kwargs
) -> nn.Module:
    msg = "FeedForwardItem requires channels, attention_multiplier"
    assert exists(channels) and exists(attention_multiplier), msg
    Item = SelectX(FeedForward)
    return Packed(
        Item(features=channels, multiplier=attention_multiplier)  # type: ignore
    )


def InjectChannelsItem(
    dim: Optional[int] = None,
    channels: Optional[int] = None,
    depth: Optional[int] = None,
    context_channels: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    msg = "InjectChannelsItem requires dim, depth, channels, context_channels"
    assert (
        exists(dim) and exists(depth) and exists(channels) and exists(context_channels)
    ), msg
    msg = "InjectChannelsItem requires context_channels > 0"
    assert context_channels > 0, msg

    conv = Conv(
        dim=dim,
        in_channels=channels + context_channels,
        out_channels=channels,
        kernel_size=1,
    )

    @no_type_check
    def forward(x: Tensor, channels: Sequence[Optional[Tensor]]) -> Tensor:
        msg_ = f"context `channels` at depth {depth} in forward"
        assert depth < len(channels), f"Required {msg_}"
        context = channels[depth]
        shape = torch.Size([x.shape[0], context_channels, *x.shape[2:]])
        msg = f"Required {msg_} to be tensor of shape {shape}, found {context.shape}"
        assert torch.is_tensor(context) and context.shape == shape, msg
        return conv(torch.cat([x, context], dim=1)) + x

    return SelectXC(Module)([conv], forward)  # type: ignore


""" Skip Adapters """


def SkipAdapter(
    dim: Optional[int] = None,
    in_channels: Optional[int] = None,
    out_channels: Optional[int] = None,
    **kwargs,
):
    msg = "SkipAdapter requires dim, in_channels, out_channels"
    assert exists(dim) and exists(in_channels) and exists(out_channels), msg
    Item = SelectX(Conv)
    return (
        Item(  # type: ignore
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        if in_channels != out_channels
        else SelectX(nn.Identity)()
    )


""" Skip Connections """


def SkipAdd(**kwargs) -> nn.Module:
    return MergeAdd()


def SkipCat(
    dim: Optional[int] = None,
    out_channels: Optional[int] = None,
    skip_scale: float = 2**-0.5,
    **kwargs,
) -> nn.Module:
    msg = "SkipCat requires dim, out_channels"
    assert exists(dim) and exists(out_channels), msg
    return MergeCat(dim=dim, channels=out_channels, scale=skip_scale)


def SkipModulate(
    dim: Optional[int] = None,
    out_channels: Optional[int] = None,
    modulation_features: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    msg = "SkipModulate requires dim, out_channels, modulation_features"
    assert exists(dim) and exists(out_channels) and exists(modulation_features), msg
    return MergeModulate(
        dim=dim, channels=out_channels, modulation_features=modulation_features
    )


""" Block """


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        downsample_t: Callable = DownsampleItem,
        upsample_t: Callable = UpsampleItem,
        skip_t: Callable = SkipAdd,
        skip_adapter_t: Callable = SkipAdapter,
        items: Sequence[Callable] = [],
        items_up: Optional[Sequence[Callable]] = None,
        out_channels: Optional[int] = None,
        inner_block: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__()
        out_channels = default(out_channels, in_channels)

        items_up = default(items_up, items)  # type: ignore
        items_down = [downsample_t] + list(items)
        items_up = list(items_up) + [upsample_t]
        items_kwargs = dict(
            in_channels=in_channels, out_channels=out_channels, **kwargs
        )

        # Build items stack: items down -> inner block -> items up
        items_all: List[nn.Module] = []
        items_all += [item_t(**items_kwargs) for item_t in items_down]
        items_all += [inner_block] if exists(inner_block) else []
        items_all += [item_t(**items_kwargs) for item_t in items_up]

        self.skip_adapter = skip_adapter_t(**items_kwargs)
        self.block = Sequential(*items_all)
        self.skip = skip_t(**items_kwargs)

    def forward(
        self,
        x: Tensor,
        features: Optional[Tensor] = None,
        embedding: Optional[Tensor] = None,
        channels: Optional[Sequence[Tensor]] = None,
    ) -> Tensor:
        skip = self.skip_adapter(x)
        x = self.block(x, features, embedding, channels)
        x = self.skip(skip, x, features)
        return x


# Block type, to be provided in UNet
XBlock = T(Block, override=False)


""" UNet """


class XUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        blocks: Sequence,
        out_channels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        num_layers = len(blocks)
        out_channels = default(out_channels, in_channels)

        def Net(i: int) -> Optional[nn.Module]:
            if i == num_layers:
                return None  # noqa
            block_t = blocks[i]
            in_ch = in_channels if i == 0 else blocks[i - 1].channels
            out_ch = out_channels if i == 0 else in_ch

            return block_t(
                in_channels=in_ch,
                out_channels=out_ch,
                depth=i,
                inner_block=Net(i + 1),
                **kwargs,
            )

        self.net = Net(0)

    def forward(
        self,
        x: Tensor,
        *,
        features: Optional[Tensor] = None,
        embedding: Optional[Tensor] = None,
        channels: Optional[Sequence[Tensor]] = None,
    ) -> Tensor:
        return self.net(x, features, embedding, channels)  # type: ignore
