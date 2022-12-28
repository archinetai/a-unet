from typing import Callable, List, Optional, Sequence, Type, Union

from torch import Tensor, nn

from ..blocks import (
    Attention,
    Conv,
    CrossAttention,
    Downsample,
    FeedForward,
    LinearAttentionBase,
    MergeAdd,
    MergeCat,
    MergeModulate,
    Modulation,
    Packed,
    ResnetBlock,
    Select,
    Sequential,
    T,
    Upsample,
    default,
    exists,
)

""" Block """

SkipAdd = MergeAdd
SkipCat = MergeCat
SkipModulate = MergeModulate


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        channels: int,
        factor: int,
        depth: Optional[int] = None,
        downsample_t: Callable = Downsample,
        upsample_t: Callable = Upsample,
        skip_t: Callable = SkipAdd,
        items: Sequence[Union[Type, str]] = [],
        items_up: Optional[Sequence[Type]] = None,
        modulation_features: Optional[int] = None,
        out_channels: Optional[int] = None,
        inner_block: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__()
        out_channels = default(out_channels, in_channels)
        items_down = items
        items_up = default(items_up, items_down)  # type: ignore

        self.downsample = downsample_t(
            dim=dim, factor=factor, in_channels=in_channels, out_channels=channels
        )

        self.upsample = upsample_t(
            dim=dim, factor=factor, in_channels=channels, out_channels=out_channels
        )

        self.skip_adapter = (
            Conv(
                dim=dim,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )
            if in_channels != out_channels
            else nn.Identity()
        )

        self.merge = skip_t(
            dim=dim, channels=out_channels, modulation_features=modulation_features
        )

        # Kwargs forwarded to all items
        item_kwargs = dict(
            dim=dim,
            channels=channels,
            depth=depth,
            modulation_features=modulation_features,
            **kwargs
        )

        # Stack and build: items down -> inner block -> items up
        items_all: List[nn.Module] = []
        for i, items in enumerate([items_down, items_up]):  # type: ignore
            for Item in items:
                items_all += [Item(**item_kwargs)]  # type: ignore
            if i == 0 and exists(inner_block):
                items_all += [inner_block]

        self.block = Sequential(*items_all)

    def forward(
        self,
        x: Tensor,
        features: Optional[Tensor] = None,
        embedding: Optional[Tensor] = None,
        channels: Optional[Sequence[Tensor]] = None,
    ) -> Tensor:
        skip = self.skip_adapter(x)
        x = self.downsample(x)
        x = self.block(x, features, embedding, channels)
        x = self.upsample(x)
        x = self.merge(skip, x, features)
        return x


# Block type, to be provided in UNet
BlockT = T(Block, override=False)


""" UNet """


class UNet(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        blocks: Sequence,
        out_channels: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        num_layers = len(blocks)
        out_channels = default(out_channels, in_channels)

        def Net(i: int) -> Optional[nn.Module]:
            if i == num_layers:
                return None
            block_t = blocks[i]
            in_ch = in_channels if i == 0 else blocks[i - 1].kwargs["channels"]
            out_ch = out_channels if i == 0 else in_ch
            return block_t(
                dim=dim,
                in_channels=in_ch,
                out_channels=out_ch,
                inner_block=Net(i + 1),
                **kwargs
            )

        self.net = Net(0)

    def forward(
        self,
        x: Tensor,
        features: Optional[Tensor] = None,
        embedding: Optional[Tensor] = None,
        channels: Optional[Sequence[Tensor]] = None,
    ) -> Tensor:
        return self.net(x, features, embedding, channels)  # type: ignore


""" Items """


# Selections for item forward paramters
SelectX = Select(lambda x, *_: (x,))
SelectXE = Select(lambda x, f, e, *_: (x, e))
SelectXF = Select(lambda x, f, *_: (x, f))


def ResnetItem(
    dim: Optional[int] = None, channels: Optional[int] = None, **kwargs
) -> nn.Module:
    msg = "ResnetItem requires dim and channels"
    assert exists(dim) and exists(channels), msg
    Item = SelectX(ResnetBlock)
    return Item(dim=dim, in_channels=channels, out_channels=channels)  # type: ignore


def AttentionItem(
    channels: Optional[int] = None,
    attention_features: Optional[int] = None,
    attention_heads: Optional[int] = None,
    **kwargs
) -> nn.Module:
    msg = "AttentionItem requires channels, attention_features, attention_heads"
    assert exists(attention_features) and exists(attention_heads), msg
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
    **kwargs
) -> nn.Module:
    msg = "CrossAttentionItem requires channels, embedding_features, attention_*"
    assert (
        exists(embedding_features)
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
    msg = "M block requires channels, modulation_features"
    assert exists(channels) and exists(modulation_features), msg
    Item = SelectXF(Modulation)
    return Packed(
        Item(in_features=channels, num_features=modulation_features)  # type: ignore
    )


def LinearAttentionItem(
    channels: Optional[int] = None,
    attention_features: Optional[int] = None,
    attention_heads: Optional[int] = None,
    **kwargs
) -> nn.Module:
    msg = "LA block requires attention_features and attention_heads"
    assert exists(attention_features) and exists(attention_heads), msg
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
    **kwargs
) -> nn.Module:
    msg = "LinearCrossAttentionItem requires channels, embedding_features, attention_*"
    assert (
        exists(embedding_features)
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
):
    msg = "FeedForwardItem block requires channels, attention_multiplier"
    assert exists(channels) and exists(attention_multiplier), msg
    Item = SelectX(FeedForward)
    return Packed(
        Item(features=channels, multiplier=attention_multiplier)  # type: ignore
    )
