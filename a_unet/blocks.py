from typing import Any, Callable, Optional, Sequence, Type, TypeVar

import torch
from einops import pack, rearrange, repeat, unpack
from torch import Tensor, einsum, nn
from typing_extensions import TypeGuard

V = TypeVar("V")


def T(t: Callable[..., V]) -> Callable[..., Callable[..., V]]:
    """Where the magic happens, builds a type template for a given type"""
    return lambda *a, **ka: lambda *b, **kb: t(*(*a, *b), **{**ka, **kb})


def exists(val: Optional[V]) -> TypeGuard[V]:
    return val is not None


def default(val: Optional[V], d: V) -> V:
    if exists(val):
        return val
    return d


def Module(modules: Sequence[nn.Module], forward_fn: Callable):
    """Functional module helper"""

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(modules)

        def forward(self, *args, **kwargs):
            return forward_fn(*args, **kwargs)

    return Module()


class Sequential(nn.Module):
    """Custom Sequential that forwards args"""

    def __init__(self, *blocks):
        super().__init__()
        self.blocks = nn.ModuleList(
            [b if isinstance(b, nn.Module) else b() for b in blocks]
        )

    def forward(self, x: Tensor, *args) -> Tensor:
        for block in self.blocks:
            x = block(x, *args)
        return x


def Select(args_fn: Callable) -> Callable[..., Type[nn.Module]]:
    """Selects (swap, remove) forward arguments given a (lambda) function"""

    def fn(block_t: Type[nn.Module]) -> Type[nn.Module]:
        class Select(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.block = block_t(*args, **kwargs)
                self.args_fn = args_fn

            def forward(self, *args, **kwargs):
                return self.block(args_fn(*args), **kwargs)

        return Select

    return fn


class Packed(Sequential):
    """Packs, and transposes non-channel dims, useful for attention-like view"""

    def forward(self, x: Tensor, *args) -> Tensor:
        x, ps = pack([x], "b d *")
        x = rearrange(x, "b d n -> b n d")
        x = super().forward(x, *args)
        x = rearrange(x, "b n d -> b d n")
        x = unpack(x, ps, "b d *")[0]
        return x


def Repeat(block: Any, times: int) -> Any:
    return Sequential(*(block,) * times)


def Skip(merge_fn: Callable[[Tensor, Tensor], Tensor] = torch.add) -> Type[Sequential]:
    class Skip(Sequential):

        """Adds skip connection around modules"""

        def forward(self, x: Tensor, *args) -> Tensor:
            return merge_fn(x, super().forward(x, *args))

    return Skip


def Conv(dim: int, *args, **kwargs) -> nn.Module:
    return [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim - 1](*args, **kwargs)


def Downsample(factor: int = 2, conv_t=Conv, **kwargs) -> nn.Module:
    return conv_t(kernel_size=factor, stride=factor, **kwargs)


def Upsample(
    factor: int = 2, mode: str = "nearest", conv_t=Conv, **kwargs
) -> nn.Module:
    return nn.Sequential(
        nn.Upsample(scale_factor=factor, mode="nearest"),
        conv_t(kernel_size=3, padding=1, **kwargs),
    )


def ConvBlock(
    in_channels: int,
    activation_t=nn.SiLU,
    norm_t=T(nn.GroupNorm)(num_groups=1),
    conv_t=Conv,
    **kwargs,
) -> nn.Module:
    return nn.Sequential(
        norm_t(num_channels=in_channels),
        activation_t(),
        conv_t(in_channels=in_channels, **kwargs),
    )


def ResnetBlock(
    in_channels: int,
    out_channels: int,
    dim: Optional[int] = None,
    conv_block_t=ConvBlock,
    conv_t=Conv,
) -> nn.Module:
    ConvBlock = T(conv_block_t)(dim=dim, kernel_size=3, padding=1)
    Conv = T(conv_t)(dim=dim, kernel_size=1)

    conv_block = Sequential(
        ConvBlock(in_channels=in_channels, out_channels=out_channels),
        ConvBlock(in_channels=out_channels, out_channels=out_channels),
    )
    conv = nn.Identity()
    if in_channels != out_channels:
        conv = Conv(in_channels=in_channels, out_channels=out_channels)

    return Module([conv_block, conv], lambda x: conv_block(x) + conv(x))


def add_mask(sim: Tensor, mask: Tensor) -> Tensor:
    b, ndim = sim.shape[0], mask.ndim
    if ndim == 3:
        mask = rearrange(mask, "b n m -> b 1 n m")
    if ndim == 2:
        mask = repeat(mask, "n m -> b 1 n m", b=b)
    max_neg_value = -torch.finfo(sim.dtype).max
    sim = sim.masked_fill(~mask, max_neg_value)
    return sim


def AttentionBase(features: int, head_features: int, num_heads: int):
    scale = head_features**-0.5
    mid_features = head_features * num_heads
    to_out = nn.Linear(in_features=mid_features, out_features=features, bias=False)

    def forward(
        q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        h = num_heads
        # Split heads
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        # Compute similarity matrix and add eventual mask
        sim = einsum("... n d, ... m d -> ... n m", q, k) * scale
        sim = add_mask(sim, mask) if exists(mask) else sim
        # Get attention matrix with softmax
        attn = sim.softmax(dim=-1, dtype=torch.float32)
        # Compute values
        out = einsum("... n m, ... m d -> ... n d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return to_out(out)

    return Module([to_out], forward)


def FixedEmbedding(max_length: int, features: int):
    embedding = nn.Embedding(max_length, features)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, length, device = *x.shape[0:2], x.device
        assert_message = "Input sequence length must be <= max_length"
        assert length <= max_length, assert_message
        position = torch.arange(length, device=device)
        fixed_embedding = embedding(position)
        fixed_embedding = repeat(fixed_embedding, "n d -> b n d", b=batch_size)
        return fixed_embedding

    return Module([embedding], forward)


class Attention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        context_features: Optional[int] = None,
        max_length: Optional[int] = None,
        attention_base_t=AttentionBase,
        positional_embedding_t=None,
    ):
        super().__init__()
        self.context_features = context_features
        self.use_positional_embedding = exists(positional_embedding_t)
        self.use_context = exists(context_features)
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.max_length = max_length
        if self.use_positional_embedding:
            assert exists(max_length)
            self.positional_embedding = positional_embedding_t(
                max_length=max_length, features=features
            )

        self.norm = nn.LayerNorm(features)
        self.norm_context = nn.LayerNorm(context_features)
        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        self.to_kv = nn.Linear(
            in_features=context_features, out_features=mid_features * 2, bias=False
        )
        self.attention = attention_base_t(
            features, num_heads=num_heads, head_features=head_features
        )

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        assert_message = "You must provide a context when using context_features"
        assert not self.context_features or exists(context), assert_message
        skip = x
        if self.use_positional_embedding:
            x = x + self.positional_embedding(x)
        # Use context if provided
        context = context if exists(context) and self.use_context else x
        # Normalize then compute q from input and k,v from context
        x, context = self.norm(x), self.norm_context(context)
        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))
        # Compute and return attention
        return skip + self.attention(q, k, v)


def FeedForward(features: int, multiplier: int) -> nn.Module:
    mid_features = features * multiplier
    return Skip(torch.add)(
        nn.Linear(in_features=features, out_features=mid_features),
        nn.GELU(),
        nn.Linear(in_features=mid_features, out_features=features),
    )


ConvT = T(Conv)
ConvBlockT = T(ConvBlock)
DownsampleT = T(Downsample)
UpsampleT = T(Upsample)
ResnetBlockT = T(ResnetBlock)
AttentionT = T(Attention)
FeedForwardT = T(FeedForward)
