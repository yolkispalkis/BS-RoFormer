from __future__ import annotations
from functools import partial

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torchvision.ops import StochasticDepth # ##### НОВОЕ: Импорт для DropPath #####

from bs_roformer.attend import Attend

from beartype.typing import Callable
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, pack, unpack, reduce, repeat
from einops.layers.torch import Rearrange

from librosa import filters

from hyper_connections import get_init_and_expand_reduce_stream_functions

# helper functions

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

# norm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

##### НОВОЕ: SwiGLU активация #####
class SwiGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.silu(gate)

# attention

##### ИЗМЕНЕНО: FeedForward теперь использует SwiGLU #####
class FeedForward(Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = int(dim * mult * 2 / 3)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner * 2, bias = False),
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        rotary_embed = None,
        flash = True,
        add_value_residual = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head **-0.5
        dim_inner = heads * dim_head
        self.rotary_embed = rotary_embed
        self.attend = Attend(flash = flash, dropout = dropout)
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_gates = nn.Linear(dim, heads)
        self.learned_value_residual_mix = nn.Sequential(
            nn.Linear(dim, heads), Rearrange('b n h -> b h n 1'), nn.Sigmoid()
        ) if add_value_residual else None
        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias = False), nn.Dropout(dropout)
        )

    def forward(self, x, value_residual = None):
        x = self.norm(x)
        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = self.heads)
        orig_v = v
        if exists(self.learned_value_residual_mix):
            mix = self.learned_value_residual_mix(x)
            assert exists(value_residual)
            v = v.lerp(mix, value_residual)
        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)
        out = self.attend(q, k, v)
        gates = self.to_gates(x)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), orig_v

class LinearAttention(Module):
    @beartype
    def __init__(self, *, dim, dim_head = 32, heads = 8, scale = 8, flash = False, dropout = 0., add_value_residual = False):
        super().__init__()
        dim_inner = dim_head * heads
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h d n', qkv = 3, h = heads)
        )
        self.temperature = nn.Parameter(torch.zeros(heads, 1, 1))
        self.attend = Attend(scale = scale, dropout = dropout, flash = flash)
        self.learned_value_residual_mix = nn.Sequential(
            nn.Linear(dim, heads), Rearrange('b n h -> b h 1 n'), nn.Sigmoid()
        ) if add_value_residual else None
        self.to_out = nn.Sequential(
            Rearrange('b h d n -> b n (h d)'), nn.Linear(dim_inner, dim, bias = False)
        )

    def forward(self, x, value_residual = None):
        x = self.norm(x)
        q, k, v = self.to_qkv(x)
        orig_v = v
        if exists(self.learned_value_residual_mix):
            mix = self.learned_value_residual_mix(x)
            assert exists(value_residual)
            v = v.lerp(mix, value_residual)
        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()
        out = self.attend(q, k, v)
        return self.to_out(out), orig_v

##### ИЗМЕНЕНО: Transformer теперь использует DropPath (StochasticDepth) и имеет явные residual-соединения #####
class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        norm_output = True,
        rotary_embed = None,
        flash_attn = True,
        linear_attn = False,
        add_value_residual = False,
        num_residual_streams = 1,
        drop_path_rate = 0.
    ):
        super().__init__()
        init_hyper_conn, *_ = get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)
        self.layers = ModuleList([])
        drop_path_rates = torch.linspace(0., drop_path_rate, depth).tolist()

        for ind in range(depth):
            if linear_attn:
                attn = LinearAttention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = flash_attn, add_value_residual = add_value_residual)
            else:
                attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_embed = rotary_embed, flash = flash_attn, add_value_residual = add_value_residual)
            ff = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            self.layers.append(ModuleList([
                init_hyper_conn(dim = dim, branch = attn),
                init_hyper_conn(dim = dim, branch = ff),
                StochasticDepth(drop_path_rates[ind], mode = 'row'),
                StochasticDepth(drop_path_rates[ind], mode = 'row')
            ]))
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x, value_residual = None):
        first_values = None
        for attn, ff, attn_drop_path, ff_drop_path in self.layers:
            attn_out, values = attn(x, value_residual = value_residual)
            x = x + attn_drop_path(attn_out)
            first_values = default(first_values, values)

            ff_out = ff(x)
            x = x + ff_drop_path(ff_out)
        return self.norm(x), first_values

# bandsplit module

class BandSplit(Module):
    @beartype
    def __init__(self, dim, dim_inputs: tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([
            nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim))
            for dim_in in dim_inputs
        ])
    def forward(self, x):
        x = x.split(self.dim_inputs, dim = -1)
        return torch.stack([net(band) for band, net in zip(x, self.to_features)], dim = -2)

##### НОВОЕ: Сверточный "стебель" #####
class ConvStem(Module):
    @beartype
    def __init__(self, dim_in, dim_out, kernel_size = 3, num_layers = 2):
        super().__init__()
        self.layers = ModuleList([])
        for i in range(num_layers):
            is_first = i == 0
            current_dim_in = dim_in if is_first else dim_out
            self.layers.append(nn.Sequential(
                nn.Conv1d(current_dim_in, dim_out, kernel_size, padding = 'same'),
                nn.BatchNorm1d(dim_out),
                nn.GELU()
            ))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def MLP(dim_in, dim_out, dim_hidden=None, depth=1, activation=nn.Tanh):
    dim_hidden = default(dim_hidden, dim_in)
    net = []
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)
    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)
        net.append(nn.Linear(layer_dim_in, layer_dim_out))
        if not is_last:
            net.append(activation())
    return nn.Sequential(*net)

class MaskEstimator(Module):
    @beartype
    def __init__(self, dim, dim_inputs: tuple[int, ...], depth, mlp_expansion_factor=4):
        super().__init__()
        self.dim_inputs = dim_inputs
        dim_hidden = dim * mlp_expansion_factor
        self.to_freqs = ModuleList([
            nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden = dim_hidden, depth = depth),
                nn.GLU(dim = -1)
            ) for dim_in in dim_inputs
        ])
    def forward(self, x):
        x_bands = x.unbind(dim = -2)
        return torch.cat([mlp(band) for band, mlp in zip(x_bands, self.to_freqs)], dim = -1)

# main class

class MelBandRoformer(Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo = False,
        num_stems = 1,
        time_transformer_depth = 2,
        freq_transformer_depth = 2,
        linear_transformer_depth = 1,
        num_bands = 60,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.1,
        ff_dropout = 0.1,
        drop_path_rate = 0.,
        use_conv_stem = True,
        flash_attn = True,
        linear_flash_attn = None,
        dim_freqs_in = 1025,
        sample_rate = 44100,
        stft_n_fft = 2048,
        stft_hop_length = 512,
        stft_win_length = 2048,
        stft_normalized = False,
        stft_window_fn: Callable | None = None,
        mask_estimator_depth = 1,
        multi_stft_resolution_loss_weight = 1.,
        multi_stft_resolutions_window_sizes: tuple[int, ...] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size = 147,
        multi_stft_normalized = False,
        multi_stft_window_fn: Callable = torch.hann_window,
        match_input_audio_length = False,
        add_value_residual = True,
        num_residual_streams = 4
    ):
        super().__init__()
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.layers = ModuleList([])
        transformer_kwargs = dict(
            dim=dim, heads=heads, dim_head=dim_head, attn_dropout=attn_dropout,
            ff_dropout=ff_dropout, num_residual_streams=num_residual_streams,
            drop_path_rate=drop_path_rate
        )
        time_rotary_embed = RotaryEmbedding(dim = dim_head)
        freq_rotary_embed = RotaryEmbedding(dim = dim_head)
        linear_flash_attn = default(linear_flash_attn, flash_attn)
        _, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        for layer_index in range(depth):
            is_first = layer_index == 0
            self.layers.append(nn.ModuleList([
                Transformer(depth = linear_transformer_depth, linear_attn = True, flash_attn = linear_flash_attn, add_value_residual = add_value_residual and not is_first, **transformer_kwargs) if linear_transformer_depth > 0 else None,
                Transformer(depth = time_transformer_depth, rotary_embed = time_rotary_embed, flash_attn = flash_attn, add_value_residual = add_value_residual and not is_first, **transformer_kwargs),
                Transformer(depth = freq_transformer_depth, rotary_embed = freq_rotary_embed, flash_attn = flash_attn, add_value_residual = add_value_residual and not is_first, **transformer_kwargs)
            ]))

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)
        self.stft_kwargs = dict(n_fft = stft_n_fft, hop_length = stft_hop_length, win_length = stft_win_length, normalized = stft_normalized)
        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, return_complex=True).shape[1]
        
        mel_fb = filters.mel(sr = sample_rate, n_fft = stft_n_fft, n_mels = num_bands)
        mel_fb = torch.from_numpy(mel_fb)
        mel_fb[0][0] = 1.; mel_fb[-1, -1] = 1.
        freqs_per_band = mel_fb > 0
        assert freqs_per_band.any(dim=0).all()
        
        freq_indices = repeat(torch.arange(freqs), 'f -> b f', b = num_bands)[freqs_per_band]
        if stereo:
            freq_indices = repeat(freq_indices, 'f -> f s', s=2)
            freq_indices = freq_indices * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, 'f s -> (f s)')

        self.register_buffer('freq_indices', freq_indices, persistent=False)
        num_freqs_per_band = reduce(freqs_per_band, 'b f -> b', 'sum')
        self.register_buffer('num_bands_per_freq', reduce(freqs_per_band, 'b f -> f', 'sum'), persistent=False)

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in num_freqs_per_band.tolist())

        self.use_conv_stem = use_conv_stem
        if use_conv_stem:
            self.conv_stem = ConvStem(dim_in=sum(freqs_per_bands_with_complex), dim_out=sum(freqs_per_bands_with_complex))
        
        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)
        self.mask_estimators = nn.ModuleList([
            MaskEstimator(dim=dim, dim_inputs=freqs_per_bands_with_complex, depth=mask_estimator_depth)
            for _ in range(num_stems)
        ])
        
        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn
        self.multi_stft_kwargs = dict(hop_length = multi_stft_hop_size, normalized = multi_stft_normalized)
        self.match_input_audio_length = match_input_audio_length

    def forward(self, raw_audio, target = None, return_loss_breakdown = False):
        device, num_stems = raw_audio.device, self.num_stems
        if raw_audio.ndim == 2: raw_audio = rearrange(raw_audio, 'b t -> b 1 t')
        batch, channels, raw_audio_length = raw_audio.shape
        istft_length = raw_audio_length if self.match_input_audio_length else None
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2)

        raw_audio, packed_shape = pack_one(raw_audio, '* t')
        stft_window = self.stft_window_fn(device=device)
        stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        stft_repr = torch.view_as_real(stft_repr)
        stft_repr = unpack_one(stft_repr, packed_shape, '* f t c')
        stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')
        
        batch_arange = torch.arange(batch, device = device)[..., None]
        x = stft_repr[batch_arange, self.freq_indices]
        x = rearrange(x, 'b f t c -> b t (f c)')
        
        if self.use_conv_stem:
            x_conv = rearrange(x, 'b t d -> b d t')
            x_conv = self.conv_stem(x_conv)
            x = rearrange(x_conv, 'b d t -> b t d')

        x = self.band_split(x)

        linear_v_res, time_v_res, freq_v_res = None, None, None
        x = self.expand_streams(x)

        for linear_tf, time_tf, freq_tf in self.layers:
            if exists(linear_tf):
                x, ft_ps = pack([x], 'b * d')
                x, next_lin_v = linear_tf(x, value_residual = linear_v_res)
                linear_v_res = default(linear_v_res, next_lin_v)
                x, = unpack(x, ft_ps, 'b * d')
            
            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')
            x, next_time_v = time_tf(x, value_residual = time_v_res)
            time_v_res = default(time_v_res, next_time_v)
            x, = unpack(x, ps, '* t d')
            
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')
            x, next_freq_v = freq_tf(x, value_residual = freq_v_res)
            freq_v_res = default(freq_v_res, next_freq_v)
            x, = unpack(x, ps, '* f d')

        x = self.reduce_streams(x)
        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        masks = rearrange(masks, 'b n t (f c) -> b n f t c', c=2)
        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')
        
        stft_repr, masks = torch.view_as_complex(stft_repr), torch.view_as_complex(masks)
        masks = masks.type(stft_repr.dtype)
        
        scatter_indices = repeat(self.freq_indices, 'f -> b n f t', b=batch, n=num_stems, t=stft_repr.shape[-1])
        stft_repr_stems = repeat(stft_repr, 'b 1 ... -> b n ...', n=num_stems)
        masks_summed = torch.zeros_like(stft_repr_stems).scatter_add_(2, scatter_indices, masks)
        denom = repeat(self.num_bands_per_freq, 'f -> (f r) 1', r=channels)
        masks_avg = masks_summed / denom.clamp(min=1e-8)
        
        stft_repr = stft_repr * masks_avg
        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)
        recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False, length=istft_length)
        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', b=batch, s=self.audio_channels, n=num_stems)
        if num_stems == 1: recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')
        if not exists(target): return recon_audio

        if num_stems > 1: assert target.ndim == 4 and target.shape[1] == num_stems
        if target.ndim == 2: target = rearrange(target, '... t -> ... 1 t')
        target = target[..., :recon_audio.shape[-1]]
        loss = F.l1_loss(recon_audio, target)

        multi_stft_loss = 0.
        for win_size in self.multi_stft_resolutions_window_sizes:
            res_kwargs = dict(n_fft = max(win_size, self.multi_stft_n_fft), win_length = win_size, return_complex = True, window = self.multi_stft_window_fn(win_size, device=device), **self.multi_stft_kwargs)
            recon_Y = torch.stft(rearrange(recon_audio, '... s t -> (... s) t'), **res_kwargs)
            target_Y = torch.stft(rearrange(target, '... s t -> (... s) t'), **res_kwargs)
            multi_stft_loss += F.l1_loss(recon_Y, target_Y)

        total_loss = loss + multi_stft_loss * self.multi_stft_resolution_loss_weight
        return total_loss if not return_loss_breakdown else (total_loss, (loss, multi_stft_loss))
