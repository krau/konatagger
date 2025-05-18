import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionModel(nn.Module):
    image_size: int
    n_tags: int

    def __init__(self, image_size: int, n_tags: int):
        super().__init__()

        self.image_size = image_size
        self.n_tags = n_tags

    @staticmethod
    def load_model(path: Path | str, device: str | None = None) -> "VisionModel":
        """
        Load a model from a directory.
        :param path: The directory containing the model.
        :return: The model, the image size, and the number of tags.
        """
        with open(Path(path) / "config.json", "r") as f:
            config = json.load(f)

        if (Path(path) / "model.safetensors").exists():
            from safetensors.torch import load_file

            resume = load_file(Path(path) / "model.safetensors", device="cpu")
        else:
            resume = torch.load(
                Path(path) / "model.pt", map_location=torch.device("cpu")
            )["model"]

        model_classes = VisionModel.__subclasses__()
        model_cls = next(
            cls for cls in model_classes if cls.__name__ == config["class"]
        )

        model = model_cls(**{k: v for k, v in config.items() if k != "class"})
        model.load(resume)
        if device is not None:
            model = model.to(device)

        return model

    @staticmethod
    def from_config(config: dict) -> "VisionModel":
        model_classes = VisionModel.__subclasses__()
        model_cls = next(
            cls for cls in model_classes if cls.__name__ == config["class"]
        )
        return model_cls(**{k: v for k, v in config.items() if k != "class"})

    def get_optimized_parameters(self, lr: float):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self, state_dict):
        raise NotImplementedError


def basic_calculate_loss(
    preds: dict[str, torch.Tensor],
    batch: dict,
    pos_weight: torch.Tensor | None,
    loss_type: str,
):
    if loss_type == "ce":
        loss = F.binary_cross_entropy_with_logits(preds["tags"], batch["tags"])
    elif loss_type == "weighted":
        loss = F.binary_cross_entropy_with_logits(
            preds["tags"], batch["tags"], pos_weight=pos_weight
        )
    elif loss_type == "focal":
        gamma = 2
        p = torch.sigmoid(preds["tags"])
        ce_loss = F.binary_cross_entropy_with_logits(
            preds["tags"], batch["tags"], reduction="none"
        )
        p_t = p * batch["tags"] + (1 - p) * (1 - batch["tags"])
        loss = ce_loss * ((1 - p_t) ** gamma)
        loss = loss.mean()
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    return loss


def sinusoidal_position_embedding(
    width: int, height: int, depth: int, dtype, device, temperature=10000
):
    """
    Sinusoidal position embedding. Returns a flat tensor of shape (h * w, d).
    """
    assert depth % 4 == 0, "Embedding dimension must be divisible by 4."

    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    omega = torch.arange(depth // 4, device=device) / (depth // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    embedding = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)

    return embedding.type(dtype)


class StochDepth(nn.Module):
    def __init__(self, drop_rate: float, scale_by_keep: bool = False):
        super().__init__()
        self.drop_rate = drop_rate
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if not self.training:
            return x

        batch_size = x.shape[0]
        r = torch.rand((batch_size, 1, 1), device=x.device)
        keep_prob = 1 - self.drop_rate
        binary_tensor = torch.floor(keep_prob + r)
        if self.scale_by_keep:
            x = x / keep_prob

        return x * binary_tensor


class SkipInitChannelwise(nn.Module):
    def __init__(self, channels, init_val=1e-6):
        super().__init__()
        self.channels = channels
        self.init_val = init_val
        self.skip = nn.Parameter(torch.ones(channels) * init_val)

    def forward(self, x):
        return x * self.skip


class PosEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int, use_sine: bool, patch_size: int):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.use_sine = use_sine
        self.patch_size = patch_size

        if not self.use_sine:
            self.embedding = nn.Embedding(max_len, d_model)
            nn.init.trunc_normal_(self.embedding.weight, std=0.02)
            self.register_buffer("position_ids", torch.arange(max_len))

    def forward(self, x, width: int, height: int):
        if self.use_sine:
            position_embeddings = sinusoidal_position_embedding(
                width // self.patch_size,
                height // self.patch_size,
                self.d_model,
                x.dtype,
                x.device,
            )
        else:
            position_embeddings = self.embedding(self.position_ids)

        return x + position_embeddings


class MLPBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, stochdepth_rate: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        if stochdepth_rate > 0:
            self.stochdepth = StochDepth(stochdepth_rate, scale_by_keep=True)
        else:
            self.stochdepth = None

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        if self.stochdepth is not None:
            x = self.stochdepth(x)
        x = self.linear2(x)
        return x


class ViTBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_ff: int,
        layerscale_init: float,
        stochdepth_rate: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # MHA
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.skip_init1 = SkipInitChannelwise(
            channels=d_model, init_val=layerscale_init
        )
        self.stochdepth1 = (
            StochDepth(stochdepth_rate, scale_by_keep=True)
            if stochdepth_rate > 0
            else None
        )

        # MLP
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLPBlock(d_model, d_ff, stochdepth_rate)
        self.skip_init2 = SkipInitChannelwise(
            channels=d_model, init_val=layerscale_init
        )
        self.stochdepth2 = (
            StochDepth(stochdepth_rate, scale_by_keep=True)
            if stochdepth_rate > 0
            else None
        )

    def forward(self, x):
        bsz, src_len, embed_dim = x.shape

        out = x
        out = self.norm1(out)

        # MHA
        qkv_states = self.qkv_proj(out).split(self.d_model, dim=-1)
        q_states = (
            qkv_states[0]
            .view(bsz, src_len, self.num_heads, embed_dim // self.num_heads)
            .transpose(1, 2)
        )  # (bsz, num_heads, src_len, embed_dim // num_heads)
        k_states = (
            qkv_states[1]
            .view(bsz, src_len, self.num_heads, embed_dim // self.num_heads)
            .transpose(1, 2)
        )  # (bsz, num_heads, src_len, embed_dim // num_heads)
        v_states = (
            qkv_states[2]
            .view(bsz, src_len, self.num_heads, embed_dim // self.num_heads)
            .transpose(1, 2)
        )  # (bsz, num_heads, src_len, embed_dim // num_heads)

        # with torch.backends.cuda.sdp_kernel(enable_math=False):
        with torch.nn.attention.sdpa_kernel(
            [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            ]
        ):
            out = F.scaled_dot_product_attention(
                q_states, k_states, v_states
            )  # (bsz, num_heads, tgt_len, head_dim)
            out = (
                out.transpose(1, 2).contiguous().view(bsz, src_len, embed_dim)
            )  # (bsz, tgt_len, embed_dim)

        out = self.out_proj(out)

        out = self.skip_init1(out)
        if self.stochdepth1 is not None:
            out = self.stochdepth1(out)
        x = out + x

        out = self.norm2(x)
        out = self.mlp(out)
        out = self.skip_init2(out)
        if self.stochdepth2 is not None:
            out = self.stochdepth2(out)

        out = out + x

        return out


def CaiT_LayerScale_init(network_depth):
    if network_depth <= 18:
        return 1e-1
    elif network_depth <= 24:
        return 1e-5
    else:
        return 1e-6


class CNNLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 3)
        x = self.norm(x)
        x = x.transpose(1, 3)
        return x


class CNNStem(nn.Module):
    def __init__(self, config: str):
        super().__init__()
        self.config = config

        layers = []
        channels = 3

        for line in config.split(";"):
            ty, line = line.split(":") if ":" in line else (line, "")
            options = line.split(",")
            options = [o.split("=") for o in options] if line else []
            options = {k: v for k, v in options}

            if ty == "conv":
                layers.append(
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=int(options["c"]),
                        kernel_size=int(options["k"] if "k" in options else 3),
                        stride=int(options["s"] if "s" in options else 2),
                        bias=True,
                        padding=int(options["p"] if "p" in options else 1),
                    )
                )
                channels = int(options["c"])
            elif ty == "bn":
                layers.append(nn.BatchNorm2d(channels))
            elif ty == "ln":
                layers.append(CNNLayerNorm(channels))
            elif ty == "relu":
                layers.append(nn.ReLU())
            elif ty == "gelu":
                layers.append(nn.GELU())

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ViT(VisionModel):
    def __init__(
        self,
        n_tags: int,
        image_size: int,
        num_blocks: int,
        patch_size: int,
        d_model: int,
        mlp_dim: int,
        num_heads: int,
        stochdepth_rate: float,
        use_sine: bool,
        loss_type: str,
        layerscale_init: Optional[float] = None,
        head_mean_after: bool = False,
        cnn_stem: str | None = None,
        patch_dropout: float = 0.0,
    ):
        super().__init__(image_size, n_tags)

        # assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        out_dim = n_tags
        self.n_tags = n_tags
        self.loss_type = loss_type
        self.patch_size = patch_size
        self.head_mean_after = head_mean_after
        self.patch_dropout = patch_dropout

        layerscale_init = (
            CaiT_LayerScale_init(num_blocks)
            if layerscale_init is None
            else layerscale_init
        )
        self.patch_embeddings = (
            nn.Conv2d(
                in_channels=3,
                out_channels=d_model,
                kernel_size=patch_size,
                stride=patch_size,
                bias=True,
            )
            if cnn_stem is None
            else CNNStem(cnn_stem)
        )
        self.pos_embedding = PosEmbedding(
            d_model,
            (image_size // patch_size) ** 2,
            use_sine=use_sine,
            patch_size=patch_size,
        )

        self.blocks = nn.ModuleList(
            [
                ViTBlock(num_heads, d_model, mlp_dim, layerscale_init, stochdepth_rate)
                for _ in range(num_blocks)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)

    def forward(
        self, batch, return_embeddings=False, return_loss: bool = False, pos_weight=None
    ):
        B, C, H, W = batch["image"].shape
        assert H % self.patch_size == 0, (
            f"Input image height ({H}) needs to be divisible by the patch size ({self.patch_size})."
        )
        assert W % self.patch_size == 0, (
            f"Input image width ({W}) needs to be divisible by the patch size ({self.patch_size})."
        )

        x = self.patch_embeddings(
            batch["image"]
        )  # (bsz, d_model, patch_num, patch_num)
        x = x.flatten(2).transpose(1, 2)  # (bsz, patch_num ** 2, d_model)
        x = self.pos_embedding(x, W, H)  # (bsz, patch_num ** 2, d_model)

        # Patch dropout
        seq_len = x.shape[1]
        patch_dropout = int(math.ceil((1.0 - self.patch_dropout) * seq_len))

        if patch_dropout != seq_len:
            # Generate a matrix of random numbers between 0 and 1 of shape (B, seq_len)
            patch_mask = torch.rand(B, seq_len, device=x.device)
            # For each batch tensor, use argsort to convert the random numbers into a permutation of the patch indices
            patch_mask = torch.argsort(patch_mask, dim=1)
            # Truncate
            patch_mask = patch_mask[:, :patch_dropout]

            x = x.gather(1, patch_mask.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

            # indices = torch.randperm(seq_len, device=x.device)[:patch_dropout]
            # x = x[:, indices, :]

        # Transformer
        for block in self.blocks:
            x = block(x)

        # Head
        result = {}

        x = self.norm(x)
        if self.head_mean_after:
            x = self.head(x)
            x = x.mean(dim=1)
        else:
            x = x.mean(dim=1)
            if return_embeddings:
                result["embeddings"] = x
            x = self.head(x)

        result["tags"] = x

        if return_loss:
            result["loss"] = self.calculate_loss(result, batch, pos_weight)

        return result

    def calculate_loss(self, preds, batch, pos_weight):
        return basic_calculate_loss(preds, batch, pos_weight, self.loss_type)

    def get_optimized_parameters(self, lr: float):
        return self.parameters()

    def save(self):
        return self.state_dict()

    def load(self, state_dict):
        if (
            "head.weight" in state_dict
            and "head.bias" in state_dict
            and state_dict["head.weight"].shape[0] == (self.n_tags + 9)
        ):
            # Support old models which included 3 rating and 6 score dimensions
            state_dict["head.weight"] = state_dict["head.weight"][: self.n_tags]
            state_dict["head.bias"] = state_dict["head.bias"][: self.n_tags]

        self.load_state_dict(state_dict)
