import timm
import torch
import torch.nn as nn


class DINOv3Backbone(nn.Module):
    """Thin timm wrapper around DINOv3 ViT models.

    Exposes the same ``get_intermediate_layers`` / ``embed_dim`` / ``patch_size``
    interface used by the DPT head so that it is a drop-in replacement for the
    custom DINOv2 backbone.

    Parameters
    ----------
    size : str
        One of ``'small'``, ``'base'``, ``'large'``.

    Notes
    -----
    Weights are **not** loaded here.  Call
    ``timm.models.load_checkpoint(backbone.model, path, strict=False)`` after
    construction.

    ``strict=False`` is required because the original Meta checkpoints carry
    zero-valued QKV bias tensors (disabled in timm) and ``bfloat16`` RoPE
    period buffers (recomputed from scratch by timm in ``float32``).
    """

    _TIMM_NAMES = {
        "small": "vit_small_patch16_dinov3.lvd1689m",
        "base": "vit_base_patch16_dinov3.lvd1689m",
        "large": "vit_large_patch16_dinov3.lvd1689m",
    }

    def __init__(self, size):
        super().__init__()

        assert size in self._TIMM_NAMES, (
            f"DINOv3Backbone: unsupported size '{size}'. "
            f"Choose from {list(self._TIMM_NAMES)}."
        )

        self.model = timm.create_model(
            self._TIMM_NAMES[size],
            pretrained=False,
            num_classes=0,
        )

        self.embed_dim = self.model.embed_dim
        self.patch_size = 16

    def get_intermediate_layers(self, x, n):
        """Return intermediate transformer features.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape ``(B, 3, H, W)``.
        n : list[int]
            Zero-based block indices to extract (e.g. ``[2, 5, 8, 11]``).

        Returns
        -------
        tuple[torch.Tensor]
            Tuple of length ``len(n)``, each tensor shaped
            ``(B, N_patches, embed_dim)`` with layer-norm applied and
            CLS + register tokens removed.
        """
        outputs = self.model.forward_intermediates(
            x,
            indices=n,
            return_prefix_tokens=False,
            norm=True,
            output_fmt="NLC",
            intermediates_only=True,
        )
        # forward_intermediates returns a list of (B, N_patches, C) tensors
        # when output_fmt="NLC" and return_prefix_tokens=False.
        return tuple(outputs)
