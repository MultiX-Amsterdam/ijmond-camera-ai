import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Weights are **not** loaded here.  Download the pre-converted timm safetensors
    from HuggingFace (``timm/vit_{size}_patch16_dinov3.lvd1689m``) and save as
    ``pretrained/dinov3_{size}.safetensors``.  Then call
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

        # CLS attention hook state (populated only when enable_cls_attn_hook() is called)
        self._last_cls_attn = None
        self._attn_hook_handle = None

    def enable_cls_attn_hook(self):
        """Register a forward hook on the last transformer block to capture CLS attention.

        Disables fused (flash) attention for the last block so that explicit
        post-softmax weights are computed and made accessible.  Call this once
        after model construction when AWL is enabled.  Idempotent: calling
        again replaces the existing hook.
        """
        if self._attn_hook_handle is not None:
            self._attn_hook_handle.remove()

        attn_module = self.model.blocks[-1].attn
        # Disable fused SDPA so that the explicit softmax path runs and
        # attn_drop receives the attention matrix as its input.
        attn_module.fused_attn = False

        def _hook(module, inp, output):
            # inp[0]: (B, num_heads, seq_len, seq_len) post-softmax attention
            self._last_cls_attn = inp[0].detach()

        self._attn_hook_handle = attn_module.attn_drop.register_forward_hook(_hook)

    def get_last_cls_attn(self, h, w):
        """Return the last captured CLS attention map upsampled to (h, w).

        Must be called after a forward pass when ``enable_cls_attn_hook()``
        has been called.  Returns None and warns if the hook did not fire
        (e.g. the backbone was not called during the forward pass).

        Parameters
        ----------
        h : int
            Target height in pixels.
        w : int
            Target width in pixels.

        Returns
        -------
        torch.Tensor or None
            Shape ``(B, h, w)``, values in [0, 1].  None if hook did not fire.
        """
        if self._last_cls_attn is None:
            import warnings
            warnings.warn(
                "DINOv3Backbone: CLS attention hook did not fire. "
                "Ensure enable_cls_attn_hook() was called and a forward pass ran.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        attn = self._last_cls_attn  # (B, num_heads, seq_len, seq_len)
        num_prefix = self.model.blocks[-1].attn.num_prefix_tokens
        # CLS token is index 0; patch tokens start at num_prefix
        cls_to_patches = attn[:, :, 0, num_prefix:]  # (B, num_heads, num_patches)
        # Average over heads
        cls_attn = cls_to_patches.mean(dim=1)  # (B, num_patches)

        B = cls_attn.shape[0]
        h_p = h // self.patch_size
        w_p = w // self.patch_size
        cls_attn = cls_attn.reshape(B, 1, h_p, w_p)
        cls_attn = F.interpolate(cls_attn, size=(h, w), mode="bilinear", align_corners=False)
        return cls_attn.squeeze(1)  # (B, h, w)

    def get_last_cls_attn_per_head(self, h, w):
        """Return per-head CLS attention maps, upsampled to (h, w).

        Useful for visualisation.  Returns None if the hook did not fire.

        Parameters
        ----------
        h : int
            Target height.
        w : int
            Target width.

        Returns
        -------
        torch.Tensor or None
            Shape ``(B, num_heads, h, w)``.
        """
        if self._last_cls_attn is None:
            return None

        attn = self._last_cls_attn  # (B, num_heads, seq_len, seq_len)
        num_prefix = self.model.blocks[-1].attn.num_prefix_tokens
        cls_to_patches = attn[:, :, 0, num_prefix:]  # (B, num_heads, num_patches)

        B, nh, _ = cls_to_patches.shape
        h_p = h // self.patch_size
        w_p = w // self.patch_size
        maps = cls_to_patches.reshape(B, nh, h_p, w_p)
        maps = F.interpolate(maps, size=(h, w), mode="bilinear", align_corners=False)
        return maps  # (B, num_heads, h, w)

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
