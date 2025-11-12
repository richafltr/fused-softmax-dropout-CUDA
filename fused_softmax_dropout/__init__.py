import torch
from . import _C  # from compiled extension


def fused_softmax_dropout(x, mask=None, p: float = 0.1, training: bool = True):
    """
    Fused softmax + dropout operation for CUDA tensors.
    
    Args:
        x: Input tensor of shape (batch, seq, d_model) on CUDA
        mask: Optional mask tensor (0.0 for masked, 1.0 for unmasked). 
              If provided, masked positions are treated as -inf in softmax.
        p: Dropout probability (default: 0.1)
        training: Whether in training mode (default: True)
    
    Returns:
        Output tensor of same shape as x
    """
    if not x.is_cuda:
        raise ValueError("fused_softmax_dropout: input must be on CUDA")
    if x.dtype != torch.float32:
        raise ValueError("Currently only float32 is supported")
    if mask is not None:
        mask = mask.to(x.device)
    return _C.fused_softmax_dropout(x, mask, p, training)

