"""
TVM FFI Python bindings for CUDA kernel.
Used when language = "cuda" in config.toml.
"""

import ctypes
import os
import torch

_lib = None


def _load_lib():
    global _lib
    if _lib is None:
        lib_path = os.path.join(os.path.dirname(__file__), "kernel.so")
        _lib = ctypes.CDLL(lib_path)
        _lib.moe_fused_kernel_launcher.restype  = None
        _lib.moe_fused_kernel_launcher.argtypes = [
            ctypes.c_void_p,  # routing_logits
            ctypes.c_void_p,  # routing_bias
            ctypes.c_void_p,  # hidden_states
            ctypes.c_void_p,  # hidden_states_scale
            ctypes.c_void_p,  # gemm1_weights
            ctypes.c_void_p,  # gemm1_weights_scale
            ctypes.c_void_p,  # gemm2_weights
            ctypes.c_void_p,  # gemm2_weights_scale
            ctypes.c_int,     # local_expert_offset
            ctypes.c_float,   # routed_scaling_factor
            ctypes.c_void_p,  # output
            ctypes.c_int,     # seq_len
            ctypes.c_void_p,  # cuda stream
        ]
    return _lib


def kernel(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
):
    lib = _load_lib()
    seq_len = routing_logits.shape[0]
    hidden_size = hidden_states.shape[1]
    output = torch.empty(seq_len, hidden_size, dtype=torch.bfloat16, device=hidden_states.device)
    stream = torch.cuda.current_stream().cuda_stream

    lib.moe_fused_kernel_launcher(
        routing_logits.data_ptr(),
        routing_bias.data_ptr(),
        hidden_states.data_ptr(),
        hidden_states_scale.data_ptr(),
        gemm1_weights.data_ptr(),
        gemm1_weights_scale.data_ptr(),
        gemm2_weights.data_ptr(),
        gemm2_weights_scale.data_ptr(),
        ctypes.c_int(int(local_expert_offset)),
        ctypes.c_float(float(routed_scaling_factor)),
        output.data_ptr(),
        ctypes.c_int(seq_len),
        ctypes.c_void_p(stream),
    )

    return output
