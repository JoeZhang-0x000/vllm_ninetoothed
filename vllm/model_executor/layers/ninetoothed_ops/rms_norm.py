from ninetoothed import Tensor, make, block_size, Symbol
import ninetoothed.language as ntl
import torch
from functools import lru_cache
from vllm.model_executor.layers.ninetoothed_ops.config import debug_log


class RMS:
    def arrangement(
        input,
        weight,
        output,
        residual,
        eps,
        USE_WEIGHT=False,
        USE_RESIDUAL=False,
    ):
        BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)
        ndim = len(input.shape)
        arrange_shape = tuple(1 for _ in range(ndim - 1)) + (BLOCK_SIZE,)

        def _squeeze(x):
            for _ in range(ndim - 1):
                x.dtype = x.dtype.squeeze(0)
            return x

        input_arranged = input.tile(arrange_shape)
        input_arranged = _squeeze(input_arranged)

        res_arranged = residual.tile(arrange_shape)
        res_arranged = _squeeze(res_arranged)

        output_arranged = output.tile(arrange_shape)
        output_arranged = _squeeze(output_arranged)

        weight_arranged = weight
        if USE_WEIGHT:
            expand_shape = tuple(input.shape[:-1]) + (-1,)
            weight_arranged = weight.tile(arrange_shape).expand(expand_shape)
            weight_arranged = _squeeze(weight_arranged)

        return (
            input_arranged,
            weight_arranged,
            output_arranged,
            res_arranged,
            eps,
            USE_WEIGHT,
            USE_RESIDUAL,
        )

    def application(input, weight, output, residual, eps, USE_WEIGHT, USE_RESIDUAL):
        if USE_RESIDUAL:
            input = ntl.cast(input, ntl.float32) + ntl.cast(residual, ntl.float32)
            residual = ntl.cast(input, residual.dtype)
        input_square = ntl.cast(input, ntl.float32) * ntl.cast(input, ntl.float32)
        input_square_mean = ntl.sum(input_square) / input.shape[-1]
        output = input * ntl.rsqrt(input_square_mean + eps)
        if USE_WEIGHT:
            output = output * weight

    @lru_cache(1)
    def premake(ndim, USE_WEIGHT=False, USE_RESIDUAL=False):
        tensors = (
            # input
            Tensor(ndim),
            # weight
            Tensor(ndim) if USE_WEIGHT else Tensor(0, constexpr=True),
            # output
            Tensor(ndim),
            # residual
            Tensor(ndim),
            # eps
            Tensor(0),
            # USE_WEIGHT
            Tensor(0, constexpr=True),
            # USE_RESIDUAL
            Tensor(0, constexpr=True),
        )
        kernel = make(
            RMS.arrangement,
            RMS.application,
            tensors,
        )
        return kernel


def forward_nt(input, weight, residual, eps, inplace=False):
    debug_log(
        f"\033[31m x:{input.shape},  weight:{weight.shape if weight is not None else None}, res:{residual.shape if residual is not None else None} eps: {eps} \033[0m"
    )
    ndim = input.dim()
    output = input
    use_weight = weight is not None
    use_residual = residual is not None
    if not inplace:
        output = torch.empty_like(input)

    if use_weight:
        assert weight.ndim == 1
        weight = weight.view((1,) * (ndim - 1) + (-1,))

    RMS.premake(ndim, use_weight, use_residual)(
        input,
        weight if use_weight else 0,
        output,
        residual if use_residual else torch.zeros_like(input),
        eps,
        use_weight,
        use_residual,
        BLOCK_SIZE=input.shape[-1],
    )

    if not use_residual:
        return output
    else:
        return output, residual
