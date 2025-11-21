from ninetoothed import Tensor, make, block_size, Symbol
import ninetoothed.language as ntl
import torch
from functools import lru_cache
from vllm.model_executor.layers.ninetoothed_ops.config import debug_log


class RMSWithWeight:

    def arrangement(
        input,
        weight,
        output,
        eps,
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

        output_arranged = output.tile(arrange_shape)
        output_arranged = _squeeze(output_arranged)

        expand_shape = tuple(input.shape[:-1]) + (-1,)
        weight_arranged = weight.tile(arrange_shape).expand(expand_shape)
        weight_arranged = _squeeze(weight_arranged)

        return input_arranged, weight_arranged, output_arranged, eps

    def application(input, weight, output, eps):
        input_square = ntl.cast(input, ntl.float32) * ntl.cast(
            input, ntl.float32)
        input_square_mean = ntl.sum(input_square) / input.shape[-1]
        output = input * ntl.rsqrt(input_square_mean + eps) * weight

    @lru_cache
    def premake(ndim):
        kernel = make(
            RMSWithWeight.arrangement,
            RMSWithWeight.application,
            (
                Tensor(ndim),
                Tensor(ndim),
                Tensor(ndim),
                Tensor(0),
            ),
        )
        return kernel


class RMSNoWeight:

    def arrangement(input, output, eps):
        BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)
        ndim = len(input.shape)
        arrange_shape = tuple(1 for _ in range(ndim - 1)) + (BLOCK_SIZE,)

        def _squeeze(x):
            for _ in range(ndim - 1):
                x.dtype = x.dtype.squeeze(0)
            return x

        input_arranged = input.tile(arrange_shape)
        input_arranged = _squeeze(input_arranged)

        output_arranged = output.tile(arrange_shape)
        output_arranged = _squeeze(output_arranged)

        return input_arranged, output_arranged, eps

    def application(input, output, eps):
        input_square = ntl.cast(input, ntl.float32) * ntl.cast(
            input, ntl.float32)
        input_square_mean = ntl.sum(input_square) / input.shape[-1]
        output = input * ntl.rsqrt(input_square_mean + eps)

    @lru_cache
    def premake(ndim):
        kernel = make(
            RMSNoWeight.arrangement,
            RMSNoWeight.application,
            (
                Tensor(ndim),
                Tensor(ndim),
                Tensor(0),
            ),
        )
        return kernel


class RMSResidualWithWeight:

    def arrangement(
        input,
        weight,
        output,
        residual,
        eps,
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

        expand_shape = tuple(input.shape[:-1]) + (-1,)
        weight_arranged = weight.tile(arrange_shape).expand(expand_shape)
        weight_arranged = _squeeze(weight_arranged)

        return input_arranged, weight_arranged, output_arranged, res_arranged, eps

    def application(input, weight, output, residual, eps):
        input = input + residual
        residual = input
        input_square = ntl.cast(input, ntl.float32) * ntl.cast(
            input, ntl.float32)
        input_square_mean = ntl.sum(input_square) / input.shape[-1]
        output = input * ntl.rsqrt(input_square_mean + eps) * weight

    @lru_cache
    def premake(ndim):
        kernel = make(
            RMSResidualWithWeight.arrangement,
            RMSResidualWithWeight.application,
            (
                Tensor(ndim),
                Tensor(ndim),
                Tensor(ndim),
                Tensor(ndim),
                Tensor(0),
            ),
        )
        return kernel


class RMSResidualNoWeight:

    def arrangement(input, output, residual, eps):
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

        return input_arranged, output_arranged, res_arranged, eps

    def application(input, output, residual, eps):
        input = input + residual
        residual = input
        input_square = ntl.cast(input, ntl.float32) * ntl.cast(
            input, ntl.float32)
        input_square_mean = ntl.sum(input_square) / input.shape[-1]
        output = input * ntl.rsqrt(input_square_mean + eps)

    @lru_cache
    def premake(ndim):
        kernel = make(
            RMSResidualNoWeight.arrangement,
            RMSResidualNoWeight.application,
            (
                Tensor(ndim),
                Tensor(ndim),
                Tensor(ndim),
                Tensor(0),
            ),
        )
        return kernel


def forward_nt(input, weight, residual, eps=1e-5, inplace=False):
    debug_log(
        f"\033[31m x:{input.shape},  weight:{weight.shape if weight is not None else None}, res:{residual.shape if residual is not None else None} \033[0m"
    )
    ndim = input.dim()
    output = input
    if not inplace:
        output = torch.empty_like(input)

    if weight is not None:
        assert weight.ndim == 1
        weight = weight.view((1,) * (ndim - 1) + (-1,))
        if residual is not None:
            RMSResidualWithWeight.premake(ndim)(input,
                                                weight,
                                                output,
                                                residual,
                                                eps,
                                                BLOCK_SIZE=input.shape[-1])
        else:
            RMSWithWeight.premake(ndim)(input,
                                        weight,
                                        output,
                                        eps,
                                        BLOCK_SIZE=input.shape[-1])
    else:
        if residual is not None:
            RMSResidualNoWeight.premake(ndim)(input,
                                              output,
                                              residual,
                                              eps,
                                              BLOCK_SIZE=input.shape[-1])
        else:
            RMSNoWeight.premake(ndim)(input,
                                      output,
                                      eps,
                                      BLOCK_SIZE=input.shape[-1])

    if residual is None:
        return output
    else:
        return output, residual