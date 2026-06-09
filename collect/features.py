"""Extract a flat feature dict from a Scheduler before benchmarking."""

from math import prod as _prod
from tinygrad.uop import Ops
from tinygrad.uop.ops import AxisType

_TRANSCENDENTAL = frozenset({Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIPROCAL})
_CAST_OPS = frozenset({Ops.CAST, Ops.BITCAST})


def extract_features(s) -> dict:
    n_transcendental = n_mulacc = n_where = n_cast = 0
    for node in s.ast.toposort():
        op = node.op
        if op in _TRANSCENDENTAL:
            n_transcendental += 1
        elif op is Ops.MULACC:
            n_mulacc += 1
        elif op is Ops.WHERE:
            n_where += 1
        elif op in _CAST_OPS:
            n_cast += 1

    full_shape = s.full_shape
    shape_len = s.shape_len
    axis_types = s.axis_types

    def _size(*types):
        vals = [full_shape[i] for i in s.axes_of(*types)]
        return int(_prod(vals)) if vals else 1

    # stride matrix: for each buffer, what stride does each loop axis have in its index?
    rngs = s.rngs
    rng_to_idx = {id(r): i for i, r in enumerate(rngs)}
    buf_bytes: list[int] = []
    stride_matrix: list[list[int]] = []

    for buf in s.bufs:
        param = buf.src[0]
        buf_bytes.append(int(param.max_numel()) * param.dtype.base.itemsize)

        strides = [0] * shape_len
        idx = buf.src[1].get_idx()
        for term in idx.split_uop(Ops.ADD):
            if id(term) in rng_to_idx:
                strides[rng_to_idx[id(term)]] = 1
            elif term.op is Ops.MUL:
                a, b = term.src
                if id(a) in rng_to_idx and b.op is Ops.CONST:
                    strides[rng_to_idx[id(a)]] = int(b.arg)
                elif id(b) in rng_to_idx and a.op is Ops.CONST:
                    strides[rng_to_idx[id(b)]] = int(a.arg)
        stride_matrix.append(strides)

    dtype = s.bufs[0].src[0].dtype.base.name if s.bufs else 'unknown'

    reduceop = s.reduceop
    has_reduce = reduceop is not None

    ren = s.ren
    global_max_list = getattr(ren, 'global_max', None)

    return {
        'dtype': dtype,
        'shape_len': shape_len,
        'full_shape': [int(x) if isinstance(x, int) else str(x) for x in full_shape],
        'axis_types': [at.name for at in axis_types],
        'global_size': _size(AxisType.GLOBAL),
        'local_size': _size(AxisType.LOCAL),
        'reduce_size': _size(AxisType.REDUCE),
        'warp_size': _size(AxisType.WARP),
        'upcast_size': int(s.upcast_size()),
        'has_reduce': has_reduce,
        'reduce_op': reduceop.arg[0].name if has_reduce else None,
        'reduce_dtype': reduceop.dtype.name if has_reduce else None,
        'upcasted': s.upcasted,
        'group_for_reduces': s.group_for_reduces,
        'dont_use_locals': s.dont_use_locals,
        'n_mulacc': n_mulacc,
        'n_transcendental': n_transcendental,
        'n_where': n_where,
        'n_cast': n_cast,
        'buf_bytes': buf_bytes,
        'stride_matrix': stride_matrix,
        'device': ren.target.device,
        'shared_max': getattr(ren, 'shared_max', 0),
        'global_max': global_max_list[0] if global_max_list else 0,
    }
