from typing import (
    List,
    Union,
    Optional,
    Sequence,
    Tuple,
)

from .arith import ArithValue
from .utils import Annot

# noinspection PyUnresolvedReferences
from ..mlir._mlir._mlir_libs._nelli_mlir import MemRefValue
from ..mlir._mlir.dialects import memref
from ..mlir._mlir.dialects._ods_common import (
    get_op_result_or_value,
    get_op_results_or_values,
)
from ..mlir._mlir.ir import Type, Value, F64Type, Operation, OpView, MemRefType


class LoadOp(memref.LoadOp):
    def __init__(
        self,
        memref: Union[Operation, OpView, Value],
        indices: Optional[Union[Operation, OpView, Sequence[Value]]] = None,
    ):
        super().__init__(memref, indices)


class StoreOp(memref.StoreOp):
    """Specialization for the MemRef store operation."""

    def __init__(
        self,
        memref: Union[Operation, OpView, Value],
        value: Union[Operation, OpView, Value],
        indices: Optional[Union[Operation, OpView, Sequence[Value]]] = None,
        *,
        loc=None,
        ip=None,
    ):
        memref_resolved = get_op_result_or_value(memref)
        value_resolved = get_op_result_or_value(value)
        indices_resolved = [] if indices is None else get_op_results_or_values(indices)
        super().__init__(
            value_resolved, memref_resolved, indices_resolved, loc=loc, ip=ip
        )


class AllocaOp(memref.AllocaOp):
    def __init__(
        self,
        dim_sizes: Union[List[int], Tuple[int]],
        el_type: Type = None,
        *,
        loc=None,
        ip=None,
    ):
        if el_type is None:
            el_type = F64Type.get()
        assert dim_sizes
        assert isinstance(dim_sizes[0], int)

        # TODO(max): this goes in dynamic/symbolic sizes
        # if isinstance(dim_sizes[0], (Operation, OpView, Value)):
        #     dim_sizes = [] if dim_sizes is None else _get_op_results_or_values(dim_sizes)

        res_type = MemRefType.get(dim_sizes, el_type)
        super().__init__(res_type, [], [], loc=loc, ip=ip)


class MemRefValue(MemRefValue):
    most_recent_store: StoreOp = None

    @staticmethod
    def alloca(dim_sizes: Union[list[int], tuple[int, ...]], el_type: Type):
        return MemRefValue(AllocaOp(dim_sizes, el_type).memref)

    def __class_getitem__(
        cls, dim_sizes_el_type: Tuple[Union[list[int], tuple[int, ...]], Type]
    ):
        assert (
            len(dim_sizes_el_type) == 2
        ), f"wrong dim_sizes_el_type: {dim_sizes_el_type}"
        dim_sizes, el_type = dim_sizes_el_type
        assert all(
            isinstance(t, int) for t in dim_sizes[:-1]
        ), f"wrong type T args for tensor: {dim_sizes}"
        assert isinstance(el_type, Type), f"wrong type T args for tensor: {el_type}"
        return Annot(cls, MemRefType.get(dim_sizes, el_type))

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = tuple([item])
        return ArithValue(LoadOp(self, item).result)

    def __setitem__(self, indices, value):
        if not isinstance(indices, tuple):
            indices = tuple([indices])
        # store op has no result...
        self.most_recent_store = StoreOp(self, value, indices)


def load(memref_, indices) -> ArithValue:
    return ArithValue(LoadOp(memref_, indices).result)
