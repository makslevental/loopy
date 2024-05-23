from typing import Optional, Union, Sequence

from ._mlir.dialects._ods_common import get_op_results_or_values
from .arith import ArithValue, constant
from .utils import doublewrap, get_dense_int64_array_attr
from ._mlir.dialects import scf
from ._mlir.ir import InsertionPoint, IndexType, Operation, OpView, Value


class IfOp(scf.IfOp):
    def __init__(self, cond, *, loc=None, ip=None):
        super().__init__(cond, hasElse=False, loc=None, ip=None)

    def add_else(self):
        self.regions[1].blocks.append(*[])
        return self.else_block


_current_if_op: list[IfOp] = []
_if_ip: InsertionPoint = None


def scf_if(cond: ArithValue):
    assert isinstance(cond, ArithValue)
    global _if_ip, _current_if_op
    if_op = IfOp(cond)
    cond.owner.move_before(if_op)
    _current_if_op.append(if_op)
    _if_ip = InsertionPoint(if_op.then_block)
    _if_ip.__enter__()
    return True


def scf_else():
    global _if_ip, _current_if_op
    _if_ip = InsertionPoint(_current_if_op[-1].add_else())
    _if_ip.__enter__()
    return True


def scf_else_if(cond):
    global _if_ip, _current_if_op
    _if_ip = InsertionPoint(_current_if_op[-1].add_else())
    _if_ip.__enter__()
    return scf_if(cond)


def scf_endif_branch():
    global _if_ip
    scf.YieldOp([])
    _if_ip.__exit__(None, None, None)


def scf_endif():
    global _current_if_op
    _current_if_op.pop()


_for_ip = None


def scf_range(start, stop=None, step=1):
    global _for_ip
    if stop is None:
        stop = start
        start = 0

    if isinstance(start, int):
        start = constant(start, index=True)
    if isinstance(stop, int):
        stop = constant(stop, index=True)
    if isinstance(step, int):
        step = constant(step, index=True)
    for_op = scf.ForOp(start, stop, step)
    _for_ip = InsertionPoint(for_op.body)
    _for_ip.__enter__()
    return [ArithValue(for_op.induction_variable)]


def end_for():
    scf.YieldOp([])
    global _for_ip
    _for_ip.__exit__(None, None, None)


# // CHECK: _ODS_OPERAND_SEGMENTS = [-1,1,0,]
# def AttrSizedOperandsOp : TestOp<"attr_sized_operands",
#                                  [AttrSizedOperandSegments]> {
#   // CHECK: def __init__(self, variadic1, non_variadic, *, variadic2=None, loc=None, ip=None):
#   // CHECK:   operands = []
#   // CHECK:   results = []
#   // CHECK:   attributes = {}
#   // CHECK:   regions = None
#   // CHECK:   operands.append(_get_op_results_or_values(variadic1))
#   // CHECK:   operands.append(_get_op_result_or_value(non_variadic))
#   // CHECK:   operands.append(_get_op_result_or_value(variadic2) if variadic2 is not None else None)
#   // CHECK:   _ods_successors = None
#   // CHECK:   super().__init__(self.build_generic(
#   // CHECK:     attributes=attributes, results=results, operands=operands,
#   // CHECK:     successors=_ods_successors, regions=regions, loc=loc, ip=ip))


class ParallelOp(scf.ParallelOp):
    def __init__(
        self,
        lower_bounds,
        upper_bounds,
        steps,
        # iter_args: Optional[Union[Operation, OpView, Sequence[Value]]] = None,
        *,
        loc=None,
        ip=None,
    ):
        assert len(lower_bounds) == len(upper_bounds) == len(steps)
        inits = []
        results = []
        # error: 'scf.parallel' op expects the same number of induction variables: 1 as bound and step values: 2
        iv_types = [IndexType.get()] * len(lower_bounds)
        super().__init__(
            self.build_generic(
                regions=1,
                results=results,
                operands=[
                    get_op_results_or_values(o)
                    for o in [lower_bounds, upper_bounds, steps, inits]
                ],
                loc=loc,
                ip=ip,
            )
        )
        self.regions[0].blocks.append(*iv_types, *results)

    @property
    def body(self):
        """Returns the body (block) of the loop."""
        return self.regions[0].blocks[0]

    @property
    def induction_variables(self):
        """Returns the induction variable of the loop."""
        return self.body.arguments


_parfor_ip = None


def par_range(starts, stops, steps=None):
    global _parfor_ip
    assert len(starts) == len(stops)
    if steps is None:
        steps = [1] * len(starts)
    for args in [starts, stops, steps]:
        for i, a in enumerate(args):
            if isinstance(a, int):
                args[i] = constant(a, index=True)
    for_op = ParallelOp(starts, stops, steps)
    _parfor_ip = InsertionPoint(for_op.body)
    _parfor_ip.__enter__()
    return [
        tuple(
            ArithValue(a) for a in get_op_results_or_values(for_op.induction_variables)
        )
    ]


def end_parfor():
    scf.YieldOp([])
    global _parfor_ip
    _parfor_ip.__exit__(None, None, None)


"""
          %10 = scf.forall (%arg0, %arg1) in (%c16, %c32) shared_outs(%arg2 = %filled) -> (tensor<16x128xf32>) {
            %11 = affine.apply #map2(%arg1)
            %extracted_slice = tensor.extract_slice %filled[%arg0, %11] [1, 4] [1, 1] : tensor<16x128xf32> to tensor<1x4xf32>
            %extracted_slice_2 = tensor.extract_slice %arg2[%arg0, %11] [1, 4] [1, 1] : tensor<16x128xf32> to tensor<1x4xf32>
            %13 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice : tensor<1x4xf32>) outs(%extracted_slice_2 : tensor<1x4xf32>) {
            ^bb0(%in: f32, %out: f32):
              %res = arith.addf %in, %in: f32
              linalg.yield %res : f32
            } -> tensor<1x4xf32>
            scf.forall.in_parallel {
              tensor.parallel_insert_slice %13 into %arg2[%arg0, %11] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<16x128xf32>
            }
          }
"""


class ForallOp(scf.ForallOp):
    def __init__(
        self,
        lower_bounds,
        upper_bounds,
        steps,
        shared_outs: Optional[Union[Operation, OpView, Sequence[Value]]] = None,
        *,
        loc=None,
        ip=None,
    ):
        assert len(lower_bounds) == len(upper_bounds) == len(steps)
        results = [o.type for o in shared_outs]
        iv_types = [IndexType.get()] * len(lower_bounds)
        dynamic_lower_bounds = []
        dynamic_upper_bounds = []
        dynamic_steps = []
        attributes = {
            "staticLowerBound": get_dense_int64_array_attr(lower_bounds),
            "staticUpperBound": get_dense_int64_array_attr(upper_bounds),
            "staticStep": get_dense_int64_array_attr(steps),
        }

        super().__init__(
            self.build_generic(
                regions=1,
                results=results,
                operands=[
                    get_op_results_or_values(o)
                    for o in [
                        dynamic_lower_bounds,
                        dynamic_upper_bounds,
                        dynamic_steps,
                        # lower_bounds,
                        # upper_bounds,
                        # steps,
                        shared_outs,
                    ]
                ],
                attributes=attributes,
                loc=loc,
                ip=ip,
            )
        )
        self.regions[0].blocks.append(*iv_types, *results)

    @property
    def body(self):
        """Returns the body (block) of the loop."""
        return self.regions[0].blocks[0]

    @property
    def arguments(self):
        """Returns the induction variable of the loop."""
        return self.body.arguments


class InParallelOp(scf.InParallelOp):
    def __init__(self, *, loc=None, ip=None):
        super().__init__(
            self.build_generic(regions=1, results=[], operands=[], loc=loc, ip=ip)
        )
        self.regions[0].blocks.append(*[])

    @property
    def body(self):
        """Returns the body (block) of the loop."""
        return self.regions[0].blocks[0]


@doublewrap
def forall(f, lower_bounds, upper_bounds=None, steps=None, shared_outs=None):
    if upper_bounds is None:
        upper_bounds = lower_bounds
        lower_bounds = [0] * len(upper_bounds)
    if steps is None:
        steps = [1] * len(upper_bounds)
    if shared_outs is None:
        shared_outs = []

    lower_bounds, upper_bounds, steps, shared_outs = (
        list(lower_bounds),
        list(upper_bounds),
        list(steps),
        list(shared_outs),
    )

    launch_op = ForallOp(lower_bounds, upper_bounds, steps, shared_outs)
    with InsertionPoint(launch_op.body):
        ivs = list(map(ArithValue, launch_op.arguments[: len(lower_bounds)]))
        shared_outs_ = launch_op.arguments[len(lower_bounds) :]
        assert len(shared_outs_) == len(shared_outs)
        parallel_insert_slice = f(ivs=ivs, shared_outs=shared_outs_)
        with InsertionPoint(InParallelOp().body):
            parallel_insert_slice()
