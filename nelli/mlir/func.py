import ast
import inspect
import logging
from textwrap import dedent
from types import FunctionType, CodeType

from bytecode import ConcreteBytecode, ConcreteInstr

from ._mlir._mlir_libs._mlir.ir import (
    InsertionPoint,
    Value,
    OpView,
    Operation,
    TypeAttr,
    StringAttr,
)
from .affine import RankedAffineMemRefValue
from .affine import end_for as affine_endfor, affine_range
from .annot import Annot
from .arith import ArithValue
from .memref import MemRefValue
from .omp.omp import ws_loop as omp_range, end_for as omp_endfor
from .scf import (
    scf_endif_branch,
    scf_if,
    scf_else,
    scf_endif,
    scf_range,
    end_for as scf_endfor,
    par_range as scf_par_range,
    end_parfor as scf_end_parfor,
)
from .tensor import TensorValue
from .utils import doublewrap, extract_wrapped
from ..mlir._mlir.dialects import func as func_dialect
from ..mlir._mlir.dialects._ods_common import (
    get_op_result_or_value,
    get_op_results_or_values,
)
from ..mlir._mlir.ir import (
    Type as MLIRType,
    MemRefType,
    RankedTensorType,
    FunctionType as MLIRFunctionType,
    FlatSymbolRefAttr,
    UnitAttr,
)

logger = logging.getLogger(__name__)


def ast_call(name, args=None):
    if args is None:
        args = []
    return ast.Call(
        func=ast.Name(id=name, ctx=ast.Load()),
        args=args,
        keywords=[],
    )


class InsertEndFors(ast.NodeTransformer):
    def __init__(self, endfor):
        self.endfor = endfor

    def visit_For(self, node):
        for i, b in enumerate(node.body):
            node.body[i] = self.visit(b)
        node.body.append(ast.Expr(ast_call(self.endfor.__name__)))
        return node


def wrap_if_test(test):
    assert isinstance(test, ast.Compare), f"unexpected if test: {ast.dump(test)}"
    assert isinstance(
        test.left, ast.Name
    ), f"unexpected if test lhs: {ast.dump(test.left)}"
    assert all(
        isinstance(c, (ast.Name, ast.Constant, ast.Call)) for c in test.comparators
    ), f"unexpected if test comparators: {test.comparators}"
    comparators = []

    for c in test.comparators:
        if isinstance(c, ast.Name):
            comparators.append(ast_call(ArithValue.__name__, args=[c]))
        # elif isinstance(c, ast.Constant):
        #     comparators.append(ast_call(constant.__name__, args=[c]))
        elif isinstance(c, ast.Call):
            assert isinstance(c.func, ast.Name)
            assert c.func.id == "constant"
            comparators.append(c)
        else:
            raise RuntimeError(f"unknown kind of comparator c={ast.unparse(c)}")

    test = ast.Compare(
        ast_call(ArithValue.__name__, args=[test.left]),
        ops=test.ops,
        comparators=comparators,
    )
    return ast_call(scf_if.__name__, args=[test])


class InsertEndIfs(ast.NodeTransformer):
    def visit_If(self, node):
        for i, b in enumerate(node.body):
            node.body[i] = self.visit(b)
        for i, b in enumerate(node.orelse):
            node.orelse[i] = self.visit(b)

        node.test = wrap_if_test(node.test)
        # every if branch needs a scf_endif_branch
        node.body.append(ast.Expr(ast_call(scf_endif_branch.__name__)))
        # no else, then need to end the whole if in the body of the true branch
        if not node.orelse:
            node.body.append(ast.Expr(ast_call(scf_endif.__name__)))
        else:
            # otherwise end the if after the else branch
            node.orelse.insert(0, ast.Expr(ast_call(scf_else.__name__)))
            node.orelse.append(ast.Expr(ast_call(scf_endif_branch.__name__)))
            node.orelse.append(ast.Expr(ast_call(scf_endif.__name__)))

        return node


def bind(func, instance, as_name=None):
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method


def rewrite_ast(f, range_ctor, endfor):
    tree = ast.parse(dedent(inspect.getsource(f)))
    assert isinstance(
        tree.body[0], ast.FunctionDef
    ), f"unexpected ast node {tree.body[0]}"
    tree = InsertEndFors(endfor=endfor).visit(tree)
    tree = InsertEndIfs().visit(tree)
    tree.body[0].body.append(ast.Return(value=ast.Constant(value=None)))

    tree = ast.fix_missing_locations(tree)
    tree = ast.increment_lineno(tree, f.__code__.co_firstlineno - 1)
    module_code_o = compile(tree, f.__code__.co_filename, "exec")
    f_code_o = next(
        c
        for c in module_code_o.co_consts
        if type(c) is CodeType and c.co_name == f.__name__
    )

    # TODO(max): handle other ifs/for loops here
    updated_f = FunctionType(
        code=f_code_o,
        globals={
            **f.__globals__,
            **{
                fr: f.__closure__[i].cell_contents
                for i, fr in enumerate(f.__code__.co_freevars)
            },
            endfor.__name__: endfor,
            "range": range_ctor,
            ArithValue.__name__: ArithValue,
            scf_else.__name__: scf_else,
            scf_if.__name__: scf_if,
            scf_endif_branch.__name__: scf_endif_branch,
            scf_endif.__name__: scf_endif,
        },
        name=f.__name__,
        argdefs=f.__defaults__,
    )
    updated_f.__qualname__ = f.__qualname__
    updated_f.__annotations__ = f.__annotations__

    if inspect.ismethod(f):
        updated_f = bind(updated_f, f.__self__)

    return updated_f


def rewrite_bytecode(f):
    src_lines = inspect.getsource(f).splitlines()
    code = ConcreteBytecode.from_code(f.__code__)
    early_returns = []
    for i, c in enumerate(code):
        if c.name == "RETURN_VALUE":
            early_returns.append(i)

        if c.name in {
            # this is the first test condition jump from python <= 3.10
            "POP_JUMP_IF_FALSE",
            # this is the test condition jump from python >= 3.11
            "POP_JUMP_FORWARD_IF_FALSE",
        }:
            logger.debug(f"subbing pop_jump {i}: {c=}")
            code[i] = ConcreteInstr("POP_TOP", lineno=c.lineno, location=c.location)

        if c.name in {
            # this is the jump after each arm in a conditional
            "JUMP_FORWARD",
            # this is the jump at the end of a for loop
            # "JUMP_BACKWARD",
            # in principle this should be no-oped too but for whatever reason it leads to a stack-size
            # miscalculation (inside bytecode). we don't really need it though because
            # affine_range returns an iterator with length 1
        }:
            # only remove the jump if generated by an if stmt (not a with stmt)
            if "with" not in src_lines[c.lineno - code.first_lineno]:
                logger.debug(f"no-oping jump {i}: {c=}")
                code[i] = ConcreteInstr("NOP", lineno=c.lineno, location=c.location)

    # early returns cause branches in conditionals to not be visited
    for idx in early_returns[:-1]:
        c = code[idx]
        code[idx] = ConcreteInstr("NOP", lineno=c.lineno, location=c.location)

    f_code_o = code.to_code()
    updated_f = FunctionType(
        code=f_code_o,
        globals={
            **f.__globals__,
            **{
                fr: f.__closure__[i].cell_contents
                for i, fr in enumerate(f.__code__.co_freevars)
            },
        },
        name=f.__name__,
        argdefs=f.__defaults__,
    )
    updated_f.__qualname__ = f.__qualname__
    updated_f.__annotations__ = f.__annotations__
    if inspect.ismethod(f):
        updated_f = bind(updated_f, f.__self__)
    return updated_f


def get_qual_name(qualname):
    if ".<locals>." in qualname:
        _, qualname = qualname.split(".<locals>.")
    return qualname


class MLIRFunc:
    _func_op = None

    def __init__(
        self,
        f,
        func_op_ctor=None,
        func_op_terminator=None,
        attributes=None,
        build=True,
        qualname=None,
    ):

        if func_op_ctor is None:
            func_op_ctor = func_dialect.FuncOp
        if func_op_terminator is None:
            func_op_terminator = func_dialect.ReturnOp
        if attributes is None:
            attributes = {}
        self.qualname = qualname

        sig_params = inspect.signature(f).parameters
        params = list(sig_params.values())

        if "self" in sig_params:
            assert params[0].name == "self"
            params.pop(0)

        annots = [p.annotation for p in params]
        assert all(
            isinstance(a, (MLIRType, Annot)) for a in annots
        ), f"wrong func args {annots}"

        self.f = f
        self.attributes = attributes
        self.annots = annots
        self.func_op_ctor = func_op_ctor
        self.func_op_terminator = func_op_terminator
        if build:
            self._func_op = self._build_func_op()

    def _build_func_op(self):
        inputs = [(an.mlir_type if isinstance(an, Annot) else an) for an in self.annots]
        function_type = MLIRFunctionType.get(inputs=inputs, results=[])
        func_op = self.func_op_ctor(name=self.f.__name__, type=function_type)
        if self.attributes is not None:
            for k, v in self.attributes.items():
                if v is None:
                    v = UnitAttr.get()
                func_op.attributes[k] = v

        with InsertionPoint(func_op.add_entry_block()):
            func_args = func_op.entry_block.arguments
            args = list(func_args)
            for i, (annot, arg) in enumerate(zip(self.annots, args)):
                logger.debug(f"{self.f.__name__} arg {i}: {arg}")
                if MemRefType.isinstance(arg.type) or RankedTensorType.isinstance(
                    arg.type
                ):
                    if annot.py_type in {
                        RankedAffineMemRefValue,
                        MemRefValue,
                        TensorValue,
                    }:
                        args[i] = annot.py_type(arg)
                    else:
                        raise RuntimeError(f"unknown annotation: {annot.py_type}")
                else:
                    args[i] = ArithValue(arg)

            return_values = self.f(*args)
            if return_values is None:
                return_values = []
            elif isinstance(return_values, tuple):
                return_values = list(return_values)
            elif isinstance(return_values, Value):
                return_values = [return_values]
            elif isinstance(return_values, OpView):
                return_values = return_values.operation.results
            elif isinstance(return_values, Operation):
                return_values = return_values.results
            else:
                return_values = list(return_values)
            self.func_op_terminator(return_values)

        # Recompute the function type.
        return_types = [v.type for v in return_values]
        function_type = MLIRFunctionType.get(inputs=inputs, results=return_types)
        func_op.attributes["function_type"] = TypeAttr.get(function_type)

        return func_op

    @property
    def func_op(self):
        if self._func_op is None:
            self._func_op = self._build_func_op()
        return self._func_op

    def __call__(self, *args):
        return_types = self.func_op.type.results
        call_op = func_dialect.CallOp(self._func_op, list(args))
        if return_types is None:
            return None
        elif len(return_types) == 1:
            return call_op.result
        else:
            return call_op.results


@doublewrap
def mlir_func(
    f,
    rewrite_ast_=None,
    rewrite_bytecode_=None,
    range_ctor=None,
    func_ctor=None,
    **kwargs,
) -> MLIRFunc:
    if rewrite_ast_ is None:
        rewrite_ast_ = True
    if rewrite_bytecode_ is None:
        rewrite_bytecode_ = True
    if range_ctor is None:
        range_ctor = affine_range
    if func_ctor is None:
        func_ctor = MLIRFunc

    if rewrite_ast_:
        if range_ctor == affine_range:
            endfor = affine_endfor
        elif range_ctor == scf_range:
            endfor = scf_endfor
        elif range_ctor == omp_range:
            endfor = omp_endfor
        elif range_ctor == scf_par_range:
            endfor = scf_end_parfor
        else:
            raise RuntimeError(f"unsupported {range_ctor=}")
        f = rewrite_ast(f, range_ctor=range_ctor, endfor=endfor)

    if rewrite_bytecode_:
        f = rewrite_bytecode(f)

    return func_ctor(f, **kwargs)


@doublewrap
def lazy_mlir_func(*args, **kwargs):
    unwrapped_sequence = extract_wrapped(mlir_func)

    assert (
        "mlir_module" in kwargs
    ), "lazy eval necessitates providing a container module"
    mlir_module = kwargs.pop("mlir_module")

    def wrapped(*call_args):
        with InsertionPoint(mlir_module.body):
            func_op = unwrapped_sequence(*args, **kwargs)
        return func_op(*call_args)

    return wrapped


def call_func(symbol_name, call_args, return_types):
    call_op = func_dialect.CallOp(
        return_types, FlatSymbolRefAttr.get(symbol_name), call_args
    )
    if return_types is None:
        return None
    elif len(return_types) == 1:
        return call_op.result
    else:
        return call_op.results


def declare(
    symbol_name,
    input_annots: list,
    result_annots=None,
    func_op_ctor=func_dialect.FuncOp,
):
    if result_annots is None:
        result_annots = []
    assert all(
        isinstance(a, (MLIRType, Annot)) for a in input_annots
    ), f"wrong func args {input_annots}"
    assert all(
        isinstance(a, (MLIRType, Annot)) for a in result_annots
    ), f"wrong func results {result_annots}"

    input_annots = [a.mlir_type if isinstance(a, Annot) else a for a in input_annots]
    result_mlir_types = [
        a.mlir_type if isinstance(a, Annot) else a for a in result_annots
    ]

    function_type = MLIRFunctionType.get(inputs=input_annots, results=result_mlir_types)
    sym_name = func_op_ctor(
        name=symbol_name, type=function_type, visibility="private"
    ).sym_name

    def callable(*call_args):
        res = call_func(sym_name.value, call_args, result_mlir_types)
        if len(result_mlir_types) == 0:
            return
        elif len(result_mlir_types) == 1:
            res_vals = [get_op_result_or_value(res)]
        else:  # if len(result_mlir_types) > 1:
            res_vals = get_op_results_or_values(res)

        for i, (annot, res_val) in enumerate(zip(result_annots, res_vals)):
            logger.debug(f"{sym_name} result {i}: {res_val}")
            if MemRefType.isinstance(res_val.type) or RankedTensorType.isinstance(
                res_val.type
            ):
                if annot.py_type in {RankedAffineMemRefValue, MemRefValue, TensorValue}:
                    res_vals[i] = annot.py_type(res_val)
                else:
                    raise RuntimeError(f"unknown annotation: {annot.py_type}")
            else:
                res_vals[i] = ArithValue(res_val)

        if len(res_vals) == 1:
            res_vals = res_vals[0]
        return res_vals

    return callable


def visibility_attr(visibility):
    return StringAttr.get(visibility)
