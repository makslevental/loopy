import contextlib
import os
import sys
import tempfile
from contextlib import ExitStack
from functools import wraps
from io import StringIO
from types import FunctionType
from typing import Optional, Sequence, Union

from ._mlir import ir
from ._mlir._mlir_libs._mlir.ir import (
    Attribute,
    ArrayAttr,
    IntegerAttr,
    IntegerType,
)
from ._mlir.ir import (
    StringAttr,
    register_attribute_builder,
    DenseI64ArrayAttr,
    Context,
)
from ._mlir.passmanager import PassManager


def _get_int64_attr(value: Union[int, Attribute]) -> IntegerAttr:
    if isinstance(value, int):
        return IntegerAttr.get(IntegerType.get_signless(64), value)
    return value


class NelliMlirCompilerError(Exception):
    def __init__(self, value: str):
        super().__init__()
        self.value = value

    def __str__(self) -> str:
        return self.value


def get_module_name_for_debug_dump(module):
    if "nelli.debug_module_name" not in module.operation.attributes:
        return "UnnammedModule"
    return StringAttr(module.operation.attributes["nelli.debug_module_name"]).value


def run_pipeline(
    module,
    pipeline: str,
    description: Optional[str] = None,
    enable_ir_printing=False,
    print_pipeline=False,
):
    """Runs `pipeline` on `module`, with a nice repro report if it fails."""
    module_name = get_module_name_for_debug_dump(module)
    try:
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        # Lower module in place to make it ready for compiler backends.
        with ExitStack() as stack:
            stack.enter_context(module.context)
            asm_for_error_report = module.operation.get_asm(
                large_elements_limit=10,
                enable_debug_info=True,
            )
            pm = PassManager.parse(pipeline)
            if print_pipeline:
                print(pm)
            if enable_ir_printing:
                stack.enter_context(disable_multithreading())
                pm.enable_ir_printing()

            pm.run(module.operation)
    except Exception as e:
        print(e, file=sys.stderr)
        filename = os.path.join(tempfile.gettempdir(), module_name + ".mlir")
        with open(filename, "w") as f:
            f.write(asm_for_error_report)
        debug_options = "-mlir-print-ir-after-all -mlir-disable-threading"
        # Put something descriptive here even if description is empty.
        description = description or f"{module_name} compile"

        message = f"""\
            {description} failed with the following diagnostics:

            {'*' * 80}
            {sys.stderr.getvalue().strip()}
            {'*' * 80}

            For developers, the error can be reproduced with:
            $ mlir-opt {debug_options} -pass-pipeline='{pipeline}' {filename}
            """
        trimmed_message = "\n".join([m.lstrip() for m in message.split("\n")])
        raise NelliMlirCompilerError(trimmed_message) from None
    finally:
        sys.stderr = original_stderr

    return module


def doublewrap(f):
    """
    a decorator decorator, allowing the decorator to be used as:
    @decorator(with, arguments, and=kwargs)
    or
    @decorator
    """

    @wraps(f)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return f(args[0])
        else:
            # decorator arguments
            return lambda realf: f(realf, *args, **kwargs)

    return new_dec


def extract_wrapped(decorated):
    closure = (c.cell_contents for c in decorated.__closure__)
    return next((c for c in closure if isinstance(c, FunctionType)), None)


def get_dense_int64_array_attr(
    values: Sequence[int], context: Optional[Context] = None
) -> DenseI64ArrayAttr:
    from . import DefaultContext

    if context is None:
        context = DefaultContext
    if values is None:
        return DenseI64ArrayAttr.get([], context)
    return DenseI64ArrayAttr.get(values, context)


@register_attribute_builder("DeviceMappingArrayAttr")
def get_device_mapping_array_attr(
    mapping: dict[int, Attribute], context: Optional[Context] = None
) -> ArrayAttr:
    from . import DefaultContext

    if context is None:
        context = DefaultContext
    if isinstance(mapping, ArrayAttr):
        return mapping

    mapping = [mapping[i] for i in range(len(mapping))]
    return ArrayAttr.get(mapping, context=context)


@register_attribute_builder("IndexListArrayAttr")
def get_index_list_array_attr(
    index_list: list[list[int]], context: Optional[Context] = None
) -> ArrayAttr:
    from . import DefaultContext

    if context is None:
        context = DefaultContext
    if isinstance(index_list, ArrayAttr):
        return index_list

    index_list = [
        ArrayAttr.get([_get_int64_attr(v) for v in group], context=context)
        for group in index_list
    ]
    return ArrayAttr.get(index_list, context=context)


# @register_attribute_builder("FlatSymbolRefAttr")
# def get_flat_symbol_ref_attr(
#     symbol: str, context: Optional[Context] = None
# ) -> FlatSymbolRefAttr:
#     from . import DefaultContext
#
#     if context is None:
#         context = DefaultContext
#     return FlatSymbolRefAttr.get(symbol, context)


# def get_symbol_ref_attr(
#     symbols: list[str], context: Optional[Context] = None
# ) -> FlatSymbolRefAttr:
#     from . import DefaultContext
#
#     if context is None:
#         context = DefaultContext
#     qualname = "::".join([f"@{q}" for q in symbols])
#     return Attribute.parse(qualname, context)


F32 = ir.F32Type.get()
F64 = ir.F64Type.get()
I64 = ir.IntegerType.get_signless(64)
I32 = ir.IntegerType.get_signless(32)
Index = ir.IndexType.get()


@contextlib.contextmanager
def allow_unregistered_dialects(context=None):
    from . import DefaultContext

    if context is None:
        context = DefaultContext
    context.allow_unregistered_dialects = True
    yield
    context.allow_unregistered_dialects = False


@contextlib.contextmanager
def enable_multithreading(context=None):
    from . import DefaultContext

    if context is None:
        context = DefaultContext
    context.enable_multithreading(True)
    yield
    context.enable_multithreading(False)


@contextlib.contextmanager
def disable_multithreading(context=None):
    from . import DefaultContext

    if context is None:
        context = DefaultContext

    context.enable_multithreading(False)
    yield
    context.enable_multithreading(True)


@contextlib.contextmanager
def enable_debug():
    ir._GlobalDebug.flag = True
    yield
    ir._GlobalDebug.flag = False
