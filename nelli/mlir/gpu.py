import builtins
from typing import Union, Optional, Any

from ._mlir import ir
from ._mlir._mlir_libs._mlir.ir import (
    UnitAttr,
    IndexType,
    Context,
    FlatSymbolRefAttr,
)
from ._mlir.dialects._func_ops_ext import ARGUMENT_ATTRIBUTE_NAME, RESULT_ATTRIBUTE_NAME
from ._mlir.dialects._ods_common import get_default_loc_context
from ._mlir.ir import (
    StringAttr,
    FunctionType,
    TypeAttr,
    ArrayAttr,
    Type,
    Value,
    InsertionPoint,
    register_attribute_builder,
    Attribute,
)
from .arith import ArithValue, constant
from .func import MLIRFunc
from .module import Module
from .scf import scf_range
from .utils import doublewrap
from ..mlir._mlir.dialects import gpu


class LaunchFuncOp(gpu.LaunchFuncOp):
    def __init__(
        self,
        grid_size: tuple[Any, Any, Any],
        block_size: tuple[Any, Any, Any],
        kernel: list[str] = None,
        operands: list[Value] = None,
        asyncDependencies=None,
        dynamicSharedMemorySize: Optional[Value] = None,
        *,
        loc=None,
        ip=None,
    ):
        if asyncDependencies is None:
            asyncDependencies = []
        gridSizeX, gridSizeY, gridSizeZ = grid_size
        blockSizeX, blockSizeY, blockSizeZ = block_size
        _ods_context = get_default_loc_context(loc)
        attributes = {
            "kernel": (
                kernel
                if (
                    issubclass(type(kernel), ir.Attribute)
                    or not ir.AttrBuilder.contains("SymbolRefAttr")
                )
                else ir.AttrBuilder.get("SymbolRefAttr")(kernel, context=_ods_context)
            )
        }

        if asyncDependencies is None:
            results = []  # GPU_AsyncToken
        else:
            results = [gpu_async_token()]  # GPU_AsyncToken
        regions = None
        _ods_context = get_default_loc_context(loc)
        _ods_successors = None
        super().__init__(
            self.build_generic(
                attributes=attributes,
                results=results,
                operands=[
                    asyncDependencies,
                    gridSizeX,
                    gridSizeY,
                    gridSizeZ,
                    blockSizeX,
                    blockSizeY,
                    blockSizeZ,
                    dynamicSharedMemorySize,
                    operands,
                ],
                successors=_ods_successors,
                regions=regions,
                loc=loc,
                ip=ip,
            )
        )


class LaunchOp(gpu.LaunchOp):
    def __init__(
        self,
        grid_size: tuple[Any, Any, Any],
        block_size: tuple[Any, Any, Any],
        asyncDependencies=None,
        dynamicSharedMemorySize: Optional[Value] = None,
        *,
        loc=None,
        ip=None,
    ):
        if asyncDependencies is None:
            results = []  # GPU_AsyncToken
        else:
            results = [gpu_async_token()]  # GPU_AsyncToken
        if asyncDependencies is None:
            asyncDependencies = []
        gridSizeX, gridSizeY, gridSizeZ = grid_size
        blockSizeX, blockSizeY, blockSizeZ = block_size
        _ods_context = get_default_loc_context(loc)
        attributes = {}

        regions = 1
        _ods_context = get_default_loc_context(loc)
        _ods_successors = None
        super().__init__(
            self.build_generic(
                attributes=attributes,
                results=results,
                operands=[
                    asyncDependencies,
                    gridSizeX,
                    gridSizeY,
                    gridSizeZ,
                    blockSizeX,
                    blockSizeY,
                    blockSizeZ,
                    dynamicSharedMemorySize,
                ],
                successors=_ods_successors,
                regions=regions,
                loc=loc,
                ip=ip,
            )
        )
        self.regions[0].blocks.append(*[IndexType.get() for _ in range(12)])

    @property
    def entry_block(self):
        return self.regions[0].blocks[0]

    @property
    def arguments(self):
        return self.entry_block.arguments


@doublewrap
def gpu_launch(f, grid_size, block_size):
    grid_size_, block_size_ = [1] * 3, [1] * 3
    grid_size_[: len(grid_size)], block_size_[: len(grid_size)] = grid_size, block_size
    for size in [grid_size_, block_size_]:
        for i, s in enumerate(size):
            if isinstance(s, int):
                size[i] = constant(s, index=True)
    launch_op = LaunchOp(grid_size_, block_size_)
    with InsertionPoint(launch_op.entry_block):
        f(
            block_ids=tuple(launch_op.arguments[:3]),
            thread_ids=tuple(launch_op.arguments[3:6]),
        )
        gpu.TerminatorOp()


class FuncOp(gpu.GPUFuncOp):
    def __init__(
        self,
        sym_name,
        function_type,
        *,
        sym_visibility=None,
        arg_attrs=None,
        res_attrs=None,
        loc=None,
        ip=None,
    ):
        operands = []
        results = []
        attributes = {}
        regions = None
        _ods_context = get_default_loc_context(loc)
        attributes["sym_name"] = (
            sym_name
            if (
                issubclass(type(sym_name), ir.Attribute)
                or not ir.AttrBuilder.contains("SymbolNameAttr")
            )
            else ir.AttrBuilder.get("SymbolNameAttr")(sym_name, context=_ods_context)
        )
        attributes["function_type"] = (
            function_type
            if (
                issubclass(type(function_type), ir.Attribute)
                or not ir.AttrBuilder.contains("anonymous_394")
            )
            else ir.AttrBuilder.get("anonymous_394")(
                function_type, context=_ods_context
            )
        )
        if sym_visibility is not None:
            attributes["sym_visibility"] = (
                sym_visibility
                if (
                    issubclass(type(sym_visibility), ir.Attribute)
                    or not ir.AttrBuilder.contains("StrAttr")
                )
                else ir.AttrBuilder.get("StrAttr")(sym_visibility, context=_ods_context)
            )
        if arg_attrs is not None:
            attributes["arg_attrs"] = (
                arg_attrs
                if (
                    issubclass(type(arg_attrs), ir.Attribute)
                    or not ir.AttrBuilder.contains("DictArrayAttr")
                )
                else ir.AttrBuilder.get("DictArrayAttr")(
                    arg_attrs, context=_ods_context
                )
            )
        if res_attrs is not None:
            attributes["res_attrs"] = (
                res_attrs
                if (
                    issubclass(type(res_attrs), ir.Attribute)
                    or not ir.AttrBuilder.contains("DictArrayAttr")
                )
                else ir.AttrBuilder.get("DictArrayAttr")(
                    res_attrs, context=_ods_context
                )
            )
        _ods_successors = None
        super().__init__(
            self.build_generic(
                attributes=attributes,
                results=results,
                operands=operands,
                successors=_ods_successors,
                regions=regions,
                loc=loc,
                ip=ip,
            )
        )

    @builtins.property
    def sym_name(self):
        return ir.StringAttr(self.operation.attributes["sym_name"])

    @sym_name.setter
    def sym_name(self, value):
        if value is None:
            raise ValueError("'None' not allowed as value for mandatory attributes")
        self.operation.attributes["sym_name"] = value

    @builtins.property
    def sym_visibility(self):
        if "sym_visibility" not in self.operation.attributes:
            return None
        return ir.StringAttr(self.operation.attributes["sym_visibility"])

    @sym_visibility.setter
    def sym_visibility(self, value):
        if value is not None:
            self.operation.attributes["sym_visibility"] = value
        elif "sym_visibility" in self.operation.attributes:
            del self.operation.attributes["sym_visibility"]

    @sym_visibility.deleter
    def sym_visibility(self):
        del self.operation.attributes["sym_visibility"]

    @builtins.property
    def body(self):
        return self.regions[0]


class FuncOp(FuncOp):
    """Specialization for the func op class."""

    def __init__(
        self, name, type, *, visibility=None, body_builder=None, loc=None, ip=None
    ):
        sym_name = StringAttr.get(str(name))

        # If the type is passed as a tuple, build a FunctionType on the fly.
        if isinstance(type, tuple):
            type = FunctionType.get(inputs=type[0], results=type[1])

        type = TypeAttr.get(type)
        sym_visibility = (
            StringAttr.get(str(visibility)) if visibility is not None else None
        )
        super().__init__(sym_name, type, sym_visibility=sym_visibility, loc=loc, ip=ip)
        if body_builder:
            entry_block = self.add_entry_block()
            with InsertionPoint(entry_block):
                body_builder(self)

    @property
    def is_external(self):
        return len(self.regions[0].blocks) == 0

    @property
    def body(self):
        return self.regions[0]

    @property
    def type(self):
        return FunctionType(TypeAttr(self.attributes["function_type"]).value)

    @property
    def visibility(self):
        return self.attributes["sym_visibility"]

    @property
    def name(self) -> StringAttr:
        return StringAttr(self.attributes["sym_name"])

    @property
    def entry_block(self):
        if self.is_external:
            raise IndexError("External function does not have a body")
        return self.regions[0].blocks[0]

    def add_entry_block(self):
        """
        Add an entry block to the function body using the function signature to
        infer block arguments.
        Returns the newly created block
        """
        if not self.is_external:
            raise IndexError("The function already has an entry block!")
        self.body.blocks.append(*self.type.inputs)
        return self.body.blocks[0]

    @property
    def arg_attrs(self):
        return ArrayAttr(self.attributes[ARGUMENT_ATTRIBUTE_NAME])

    @arg_attrs.setter
    def arg_attrs(self, attribute: Union[ArrayAttr, list]):
        if isinstance(attribute, ArrayAttr):
            self.attributes[ARGUMENT_ATTRIBUTE_NAME] = attribute
        else:
            self.attributes[ARGUMENT_ATTRIBUTE_NAME] = ArrayAttr.get(
                attribute, context=self.context
            )

    @property
    def arguments(self):
        return self.entry_block.arguments

    @property
    def result_attrs(self):
        return self.attributes[RESULT_ATTRIBUTE_NAME]

    @result_attrs.setter
    def result_attrs(self, attribute: ArrayAttr):
        self.attributes[RESULT_ATTRIBUTE_NAME] = attribute


class ModuleOp(gpu.GPUModuleOp):
    @builtins.property
    def sym_name(self):
        if "sym_name" not in self.operation.attributes:
            return None
        return ir.StringAttr(self.operation.attributes["sym_name"])

    @sym_name.setter
    def sym_name(self, value):
        if value is not None:
            self.operation.attributes["sym_name"] = value
        elif "sym_name" in self.operation.attributes:
            del self.operation.attributes["sym_name"]

    @sym_name.deleter
    def sym_name(self):
        del self.operation.attributes["sym_name"]

    @builtins.property
    def sym_visibility(self):
        if "sym_visibility" not in self.operation.attributes:
            return None
        return ir.StringAttr(self.operation.attributes["sym_visibility"])

    @sym_visibility.setter
    def sym_visibility(self, value):
        if value is not None:
            self.operation.attributes["sym_visibility"] = value
        elif "sym_visibility" in self.operation.attributes:
            del self.operation.attributes["sym_visibility"]

    @sym_visibility.deleter
    def sym_visibility(self):
        del self.operation.attributes["sym_visibility"]

    @builtins.property
    def bodyRegion(self):
        return self.regions[0]


class ModuleOp(ModuleOp):
    def __init__(self, *, loc=None, ip=None):
        super().__init__(self.build_generic(results=[], operands=[], loc=loc, ip=ip))
        self.regions[0].blocks.append()

    @property
    def body(self):
        return self.regions[0].blocks[0]


class MLIRFunc(MLIRFunc):
    def __init__(
        self,
        f,
        attributes=None,
        build=True,
        qualname=None,
    ):
        super().__init__(
            f=f,
            func_op_ctor=FuncOp,
            func_op_terminator=gpu.ReturnOp,
            attributes=attributes,
            build=build,
            qualname=qualname,
        )
        self.func_op.operation.attributes["gpu.kernel"] = UnitAttr.get()

    def __call__(self, *operands, grid_size=None, block_size=None):
        for size in [grid_size, block_size]:
            for i, s in enumerate(size):
                if isinstance(s, int):
                    size[i] = constant(s, index=True)
        # hack
        call_op = LaunchFuncOp(
            grid_size,
            block_size,
            kernel=[self.qualname] + [self.f.__name__],
            operands=operands,
        )
        return call_op.result


class Module(Module):
    def __init__(self, **kwargs):
        super().__init__(
            ctor=ModuleOp,
            func_ctor=MLIRFunc,
            range_ctor=scf_range,
            func_qualname=self.__class__.__name__,
            **kwargs,
        )
        with InsertionPoint(self.mlir_module.body):
            gpu.ModuleEndOp()


@register_attribute_builder("GPU_DimensionAttr")
def _dimAttr(dim, context=None):
    return Attribute.parse(f"#gpu<dim {dim}>", context=context)


def block_id(dim):
    return ArithValue(gpu.BlockIdOp(dim).result)


def block_id_x():
    return block_id("x")


def block_id_y():
    return block_id("y")


def block_attr(dim):
    assert dim in {"x", "y", "z"}
    return Attribute.parse(f"#gpu.block<{dim}>")


def thread_attr(dim):
    assert dim in {"x", "y", "z"}
    return Attribute.parse(f"#gpu.thread<{dim}>")


def gpu_async_token():
    return Type.parse("!gpu.async.token")


def set_container_module(module):
    module.operation.attributes["gpu.container_module"] = UnitAttr.get()
    return module


def host_register(memref):
    return gpu.HostRegisterOp(memref)


def all_reduce(op, val, uniform=False):
    return gpu.AllReduceOp(val, op=op, uniform=uniform).result


# def GPU_AllReduceOpAdd : I32EnumAttrCase<"ADD", 0, "add">;
# def GPU_AllReduceOpAnd : I32EnumAttrCase<"AND", 1, "and">;
# def GPU_AllReduceOpMax : I32EnumAttrCase<"MAX", 2, "max">;
# def GPU_AllReduceOpMin : I32EnumAttrCase<"MIN", 3, "min">;
# def GPU_AllReduceOpMul : I32EnumAttrCase<"MUL", 4, "mul">;
# def GPU_AllReduceOpOr  : I32EnumAttrCase<"OR",  5, "or">;
# def GPU_AllReduceOpXor : I32EnumAttrCase<"XOR", 6, "xor">;


@register_attribute_builder("GPU_AllReduceOperationAttr")
def gpu_all_reduce_op_attr(
    op: str, context: Optional[Context] = None
) -> FlatSymbolRefAttr:
    from . import DefaultContext

    if context is None:
        context = DefaultContext
    return Attribute.parse(f"#gpu<all_reduce_op {op}>", context)
