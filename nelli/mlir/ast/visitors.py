from .._mlir.ir import OpView, Operation
from .._mlir.dialects import (
    _arith_ops_gen as arith,
    # _async_dialect_ops_gen as async_dialect,
    _bufferization_ops_gen as bufferization,
    _builtin_ops_gen as builtin,
    _cf_ops_gen as cf,
    _complex_ops_gen as complex,
    _func_ops_gen as func,
    _gpu_ops_gen as gpu,
    _linalg_ops_gen as linalg,
    _loop_transform_ops_gen as transform_loop,
    _math_ops_gen as math,
    _memref_ops_gen as memref,
    _ml_program_ops_gen as ml_program,
    _pdl_ops_gen as pdl,
    _scf_ops_gen as scf,
    _shape_ops_gen as shape,
    _sparse_tensor_ops_gen as sparse_tensor,
    _structured_transform_ops_gen as transform_structured,
    _tensor_ops_gen as tensor,
    _tosa_ops_gen as tosa,
    _transform_ops_gen as transform,
    _vector_ops_gen as vector,
)
from ..affine import _affine_ops_gen as affine
from ..async_dialect import _async_ops_gen as async_
from ..llvm import _llvm_ops_gen as llvm
from ..omp import _omp_ops_gen as omp


class DialectVisitor:
    def __init__(self, all_dialect_visitor):
        self.all_dialect_visitor = all_dialect_visitor

    def visit(self, op: OpView):
        visitor = getattr(self, f"visit_{op.__class__.__name__}", None)
        visitor(op)


############################
# autogen starts
############################


class AllDialectVisitor:
    def __init__(
        self,
        affine_visitor_ctor=None,
        arith_visitor_ctor=None,
        async_visitor_ctor=None,
        bufferization_visitor_ctor=None,
        builtin_visitor_ctor=None,
        cf_visitor_ctor=None,
        complex_visitor_ctor=None,
        func_visitor_ctor=None,
        gpu_visitor_ctor=None,
        linalg_visitor_ctor=None,
        llvm_visitor_ctor=None,
        math_visitor_ctor=None,
        memref_visitor_ctor=None,
        ml_program_visitor_ctor=None,
        omp_visitor_ctor=None,
        pdl_visitor_ctor=None,
        scf_visitor_ctor=None,
        shape_visitor_ctor=None,
        sparse_tensor_visitor_ctor=None,
        tensor_visitor_ctor=None,
        tosa_visitor_ctor=None,
        transform_visitor_ctor=None,
        transform_loop_visitor_ctor=None,
        transform_structured_visitor_ctor=None,
        vector_visitor_ctor=None,
    ):
        if affine_visitor_ctor is None:
            affine_visitor_ctor = AffineDialectVisitor
        if arith_visitor_ctor is None:
            arith_visitor_ctor = ArithDialectVisitor
        if async_visitor_ctor is None:
            async_visitor_ctor = AsyncDialectVisitor
        if bufferization_visitor_ctor is None:
            bufferization_visitor_ctor = BufferizationDialectVisitor
        if builtin_visitor_ctor is None:
            builtin_visitor_ctor = BuiltinDialectVisitor
        if cf_visitor_ctor is None:
            cf_visitor_ctor = CfDialectVisitor
        if complex_visitor_ctor is None:
            complex_visitor_ctor = ComplexDialectVisitor
        if func_visitor_ctor is None:
            func_visitor_ctor = FuncDialectVisitor
        if gpu_visitor_ctor is None:
            gpu_visitor_ctor = GpuDialectVisitor
        if linalg_visitor_ctor is None:
            linalg_visitor_ctor = LinalgDialectVisitor
        if llvm_visitor_ctor is None:
            llvm_visitor_ctor = LlvmDialectVisitor
        if math_visitor_ctor is None:
            math_visitor_ctor = MathDialectVisitor
        if memref_visitor_ctor is None:
            memref_visitor_ctor = MemrefDialectVisitor
        if ml_program_visitor_ctor is None:
            ml_program_visitor_ctor = MlProgramDialectVisitor
        if omp_visitor_ctor is None:
            omp_visitor_ctor = OmpDialectVisitor
        if pdl_visitor_ctor is None:
            pdl_visitor_ctor = PdlDialectVisitor
        if scf_visitor_ctor is None:
            scf_visitor_ctor = ScfDialectVisitor
        if shape_visitor_ctor is None:
            shape_visitor_ctor = ShapeDialectVisitor
        if sparse_tensor_visitor_ctor is None:
            sparse_tensor_visitor_ctor = SparseTensorDialectVisitor
        if tensor_visitor_ctor is None:
            tensor_visitor_ctor = TensorDialectVisitor
        if tosa_visitor_ctor is None:
            tosa_visitor_ctor = TosaDialectVisitor
        if transform_visitor_ctor is None:
            transform_visitor_ctor = TransformDialectVisitor
        if transform_loop_visitor_ctor is None:
            transform_loop_visitor_ctor = TransformLoopDialectVisitor
        if transform_structured_visitor_ctor is None:
            transform_structured_visitor_ctor = TransformStructuredDialectVisitor
        if vector_visitor_ctor is None:
            vector_visitor_ctor = VectorDialectVisitor
        self.affine_visitor = affine_visitor_ctor(self)
        self.arith_visitor = arith_visitor_ctor(self)
        self.async_visitor = async_visitor_ctor(self)
        self.bufferization_visitor = bufferization_visitor_ctor(self)
        self.builtin_visitor = builtin_visitor_ctor(self)
        self.cf_visitor = cf_visitor_ctor(self)
        self.complex_visitor = complex_visitor_ctor(self)
        self.func_visitor = func_visitor_ctor(self)
        self.gpu_visitor = gpu_visitor_ctor(self)
        self.linalg_visitor = linalg_visitor_ctor(self)
        self.llvm_visitor = llvm_visitor_ctor(self)
        self.math_visitor = math_visitor_ctor(self)
        self.memref_visitor = memref_visitor_ctor(self)
        self.ml_program_visitor = ml_program_visitor_ctor(self)
        self.omp_visitor = omp_visitor_ctor(self)
        self.pdl_visitor = pdl_visitor_ctor(self)
        self.scf_visitor = scf_visitor_ctor(self)
        self.shape_visitor = shape_visitor_ctor(self)
        self.sparse_tensor_visitor = sparse_tensor_visitor_ctor(self)
        self.tensor_visitor = tensor_visitor_ctor(self)
        self.tosa_visitor = tosa_visitor_ctor(self)
        self.transform_visitor = transform_visitor_ctor(self)
        self.transform_loop_visitor = transform_loop_visitor_ctor(self)
        self.transform_structured_visitor = transform_structured_visitor_ctor(self)
        self.vector_visitor = vector_visitor_ctor(self)

    def visit(self, op: Operation):
        for r in op.regions:
            for b in r.blocks:
                for o in b.operations:
                    *dialect, name = o.operation.name.split(".")
                    dialect = "_".join(dialect)
                    dialect_visitor: DialectVisitor = getattr(
                        self, f"{dialect}_visitor"
                    )
                    assert dialect_visitor is not None, f"missing {dialect} visitor"
                    dialect_visitor.visit(o.operation.opview)
                    self.visit(o.operation)


class AffineDialectVisitor(DialectVisitor):
    def visit_AffineApplyOp(self, op: affine.AffineApplyOp):
        pass

    def visit_AffineForOp(self, op: affine.AffineForOp):
        pass

    def visit_AffineIfOp(self, op: affine.AffineIfOp):
        pass

    def visit_AffineLoadOp(self, op: affine.AffineLoadOp):
        pass

    def visit_AffineMaxOp(self, op: affine.AffineMaxOp):
        pass

    def visit_AffineMinOp(self, op: affine.AffineMinOp):
        pass

    def visit_AffineParallelOp(self, op: affine.AffineParallelOp):
        pass

    def visit_AffinePrefetchOp(self, op: affine.AffinePrefetchOp):
        pass

    def visit_AffineStoreOp(self, op: affine.AffineStoreOp):
        pass

    def visit_AffineVectorLoadOp(self, op: affine.AffineVectorLoadOp):
        pass

    def visit_AffineVectorStoreOp(self, op: affine.AffineVectorStoreOp):
        pass

    def visit_AffineYieldOp(self, op: affine.AffineYieldOp):
        pass


class ArithDialectVisitor(DialectVisitor):
    def visit_AddFOp(self, op: arith.AddFOp):
        pass

    def visit_AddIOp(self, op: arith.AddIOp):
        pass

    def visit_AddUIExtendedOp(self, op: arith.AddUIExtendedOp):
        pass

    def visit_AndIOp(self, op: arith.AndIOp):
        pass

    def visit_BitcastOp(self, op: arith.BitcastOp):
        pass

    def visit_CeilDivSIOp(self, op: arith.CeilDivSIOp):
        pass

    def visit_CeilDivUIOp(self, op: arith.CeilDivUIOp):
        pass

    def visit_CmpFOp(self, op: arith.CmpFOp):
        pass

    def visit_CmpIOp(self, op: arith.CmpIOp):
        pass

    def visit_ConstantOp(self, op: arith.ConstantOp):
        pass

    def visit_DivFOp(self, op: arith.DivFOp):
        pass

    def visit_DivSIOp(self, op: arith.DivSIOp):
        pass

    def visit_DivUIOp(self, op: arith.DivUIOp):
        pass

    def visit_ExtFOp(self, op: arith.ExtFOp):
        pass

    def visit_ExtSIOp(self, op: arith.ExtSIOp):
        pass

    def visit_ExtUIOp(self, op: arith.ExtUIOp):
        pass

    def visit_FPToSIOp(self, op: arith.FPToSIOp):
        pass

    def visit_FPToUIOp(self, op: arith.FPToUIOp):
        pass

    def visit_FloorDivSIOp(self, op: arith.FloorDivSIOp):
        pass

    def visit_IndexCastOp(self, op: arith.IndexCastOp):
        pass

    def visit_IndexCastUIOp(self, op: arith.IndexCastUIOp):
        pass

    def visit_MaxFOp(self, op: arith.MaxFOp):
        pass

    def visit_MaxSIOp(self, op: arith.MaxSIOp):
        pass

    def visit_MaxUIOp(self, op: arith.MaxUIOp):
        pass

    def visit_MinFOp(self, op: arith.MinFOp):
        pass

    def visit_MinSIOp(self, op: arith.MinSIOp):
        pass

    def visit_MinUIOp(self, op: arith.MinUIOp):
        pass

    def visit_MulFOp(self, op: arith.MulFOp):
        pass

    def visit_MulIOp(self, op: arith.MulIOp):
        pass

    def visit_MulSIExtendedOp(self, op: arith.MulSIExtendedOp):
        pass

    def visit_MulUIExtendedOp(self, op: arith.MulUIExtendedOp):
        pass

    def visit_NegFOp(self, op: arith.NegFOp):
        pass

    def visit_OrIOp(self, op: arith.OrIOp):
        pass

    def visit_RemFOp(self, op: arith.RemFOp):
        pass

    def visit_RemSIOp(self, op: arith.RemSIOp):
        pass

    def visit_RemUIOp(self, op: arith.RemUIOp):
        pass

    def visit_SIToFPOp(self, op: arith.SIToFPOp):
        pass

    def visit_SelectOp(self, op: arith.SelectOp):
        pass

    def visit_ShLIOp(self, op: arith.ShLIOp):
        pass

    def visit_ShRSIOp(self, op: arith.ShRSIOp):
        pass

    def visit_ShRUIOp(self, op: arith.ShRUIOp):
        pass

    def visit_SubFOp(self, op: arith.SubFOp):
        pass

    def visit_SubIOp(self, op: arith.SubIOp):
        pass

    def visit_TruncFOp(self, op: arith.TruncFOp):
        pass

    def visit_TruncIOp(self, op: arith.TruncIOp):
        pass

    def visit_UIToFPOp(self, op: arith.UIToFPOp):
        pass

    def visit_XOrIOp(self, op: arith.XOrIOp):
        pass


class AsyncDialectVisitor(DialectVisitor):
    def visit_AddToGroupOp(self, op: async_.AddToGroupOp):
        pass

    def visit_AwaitAllOp(self, op: async_.AwaitAllOp):
        pass

    def visit_AwaitOp(self, op: async_.AwaitOp):
        pass

    def visit_CallOp(self, op: async_.CallOp):
        pass

    def visit_CoroBeginOp(self, op: async_.CoroBeginOp):
        pass

    def visit_CoroEndOp(self, op: async_.CoroEndOp):
        pass

    def visit_CoroFreeOp(self, op: async_.CoroFreeOp):
        pass

    def visit_CoroIdOp(self, op: async_.CoroIdOp):
        pass

    def visit_CoroSaveOp(self, op: async_.CoroSaveOp):
        pass

    def visit_CoroSuspendOp(self, op: async_.CoroSuspendOp):
        pass

    def visit_CreateGroupOp(self, op: async_.CreateGroupOp):
        pass

    def visit_ExecuteOp(self, op: async_.ExecuteOp):
        pass

    def visit_FuncOp(self, op: async_.FuncOp):
        pass

    def visit_ReturnOp(self, op: async_.ReturnOp):
        pass

    def visit_RuntimeAddRefOp(self, op: async_.RuntimeAddRefOp):
        pass

    def visit_RuntimeAddToGroupOp(self, op: async_.RuntimeAddToGroupOp):
        pass

    def visit_RuntimeAwaitAndResumeOp(self, op: async_.RuntimeAwaitAndResumeOp):
        pass

    def visit_RuntimeAwaitOp(self, op: async_.RuntimeAwaitOp):
        pass

    def visit_RuntimeCreateGroupOp(self, op: async_.RuntimeCreateGroupOp):
        pass

    def visit_RuntimeCreateOp(self, op: async_.RuntimeCreateOp):
        pass

    def visit_RuntimeDropRefOp(self, op: async_.RuntimeDropRefOp):
        pass

    def visit_RuntimeIsErrorOp(self, op: async_.RuntimeIsErrorOp):
        pass

    def visit_RuntimeLoadOp(self, op: async_.RuntimeLoadOp):
        pass

    def visit_RuntimeNumWorkerThreadsOp(self, op: async_.RuntimeNumWorkerThreadsOp):
        pass

    def visit_RuntimeResumeOp(self, op: async_.RuntimeResumeOp):
        pass

    def visit_RuntimeSetAvailableOp(self, op: async_.RuntimeSetAvailableOp):
        pass

    def visit_RuntimeSetErrorOp(self, op: async_.RuntimeSetErrorOp):
        pass

    def visit_RuntimeStoreOp(self, op: async_.RuntimeStoreOp):
        pass

    def visit_YieldOp(self, op: async_.YieldOp):
        pass


class BufferizationDialectVisitor(DialectVisitor):
    def visit_AllocTensorOp(self, op: bufferization.AllocTensorOp):
        pass

    def visit_CloneOp(self, op: bufferization.CloneOp):
        pass

    def visit_DeallocTensorOp(self, op: bufferization.DeallocTensorOp):
        pass

    def visit_ToMemrefOp(self, op: bufferization.ToMemrefOp):
        pass

    def visit_ToTensorOp(self, op: bufferization.ToTensorOp):
        pass


class BuiltinDialectVisitor(DialectVisitor):
    def visit_ModuleOp(self, op: builtin.ModuleOp):
        pass

    def visit_UnrealizedConversionCastOp(self, op: builtin.UnrealizedConversionCastOp):
        pass


class CfDialectVisitor(DialectVisitor):
    def visit_AssertOp(self, op: cf.AssertOp):
        pass

    def visit_BranchOp(self, op: cf.BranchOp):
        pass

    def visit_CondBranchOp(self, op: cf.CondBranchOp):
        pass

    def visit_SwitchOp(self, op: cf.SwitchOp):
        pass


class ComplexDialectVisitor(DialectVisitor):
    def visit_AbsOp(self, op: complex.AbsOp):
        pass

    def visit_AddOp(self, op: complex.AddOp):
        pass

    def visit_AngleOp(self, op: complex.AngleOp):
        pass

    def visit_Atan2Op(self, op: complex.Atan2Op):
        pass

    def visit_ConjOp(self, op: complex.ConjOp):
        pass

    def visit_ConstantOp(self, op: complex.ConstantOp):
        pass

    def visit_CosOp(self, op: complex.CosOp):
        pass

    def visit_CreateOp(self, op: complex.CreateOp):
        pass

    def visit_DivOp(self, op: complex.DivOp):
        pass

    def visit_EqualOp(self, op: complex.EqualOp):
        pass

    def visit_ExpOp(self, op: complex.ExpOp):
        pass

    def visit_Expm1Op(self, op: complex.Expm1Op):
        pass

    def visit_ImOp(self, op: complex.ImOp):
        pass

    def visit_Log1pOp(self, op: complex.Log1pOp):
        pass

    def visit_LogOp(self, op: complex.LogOp):
        pass

    def visit_MulOp(self, op: complex.MulOp):
        pass

    def visit_NegOp(self, op: complex.NegOp):
        pass

    def visit_NotEqualOp(self, op: complex.NotEqualOp):
        pass

    def visit_PowOp(self, op: complex.PowOp):
        pass

    def visit_ReOp(self, op: complex.ReOp):
        pass

    def visit_RsqrtOp(self, op: complex.RsqrtOp):
        pass

    def visit_SignOp(self, op: complex.SignOp):
        pass

    def visit_SinOp(self, op: complex.SinOp):
        pass

    def visit_SqrtOp(self, op: complex.SqrtOp):
        pass

    def visit_SubOp(self, op: complex.SubOp):
        pass

    def visit_TanOp(self, op: complex.TanOp):
        pass

    def visit_TanhOp(self, op: complex.TanhOp):
        pass


class FuncDialectVisitor(DialectVisitor):
    def visit_CallIndirectOp(self, op: func.CallIndirectOp):
        pass

    def visit_CallOp(self, op: func.CallOp):
        pass

    def visit_ConstantOp(self, op: func.ConstantOp):
        pass

    def visit_FuncOp(self, op: func.FuncOp):
        pass

    def visit_ReturnOp(self, op: func.ReturnOp):
        pass


class GpuDialectVisitor(DialectVisitor):
    def visit_AllReduceOp(self, op: gpu.AllReduceOp):
        pass

    def visit_AllocOp(self, op: gpu.AllocOp):
        pass

    def visit_BarrierOp(self, op: gpu.BarrierOp):
        pass

    def visit_BlockDimOp(self, op: gpu.BlockDimOp):
        pass

    def visit_BlockIdOp(self, op: gpu.BlockIdOp):
        pass

    def visit_DeallocOp(self, op: gpu.DeallocOp):
        pass

    def visit_GPUFuncOp(self, op: gpu.GPUFuncOp):
        pass

    def visit_GPUModuleOp(self, op: gpu.GPUModuleOp):
        pass

    def visit_GlobalIdOp(self, op: gpu.GlobalIdOp):
        pass

    def visit_GridDimOp(self, op: gpu.GridDimOp):
        pass

    def visit_HostRegisterOp(self, op: gpu.HostRegisterOp):
        pass

    def visit_LaneIdOp(self, op: gpu.LaneIdOp):
        pass

    def visit_LaunchFuncOp(self, op: gpu.LaunchFuncOp):
        pass

    def visit_LaunchOp(self, op: gpu.LaunchOp):
        pass

    def visit_MemcpyOp(self, op: gpu.MemcpyOp):
        pass

    def visit_MemsetOp(self, op: gpu.MemsetOp):
        pass

    def visit_ModuleEndOp(self, op: gpu.ModuleEndOp):
        pass

    def visit_NumSubgroupsOp(self, op: gpu.NumSubgroupsOp):
        pass

    def visit_PrintfOp(self, op: gpu.PrintfOp):
        pass

    def visit_ReturnOp(self, op: gpu.ReturnOp):
        pass

    def visit_SetDefaultDeviceOp(self, op: gpu.SetDefaultDeviceOp):
        pass

    def visit_ShuffleOp(self, op: gpu.ShuffleOp):
        pass

    def visit_SubgroupIdOp(self, op: gpu.SubgroupIdOp):
        pass

    def visit_SubgroupMmaComputeOp(self, op: gpu.SubgroupMmaComputeOp):
        pass

    def visit_SubgroupMmaConstantMatrixOp(self, op: gpu.SubgroupMmaConstantMatrixOp):
        pass

    def visit_SubgroupMmaElementwiseOp(self, op: gpu.SubgroupMmaElementwiseOp):
        pass

    def visit_SubgroupMmaLoadMatrixOp(self, op: gpu.SubgroupMmaLoadMatrixOp):
        pass

    def visit_SubgroupMmaStoreMatrixOp(self, op: gpu.SubgroupMmaStoreMatrixOp):
        pass

    def visit_SubgroupReduceOp(self, op: gpu.SubgroupReduceOp):
        pass

    def visit_SubgroupSizeOp(self, op: gpu.SubgroupSizeOp):
        pass

    def visit_TerminatorOp(self, op: gpu.TerminatorOp):
        pass

    def visit_ThreadIdOp(self, op: gpu.ThreadIdOp):
        pass

    def visit_WaitOp(self, op: gpu.WaitOp):
        pass

    def visit_YieldOp(self, op: gpu.YieldOp):
        pass


class LinalgDialectVisitor(DialectVisitor):
    def visit_BatchMatmulOp(self, op: linalg.BatchMatmulOp):
        pass

    def visit_BatchMatmulTransposeBOp(self, op: linalg.BatchMatmulTransposeBOp):
        pass

    def visit_BatchMatvecOp(self, op: linalg.BatchMatvecOp):
        pass

    def visit_BatchReduceMatmulOp(self, op: linalg.BatchReduceMatmulOp):
        pass

    def visit_BroadcastOp(self, op: linalg.BroadcastOp):
        pass

    def visit_Conv1DNcwFcwOp(self, op: linalg.Conv1DNcwFcwOp):
        pass

    def visit_Conv1DNwcWcfOp(self, op: linalg.Conv1DNwcWcfOp):
        pass

    def visit_Conv1DOp(self, op: linalg.Conv1DOp):
        pass

    def visit_Conv2DNchwFchwOp(self, op: linalg.Conv2DNchwFchwOp):
        pass

    def visit_Conv2DNgchwFgchwOp(self, op: linalg.Conv2DNgchwFgchwOp):
        pass

    def visit_Conv2DNhwcFhwcOp(self, op: linalg.Conv2DNhwcFhwcOp):
        pass

    def visit_Conv2DNhwcHwcfOp(self, op: linalg.Conv2DNhwcHwcfOp):
        pass

    def visit_Conv2DNhwcHwcfQOp(self, op: linalg.Conv2DNhwcHwcfQOp):
        pass

    def visit_Conv2DOp(self, op: linalg.Conv2DOp):
        pass

    def visit_Conv3DNdhwcDhwcfOp(self, op: linalg.Conv3DNdhwcDhwcfOp):
        pass

    def visit_Conv3DNdhwcDhwcfQOp(self, op: linalg.Conv3DNdhwcDhwcfQOp):
        pass

    def visit_Conv3DOp(self, op: linalg.Conv3DOp):
        pass

    def visit_CopyOp(self, op: linalg.CopyOp):
        pass

    def visit_DepthwiseConv1DNwcWcOp(self, op: linalg.DepthwiseConv1DNwcWcOp):
        pass

    def visit_DepthwiseConv1DNwcWcmOp(self, op: linalg.DepthwiseConv1DNwcWcmOp):
        pass

    def visit_DepthwiseConv2DNchwChwOp(self, op: linalg.DepthwiseConv2DNchwChwOp):
        pass

    def visit_DepthwiseConv2DNhwcHwcOp(self, op: linalg.DepthwiseConv2DNhwcHwcOp):
        pass

    def visit_DepthwiseConv2DNhwcHwcQOp(self, op: linalg.DepthwiseConv2DNhwcHwcQOp):
        pass

    def visit_DepthwiseConv2DNhwcHwcmOp(self, op: linalg.DepthwiseConv2DNhwcHwcmOp):
        pass

    def visit_DepthwiseConv2DNhwcHwcmQOp(self, op: linalg.DepthwiseConv2DNhwcHwcmQOp):
        pass

    def visit_DepthwiseConv3DNdhwcDhwcOp(self, op: linalg.DepthwiseConv3DNdhwcDhwcOp):
        pass

    def visit_DepthwiseConv3DNdhwcDhwcmOp(self, op: linalg.DepthwiseConv3DNdhwcDhwcmOp):
        pass

    def visit_DotOp(self, op: linalg.DotOp):
        pass

    def visit_ElemwiseBinaryOp(self, op: linalg.ElemwiseBinaryOp):
        pass

    def visit_ElemwiseUnaryOp(self, op: linalg.ElemwiseUnaryOp):
        pass

    def visit_FillOp(self, op: linalg.FillOp):
        pass

    def visit_FillRng2DOp(self, op: linalg.FillRng2DOp):
        pass

    def visit_GenericOp(self, op: linalg.GenericOp):
        pass

    def visit_IndexOp(self, op: linalg.IndexOp):
        pass

    def visit_MapOp(self, op: linalg.MapOp):
        pass

    def visit_MatmulOp(self, op: linalg.MatmulOp):
        pass

    def visit_MatmulTransposeBOp(self, op: linalg.MatmulTransposeBOp):
        pass

    def visit_MatmulUnsignedOp(self, op: linalg.MatmulUnsignedOp):
        pass

    def visit_MatvecOp(self, op: linalg.MatvecOp):
        pass

    def visit_Mmt4DOp(self, op: linalg.Mmt4DOp):
        pass

    def visit_PoolingNchwMaxOp(self, op: linalg.PoolingNchwMaxOp):
        pass

    def visit_PoolingNchwSumOp(self, op: linalg.PoolingNchwSumOp):
        pass

    def visit_PoolingNcwMaxOp(self, op: linalg.PoolingNcwMaxOp):
        pass

    def visit_PoolingNcwSumOp(self, op: linalg.PoolingNcwSumOp):
        pass

    def visit_PoolingNdhwcMaxOp(self, op: linalg.PoolingNdhwcMaxOp):
        pass

    def visit_PoolingNdhwcMinOp(self, op: linalg.PoolingNdhwcMinOp):
        pass

    def visit_PoolingNdhwcSumOp(self, op: linalg.PoolingNdhwcSumOp):
        pass

    def visit_PoolingNhwcMaxOp(self, op: linalg.PoolingNhwcMaxOp):
        pass

    def visit_PoolingNhwcMaxUnsignedOp(self, op: linalg.PoolingNhwcMaxUnsignedOp):
        pass

    def visit_PoolingNhwcMinOp(self, op: linalg.PoolingNhwcMinOp):
        pass

    def visit_PoolingNhwcMinUnsignedOp(self, op: linalg.PoolingNhwcMinUnsignedOp):
        pass

    def visit_PoolingNhwcSumOp(self, op: linalg.PoolingNhwcSumOp):
        pass

    def visit_PoolingNwcMaxOp(self, op: linalg.PoolingNwcMaxOp):
        pass

    def visit_PoolingNwcMaxUnsignedOp(self, op: linalg.PoolingNwcMaxUnsignedOp):
        pass

    def visit_PoolingNwcMinOp(self, op: linalg.PoolingNwcMinOp):
        pass

    def visit_PoolingNwcMinUnsignedOp(self, op: linalg.PoolingNwcMinUnsignedOp):
        pass

    def visit_PoolingNwcSumOp(self, op: linalg.PoolingNwcSumOp):
        pass

    def visit_QuantizedBatchMatmulOp(self, op: linalg.QuantizedBatchMatmulOp):
        pass

    def visit_QuantizedMatmulOp(self, op: linalg.QuantizedMatmulOp):
        pass

    def visit_ReduceOp(self, op: linalg.ReduceOp):
        pass

    def visit_TransposeOp(self, op: linalg.TransposeOp):
        pass

    def visit_VecmatOp(self, op: linalg.VecmatOp):
        pass

    def visit_YieldOp(self, op: linalg.YieldOp):
        pass


class LlvmDialectVisitor(DialectVisitor):
    def visit_AShrOp(self, op: llvm.AShrOp):
        pass

    def visit_AccessGroupMetadataOp(self, op: llvm.AccessGroupMetadataOp):
        pass

    def visit_AddOp(self, op: llvm.AddOp):
        pass

    def visit_AddrSpaceCastOp(self, op: llvm.AddrSpaceCastOp):
        pass

    def visit_AddressOfOp(self, op: llvm.AddressOfOp):
        pass

    def visit_AliasScopeDomainMetadataOp(self, op: llvm.AliasScopeDomainMetadataOp):
        pass

    def visit_AliasScopeMetadataOp(self, op: llvm.AliasScopeMetadataOp):
        pass

    def visit_AllocaOp(self, op: llvm.AllocaOp):
        pass

    def visit_AndOp(self, op: llvm.AndOp):
        pass

    def visit_AtomicCmpXchgOp(self, op: llvm.AtomicCmpXchgOp):
        pass

    def visit_AtomicRMWOp(self, op: llvm.AtomicRMWOp):
        pass

    def visit_BitcastOp(self, op: llvm.BitcastOp):
        pass

    def visit_BrOp(self, op: llvm.BrOp):
        pass

    def visit_CallOp(self, op: llvm.CallOp):
        pass

    def visit_CondBrOp(self, op: llvm.CondBrOp):
        pass

    def visit_ConstantOp(self, op: llvm.ConstantOp):
        pass

    def visit_ExtractElementOp(self, op: llvm.ExtractElementOp):
        pass

    def visit_ExtractValueOp(self, op: llvm.ExtractValueOp):
        pass

    def visit_FAddOp(self, op: llvm.FAddOp):
        pass

    def visit_FCmpOp(self, op: llvm.FCmpOp):
        pass

    def visit_FDivOp(self, op: llvm.FDivOp):
        pass

    def visit_FMulOp(self, op: llvm.FMulOp):
        pass

    def visit_FNegOp(self, op: llvm.FNegOp):
        pass

    def visit_FPExtOp(self, op: llvm.FPExtOp):
        pass

    def visit_FPToSIOp(self, op: llvm.FPToSIOp):
        pass

    def visit_FPToUIOp(self, op: llvm.FPToUIOp):
        pass

    def visit_FPTruncOp(self, op: llvm.FPTruncOp):
        pass

    def visit_FRemOp(self, op: llvm.FRemOp):
        pass

    def visit_FSubOp(self, op: llvm.FSubOp):
        pass

    def visit_FenceOp(self, op: llvm.FenceOp):
        pass

    def visit_FreezeOp(self, op: llvm.FreezeOp):
        pass

    def visit_GEPOp(self, op: llvm.GEPOp):
        pass

    def visit_GlobalCtorsOp(self, op: llvm.GlobalCtorsOp):
        pass

    def visit_GlobalDtorsOp(self, op: llvm.GlobalDtorsOp):
        pass

    def visit_GlobalOp(self, op: llvm.GlobalOp):
        pass

    def visit_ICmpOp(self, op: llvm.ICmpOp):
        pass

    def visit_InlineAsmOp(self, op: llvm.InlineAsmOp):
        pass

    def visit_InsertElementOp(self, op: llvm.InsertElementOp):
        pass

    def visit_InsertValueOp(self, op: llvm.InsertValueOp):
        pass

    def visit_IntToPtrOp(self, op: llvm.IntToPtrOp):
        pass

    def visit_InvokeOp(self, op: llvm.InvokeOp):
        pass

    def visit_LLVMFuncOp(self, op: llvm.LLVMFuncOp):
        pass

    def visit_LShrOp(self, op: llvm.LShrOp):
        pass

    def visit_LandingpadOp(self, op: llvm.LandingpadOp):
        pass

    def visit_LoadOp(self, op: llvm.LoadOp):
        pass

    def visit_MetadataOp(self, op: llvm.MetadataOp):
        pass

    def visit_MulOp(self, op: llvm.MulOp):
        pass

    def visit_NullOp(self, op: llvm.NullOp):
        pass

    def visit_OrOp(self, op: llvm.OrOp):
        pass

    def visit_PtrToIntOp(self, op: llvm.PtrToIntOp):
        pass

    def visit_ResumeOp(self, op: llvm.ResumeOp):
        pass

    def visit_ReturnOp(self, op: llvm.ReturnOp):
        pass

    def visit_SDivOp(self, op: llvm.SDivOp):
        pass

    def visit_SExtOp(self, op: llvm.SExtOp):
        pass

    def visit_SIToFPOp(self, op: llvm.SIToFPOp):
        pass

    def visit_SRemOp(self, op: llvm.SRemOp):
        pass

    def visit_SelectOp(self, op: llvm.SelectOp):
        pass

    def visit_ShlOp(self, op: llvm.ShlOp):
        pass

    def visit_ShuffleVectorOp(self, op: llvm.ShuffleVectorOp):
        pass

    def visit_StoreOp(self, op: llvm.StoreOp):
        pass

    def visit_SubOp(self, op: llvm.SubOp):
        pass

    def visit_SwitchOp(self, op: llvm.SwitchOp):
        pass

    def visit_TBAARootMetadataOp(self, op: llvm.TBAARootMetadataOp):
        pass

    def visit_TBAATagOp(self, op: llvm.TBAATagOp):
        pass

    def visit_TBAATypeDescriptorOp(self, op: llvm.TBAATypeDescriptorOp):
        pass

    def visit_TruncOp(self, op: llvm.TruncOp):
        pass

    def visit_UDivOp(self, op: llvm.UDivOp):
        pass

    def visit_UIToFPOp(self, op: llvm.UIToFPOp):
        pass

    def visit_URemOp(self, op: llvm.URemOp):
        pass

    def visit_UndefOp(self, op: llvm.UndefOp):
        pass

    def visit_UnreachableOp(self, op: llvm.UnreachableOp):
        pass

    def visit_XOrOp(self, op: llvm.XOrOp):
        pass

    def visit_ZExtOp(self, op: llvm.ZExtOp):
        pass


class MathDialectVisitor(DialectVisitor):
    def visit_AbsFOp(self, op: math.AbsFOp):
        pass

    def visit_AbsIOp(self, op: math.AbsIOp):
        pass

    def visit_Atan2Op(self, op: math.Atan2Op):
        pass

    def visit_AtanOp(self, op: math.AtanOp):
        pass

    def visit_CbrtOp(self, op: math.CbrtOp):
        pass

    def visit_CeilOp(self, op: math.CeilOp):
        pass

    def visit_CopySignOp(self, op: math.CopySignOp):
        pass

    def visit_CosOp(self, op: math.CosOp):
        pass

    def visit_CountLeadingZerosOp(self, op: math.CountLeadingZerosOp):
        pass

    def visit_CountTrailingZerosOp(self, op: math.CountTrailingZerosOp):
        pass

    def visit_CtPopOp(self, op: math.CtPopOp):
        pass

    def visit_ErfOp(self, op: math.ErfOp):
        pass

    def visit_Exp2Op(self, op: math.Exp2Op):
        pass

    def visit_ExpM1Op(self, op: math.ExpM1Op):
        pass

    def visit_ExpOp(self, op: math.ExpOp):
        pass

    def visit_FPowIOp(self, op: math.FPowIOp):
        pass

    def visit_FloorOp(self, op: math.FloorOp):
        pass

    def visit_FmaOp(self, op: math.FmaOp):
        pass

    def visit_IPowIOp(self, op: math.IPowIOp):
        pass

    def visit_Log10Op(self, op: math.Log10Op):
        pass

    def visit_Log1pOp(self, op: math.Log1pOp):
        pass

    def visit_Log2Op(self, op: math.Log2Op):
        pass

    def visit_LogOp(self, op: math.LogOp):
        pass

    def visit_PowFOp(self, op: math.PowFOp):
        pass

    def visit_RoundEvenOp(self, op: math.RoundEvenOp):
        pass

    def visit_RoundOp(self, op: math.RoundOp):
        pass

    def visit_RsqrtOp(self, op: math.RsqrtOp):
        pass

    def visit_SinOp(self, op: math.SinOp):
        pass

    def visit_SqrtOp(self, op: math.SqrtOp):
        pass

    def visit_TanOp(self, op: math.TanOp):
        pass

    def visit_TanhOp(self, op: math.TanhOp):
        pass

    def visit_TruncOp(self, op: math.TruncOp):
        pass


class MemrefDialectVisitor(DialectVisitor):
    def visit_AllocOp(self, op: memref.AllocOp):
        pass

    def visit_AllocaOp(self, op: memref.AllocaOp):
        pass

    def visit_AllocaScopeOp(self, op: memref.AllocaScopeOp):
        pass

    def visit_AllocaScopeReturnOp(self, op: memref.AllocaScopeReturnOp):
        pass

    def visit_AssumeAlignmentOp(self, op: memref.AssumeAlignmentOp):
        pass

    def visit_AtomicRMWOp(self, op: memref.AtomicRMWOp):
        pass

    def visit_AtomicYieldOp(self, op: memref.AtomicYieldOp):
        pass

    def visit_CastOp(self, op: memref.CastOp):
        pass

    def visit_CollapseShapeOp(self, op: memref.CollapseShapeOp):
        pass

    def visit_CopyOp(self, op: memref.CopyOp):
        pass

    def visit_DeallocOp(self, op: memref.DeallocOp):
        pass

    def visit_DimOp(self, op: memref.DimOp):
        pass

    def visit_DmaStartOp(self, op: memref.DmaStartOp):
        pass

    def visit_DmaWaitOp(self, op: memref.DmaWaitOp):
        pass

    def visit_ExpandShapeOp(self, op: memref.ExpandShapeOp):
        pass

    def visit_ExtractAlignedPointerAsIndexOp(
        self, op: memref.ExtractAlignedPointerAsIndexOp
    ):
        pass

    def visit_ExtractStridedMetadataOp(self, op: memref.ExtractStridedMetadataOp):
        pass

    def visit_GenericAtomicRMWOp(self, op: memref.GenericAtomicRMWOp):
        pass

    def visit_GetGlobalOp(self, op: memref.GetGlobalOp):
        pass

    def visit_GlobalOp(self, op: memref.GlobalOp):
        pass

    def visit_LoadOp(self, op: memref.LoadOp):
        pass

    def visit_MemorySpaceCastOp(self, op: memref.MemorySpaceCastOp):
        pass

    def visit_PrefetchOp(self, op: memref.PrefetchOp):
        pass

    def visit_RankOp(self, op: memref.RankOp):
        pass

    def visit_ReallocOp(self, op: memref.ReallocOp):
        pass

    def visit_ReinterpretCastOp(self, op: memref.ReinterpretCastOp):
        pass

    def visit_ReshapeOp(self, op: memref.ReshapeOp):
        pass

    def visit_StoreOp(self, op: memref.StoreOp):
        pass

    def visit_SubViewOp(self, op: memref.SubViewOp):
        pass

    def visit_TensorStoreOp(self, op: memref.TensorStoreOp):
        pass

    def visit_TransposeOp(self, op: memref.TransposeOp):
        pass

    def visit_ViewOp(self, op: memref.ViewOp):
        pass


class MlProgramDialectVisitor(DialectVisitor):
    def visit_FuncOp(self, op: ml_program.FuncOp):
        pass

    def visit_GlobalLoadConstOp(self, op: ml_program.GlobalLoadConstOp):
        pass

    def visit_GlobalLoadGraphOp(self, op: ml_program.GlobalLoadGraphOp):
        pass

    def visit_GlobalLoadOp(self, op: ml_program.GlobalLoadOp):
        pass

    def visit_GlobalOp(self, op: ml_program.GlobalOp):
        pass

    def visit_GlobalStoreGraphOp(self, op: ml_program.GlobalStoreGraphOp):
        pass

    def visit_GlobalStoreOp(self, op: ml_program.GlobalStoreOp):
        pass

    def visit_OutputOp(self, op: ml_program.OutputOp):
        pass

    def visit_ReturnOp(self, op: ml_program.ReturnOp):
        pass

    def visit_SubgraphOp(self, op: ml_program.SubgraphOp):
        pass

    def visit_TokenOp(self, op: ml_program.TokenOp):
        pass


class OmpDialectVisitor(DialectVisitor):
    def visit_AtomicCaptureOp(self, op: omp.AtomicCaptureOp):
        pass

    def visit_AtomicReadOp(self, op: omp.AtomicReadOp):
        pass

    def visit_AtomicUpdateOp(self, op: omp.AtomicUpdateOp):
        pass

    def visit_AtomicWriteOp(self, op: omp.AtomicWriteOp):
        pass

    def visit_BarrierOp(self, op: omp.BarrierOp):
        pass

    def visit_CancelOp(self, op: omp.CancelOp):
        pass

    def visit_CancellationPointOp(self, op: omp.CancellationPointOp):
        pass

    def visit_CriticalDeclareOp(self, op: omp.CriticalDeclareOp):
        pass

    def visit_CriticalOp(self, op: omp.CriticalOp):
        pass

    def visit_DataOp(self, op: omp.DataOp):
        pass

    def visit_EnterDataOp(self, op: omp.EnterDataOp):
        pass

    def visit_ExitDataOp(self, op: omp.ExitDataOp):
        pass

    def visit_FlushOp(self, op: omp.FlushOp):
        pass

    def visit_MasterOp(self, op: omp.MasterOp):
        pass

    def visit_OrderedOp(self, op: omp.OrderedOp):
        pass

    def visit_OrderedRegionOp(self, op: omp.OrderedRegionOp):
        pass

    def visit_ParallelOp(self, op: omp.ParallelOp):
        pass

    def visit_ReductionDeclareOp(self, op: omp.ReductionDeclareOp):
        pass

    def visit_ReductionOp(self, op: omp.ReductionOp):
        pass

    def visit_SectionOp(self, op: omp.SectionOp):
        pass

    def visit_SectionsOp(self, op: omp.SectionsOp):
        pass

    def visit_SimdLoopOp(self, op: omp.SimdLoopOp):
        pass

    def visit_SingleOp(self, op: omp.SingleOp):
        pass

    def visit_TargetOp(self, op: omp.TargetOp):
        pass

    def visit_TaskGroupOp(self, op: omp.TaskGroupOp):
        pass

    def visit_TaskLoopOp(self, op: omp.TaskLoopOp):
        pass

    def visit_TaskOp(self, op: omp.TaskOp):
        pass

    def visit_TaskwaitOp(self, op: omp.TaskwaitOp):
        pass

    def visit_TaskyieldOp(self, op: omp.TaskyieldOp):
        pass

    def visit_TerminatorOp(self, op: omp.TerminatorOp):
        pass

    def visit_ThreadprivateOp(self, op: omp.ThreadprivateOp):
        pass

    def visit_WsLoopOp(self, op: omp.WsLoopOp):
        pass

    def visit_YieldOp(self, op: omp.YieldOp):
        pass


class PdlDialectVisitor(DialectVisitor):
    def visit_ApplyNativeConstraintOp(self, op: pdl.ApplyNativeConstraintOp):
        pass

    def visit_ApplyNativeRewriteOp(self, op: pdl.ApplyNativeRewriteOp):
        pass

    def visit_AttributeOp(self, op: pdl.AttributeOp):
        pass

    def visit_EraseOp(self, op: pdl.EraseOp):
        pass

    def visit_OperandOp(self, op: pdl.OperandOp):
        pass

    def visit_OperandsOp(self, op: pdl.OperandsOp):
        pass

    def visit_OperationOp(self, op: pdl.OperationOp):
        pass

    def visit_PatternOp(self, op: pdl.PatternOp):
        pass

    def visit_RangeOp(self, op: pdl.RangeOp):
        pass

    def visit_ReplaceOp(self, op: pdl.ReplaceOp):
        pass

    def visit_ResultOp(self, op: pdl.ResultOp):
        pass

    def visit_ResultsOp(self, op: pdl.ResultsOp):
        pass

    def visit_RewriteOp(self, op: pdl.RewriteOp):
        pass

    def visit_TypeOp(self, op: pdl.TypeOp):
        pass

    def visit_TypesOp(self, op: pdl.TypesOp):
        pass


class ScfDialectVisitor(DialectVisitor):
    def visit_ConditionOp(self, op: scf.ConditionOp):
        pass

    def visit_ExecuteRegionOp(self, op: scf.ExecuteRegionOp):
        pass

    def visit_ForOp(self, op: scf.ForOp):
        pass

    def visit_ForallOp(self, op: scf.ForallOp):
        pass

    def visit_IfOp(self, op: scf.IfOp):
        pass

    def visit_InParallelOp(self, op: scf.InParallelOp):
        pass

    def visit_IndexSwitchOp(self, op: scf.IndexSwitchOp):
        pass

    def visit_ParallelOp(self, op: scf.ParallelOp):
        pass

    def visit_ReduceOp(self, op: scf.ReduceOp):
        pass

    def visit_ReduceReturnOp(self, op: scf.ReduceReturnOp):
        pass

    def visit_WhileOp(self, op: scf.WhileOp):
        pass

    def visit_YieldOp(self, op: scf.YieldOp):
        pass


class ShapeDialectVisitor(DialectVisitor):
    def visit_AddOp(self, op: shape.AddOp):
        pass

    def visit_AnyOp(self, op: shape.AnyOp):
        pass

    def visit_AssumingAllOp(self, op: shape.AssumingAllOp):
        pass

    def visit_AssumingOp(self, op: shape.AssumingOp):
        pass

    def visit_AssumingYieldOp(self, op: shape.AssumingYieldOp):
        pass

    def visit_BroadcastOp(self, op: shape.BroadcastOp):
        pass

    def visit_ConcatOp(self, op: shape.ConcatOp):
        pass

    def visit_ConstShapeOp(self, op: shape.ConstShapeOp):
        pass

    def visit_ConstSizeOp(self, op: shape.ConstSizeOp):
        pass

    def visit_ConstWitnessOp(self, op: shape.ConstWitnessOp):
        pass

    def visit_CstrBroadcastableOp(self, op: shape.CstrBroadcastableOp):
        pass

    def visit_CstrEqOp(self, op: shape.CstrEqOp):
        pass

    def visit_CstrRequireOp(self, op: shape.CstrRequireOp):
        pass

    def visit_DebugPrintOp(self, op: shape.DebugPrintOp):
        pass

    def visit_DimOp(self, op: shape.DimOp):
        pass

    def visit_DivOp(self, op: shape.DivOp):
        pass

    def visit_FromExtentTensorOp(self, op: shape.FromExtentTensorOp):
        pass

    def visit_FromExtentsOp(self, op: shape.FromExtentsOp):
        pass

    def visit_FuncOp(self, op: shape.FuncOp):
        pass

    def visit_FunctionLibraryOp(self, op: shape.FunctionLibraryOp):
        pass

    def visit_GetExtentOp(self, op: shape.GetExtentOp):
        pass

    def visit_IndexToSizeOp(self, op: shape.IndexToSizeOp):
        pass

    def visit_IsBroadcastableOp(self, op: shape.IsBroadcastableOp):
        pass

    def visit_MaxOp(self, op: shape.MaxOp):
        pass

    def visit_MeetOp(self, op: shape.MeetOp):
        pass

    def visit_MinOp(self, op: shape.MinOp):
        pass

    def visit_MulOp(self, op: shape.MulOp):
        pass

    def visit_NumElementsOp(self, op: shape.NumElementsOp):
        pass

    def visit_RankOp(self, op: shape.RankOp):
        pass

    def visit_ReduceOp(self, op: shape.ReduceOp):
        pass

    def visit_ReturnOp(self, op: shape.ReturnOp):
        pass

    def visit_ShapeEqOp(self, op: shape.ShapeEqOp):
        pass

    def visit_ShapeOfOp(self, op: shape.ShapeOfOp):
        pass

    def visit_SizeToIndexOp(self, op: shape.SizeToIndexOp):
        pass

    def visit_SplitAtOp(self, op: shape.SplitAtOp):
        pass

    def visit_ToExtentTensorOp(self, op: shape.ToExtentTensorOp):
        pass

    def visit_ValueAsShapeOp(self, op: shape.ValueAsShapeOp):
        pass

    def visit_ValueOfOp(self, op: shape.ValueOfOp):
        pass

    def visit_WithOp(self, op: shape.WithOp):
        pass

    def visit_YieldOp(self, op: shape.YieldOp):
        pass


class SparseTensorDialectVisitor(DialectVisitor):
    def visit_BinaryOp(self, op: sparse_tensor.BinaryOp):
        pass

    def visit_CompressOp(self, op: sparse_tensor.CompressOp):
        pass

    def visit_ConcatenateOp(self, op: sparse_tensor.ConcatenateOp):
        pass

    def visit_ConvertOp(self, op: sparse_tensor.ConvertOp):
        pass

    def visit_ExpandOp(self, op: sparse_tensor.ExpandOp):
        pass

    def visit_ForeachOp(self, op: sparse_tensor.ForeachOp):
        pass

    def visit_GetStorageSpecifierOp(self, op: sparse_tensor.GetStorageSpecifierOp):
        pass

    def visit_InsertOp(self, op: sparse_tensor.InsertOp):
        pass

    def visit_LoadOp(self, op: sparse_tensor.LoadOp):
        pass

    def visit_NewOp(self, op: sparse_tensor.NewOp):
        pass

    def visit_NumberOfEntriesOp(self, op: sparse_tensor.NumberOfEntriesOp):
        pass

    def visit_OutOp(self, op: sparse_tensor.OutOp):
        pass

    def visit_PackOp(self, op: sparse_tensor.PackOp):
        pass

    def visit_PushBackOp(self, op: sparse_tensor.PushBackOp):
        pass

    def visit_ReduceOp(self, op: sparse_tensor.ReduceOp):
        pass

    def visit_SelectOp(self, op: sparse_tensor.SelectOp):
        pass

    def visit_SetStorageSpecifierOp(self, op: sparse_tensor.SetStorageSpecifierOp):
        pass

    def visit_SortCooOp(self, op: sparse_tensor.SortCooOp):
        pass

    def visit_SortOp(self, op: sparse_tensor.SortOp):
        pass

    def visit_StorageSpecifierInitOp(self, op: sparse_tensor.StorageSpecifierInitOp):
        pass

    def visit_ToCoordinatesBufferOp(self, op: sparse_tensor.ToCoordinatesBufferOp):
        pass

    def visit_ToCoordinatesOp(self, op: sparse_tensor.ToCoordinatesOp):
        pass

    def visit_ToPositionsOp(self, op: sparse_tensor.ToPositionsOp):
        pass

    def visit_ToSliceOffsetOp(self, op: sparse_tensor.ToSliceOffsetOp):
        pass

    def visit_ToSliceStrideOp(self, op: sparse_tensor.ToSliceStrideOp):
        pass

    def visit_ToValuesOp(self, op: sparse_tensor.ToValuesOp):
        pass

    def visit_UnaryOp(self, op: sparse_tensor.UnaryOp):
        pass

    def visit_UnpackOp(self, op: sparse_tensor.UnpackOp):
        pass

    def visit_YieldOp(self, op: sparse_tensor.YieldOp):
        pass


class TensorDialectVisitor(DialectVisitor):
    def visit_CastOp(self, op: tensor.CastOp):
        pass

    def visit_CollapseShapeOp(self, op: tensor.CollapseShapeOp):
        pass

    def visit_DimOp(self, op: tensor.DimOp):
        pass

    def visit_EmptyOp(self, op: tensor.EmptyOp):
        pass

    def visit_ExpandShapeOp(self, op: tensor.ExpandShapeOp):
        pass

    def visit_ExtractOp(self, op: tensor.ExtractOp):
        pass

    def visit_ExtractSliceOp(self, op: tensor.ExtractSliceOp):
        pass

    def visit_FromElementsOp(self, op: tensor.FromElementsOp):
        pass

    def visit_GatherOp(self, op: tensor.GatherOp):
        pass

    def visit_GenerateOp(self, op: tensor.GenerateOp):
        pass

    def visit_InsertOp(self, op: tensor.InsertOp):
        pass

    def visit_InsertSliceOp(self, op: tensor.InsertSliceOp):
        pass

    def visit_PackOp(self, op: tensor.PackOp):
        pass

    def visit_PadOp(self, op: tensor.PadOp):
        pass

    def visit_ParallelInsertSliceOp(self, op: tensor.ParallelInsertSliceOp):
        pass

    def visit_RankOp(self, op: tensor.RankOp):
        pass

    def visit_ReshapeOp(self, op: tensor.ReshapeOp):
        pass

    def visit_ScatterOp(self, op: tensor.ScatterOp):
        pass

    def visit_SplatOp(self, op: tensor.SplatOp):
        pass

    def visit_UnPackOp(self, op: tensor.UnPackOp):
        pass

    def visit_YieldOp(self, op: tensor.YieldOp):
        pass


class TosaDialectVisitor(DialectVisitor):
    def visit_AbsOp(self, op: tosa.AbsOp):
        pass

    def visit_AddOp(self, op: tosa.AddOp):
        pass

    def visit_ApplyScaleOp(self, op: tosa.ApplyScaleOp):
        pass

    def visit_ArgMaxOp(self, op: tosa.ArgMaxOp):
        pass

    def visit_ArithmeticRightShiftOp(self, op: tosa.ArithmeticRightShiftOp):
        pass

    def visit_AvgPool2dOp(self, op: tosa.AvgPool2dOp):
        pass

    def visit_BitwiseAndOp(self, op: tosa.BitwiseAndOp):
        pass

    def visit_BitwiseNotOp(self, op: tosa.BitwiseNotOp):
        pass

    def visit_BitwiseOrOp(self, op: tosa.BitwiseOrOp):
        pass

    def visit_BitwiseXorOp(self, op: tosa.BitwiseXorOp):
        pass

    def visit_CastOp(self, op: tosa.CastOp):
        pass

    def visit_CeilOp(self, op: tosa.CeilOp):
        pass

    def visit_ClampOp(self, op: tosa.ClampOp):
        pass

    def visit_ClzOp(self, op: tosa.ClzOp):
        pass

    def visit_ConcatOp(self, op: tosa.ConcatOp):
        pass

    def visit_ConstOp(self, op: tosa.ConstOp):
        pass

    def visit_Conv2DOp(self, op: tosa.Conv2DOp):
        pass

    def visit_Conv3DOp(self, op: tosa.Conv3DOp):
        pass

    def visit_CustomOp(self, op: tosa.CustomOp):
        pass

    def visit_DepthwiseConv2DOp(self, op: tosa.DepthwiseConv2DOp):
        pass

    def visit_DivOp(self, op: tosa.DivOp):
        pass

    def visit_EqualOp(self, op: tosa.EqualOp):
        pass

    def visit_ExpOp(self, op: tosa.ExpOp):
        pass

    def visit_FFT2dOp(self, op: tosa.FFT2dOp):
        pass

    def visit_FloorOp(self, op: tosa.FloorOp):
        pass

    def visit_FullyConnectedOp(self, op: tosa.FullyConnectedOp):
        pass

    def visit_GatherOp(self, op: tosa.GatherOp):
        pass

    def visit_GreaterEqualOp(self, op: tosa.GreaterEqualOp):
        pass

    def visit_GreaterOp(self, op: tosa.GreaterOp):
        pass

    def visit_IdentityOp(self, op: tosa.IdentityOp):
        pass

    def visit_IfOp(self, op: tosa.IfOp):
        pass

    def visit_LogOp(self, op: tosa.LogOp):
        pass

    def visit_LogicalAndOp(self, op: tosa.LogicalAndOp):
        pass

    def visit_LogicalLeftShiftOp(self, op: tosa.LogicalLeftShiftOp):
        pass

    def visit_LogicalNotOp(self, op: tosa.LogicalNotOp):
        pass

    def visit_LogicalOrOp(self, op: tosa.LogicalOrOp):
        pass

    def visit_LogicalRightShiftOp(self, op: tosa.LogicalRightShiftOp):
        pass

    def visit_LogicalXorOp(self, op: tosa.LogicalXorOp):
        pass

    def visit_MatMulOp(self, op: tosa.MatMulOp):
        pass

    def visit_MaxPool2dOp(self, op: tosa.MaxPool2dOp):
        pass

    def visit_MaximumOp(self, op: tosa.MaximumOp):
        pass

    def visit_MinimumOp(self, op: tosa.MinimumOp):
        pass

    def visit_MulOp(self, op: tosa.MulOp):
        pass

    def visit_NegateOp(self, op: tosa.NegateOp):
        pass

    def visit_PadOp(self, op: tosa.PadOp):
        pass

    def visit_PowOp(self, op: tosa.PowOp):
        pass

    def visit_RFFT2dOp(self, op: tosa.RFFT2dOp):
        pass

    def visit_ReciprocalOp(self, op: tosa.ReciprocalOp):
        pass

    def visit_ReduceAllOp(self, op: tosa.ReduceAllOp):
        pass

    def visit_ReduceAnyOp(self, op: tosa.ReduceAnyOp):
        pass

    def visit_ReduceMaxOp(self, op: tosa.ReduceMaxOp):
        pass

    def visit_ReduceMinOp(self, op: tosa.ReduceMinOp):
        pass

    def visit_ReduceProdOp(self, op: tosa.ReduceProdOp):
        pass

    def visit_ReduceSumOp(self, op: tosa.ReduceSumOp):
        pass

    def visit_RescaleOp(self, op: tosa.RescaleOp):
        pass

    def visit_ReshapeOp(self, op: tosa.ReshapeOp):
        pass

    def visit_ResizeOp(self, op: tosa.ResizeOp):
        pass

    def visit_ReverseOp(self, op: tosa.ReverseOp):
        pass

    def visit_RsqrtOp(self, op: tosa.RsqrtOp):
        pass

    def visit_ScatterOp(self, op: tosa.ScatterOp):
        pass

    def visit_SelectOp(self, op: tosa.SelectOp):
        pass

    def visit_SigmoidOp(self, op: tosa.SigmoidOp):
        pass

    def visit_SliceOp(self, op: tosa.SliceOp):
        pass

    def visit_SubOp(self, op: tosa.SubOp):
        pass

    def visit_TableOp(self, op: tosa.TableOp):
        pass

    def visit_TanhOp(self, op: tosa.TanhOp):
        pass

    def visit_TileOp(self, op: tosa.TileOp):
        pass

    def visit_TransposeConv2DOp(self, op: tosa.TransposeConv2DOp):
        pass

    def visit_TransposeOp(self, op: tosa.TransposeOp):
        pass

    def visit_WhileOp(self, op: tosa.WhileOp):
        pass

    def visit_YieldOp(self, op: tosa.YieldOp):
        pass


class TransformDialectVisitor(DialectVisitor):
    def visit_AlternativesOp(self, op: transform.AlternativesOp):
        pass

    def visit_CastOp(self, op: transform.CastOp):
        pass

    def visit_ForeachOp(self, op: transform.ForeachOp):
        pass

    def visit_GetConsumersOfResult(self, op: transform.GetConsumersOfResult):
        pass

    def visit_GetDefiningOp(self, op: transform.GetDefiningOp):
        pass

    def visit_GetProducerOfOperand(self, op: transform.GetProducerOfOperand):
        pass

    def visit_GetResultOp(self, op: transform.GetResultOp):
        pass

    def visit_MergeHandlesOp(self, op: transform.MergeHandlesOp):
        pass

    def visit_PrintOp(self, op: transform.PrintOp):
        pass

    def visit_ReplicateOp(self, op: transform.ReplicateOp):
        pass

    def visit_SequenceOp(self, op: transform.SequenceOp):
        pass

    def visit_YieldOp(self, op: transform.YieldOp):
        pass


class TransformLoopDialectVisitor(DialectVisitor):
    def visit_GetParentForOp(self, op: transform_loop.GetParentForOp):
        pass

    def visit_LoopCoalesceOp(self, op: transform_loop.LoopCoalesceOp):
        pass

    def visit_LoopOutlineOp(self, op: transform_loop.LoopOutlineOp):
        pass

    def visit_LoopPeelOp(self, op: transform_loop.LoopPeelOp):
        pass

    def visit_LoopPipelineOp(self, op: transform_loop.LoopPipelineOp):
        pass

    def visit_LoopUnrollOp(self, op: transform_loop.LoopUnrollOp):
        pass


class TransformStructuredDialectVisitor(DialectVisitor):
    def visit_BufferizeToAllocationOp(
        self, op: transform_structured.BufferizeToAllocationOp
    ):
        pass

    def visit_ConvertConv2DToImg2ColOp(
        self, op: transform_structured.ConvertConv2DToImg2ColOp
    ):
        pass

    def visit_DecomposeOp(self, op: transform_structured.DecomposeOp):
        pass

    def visit_FuseIntoContainingOp(self, op: transform_structured.FuseIntoContainingOp):
        pass

    def visit_FuseOp(self, op: transform_structured.FuseOp):
        pass

    def visit_GeneralizeOp(self, op: transform_structured.GeneralizeOp):
        pass

    def visit_HoistPadOp(self, op: transform_structured.HoistPadOp):
        pass

    def visit_HoistRedundantTensorSubsetsOp(
        self, op: transform_structured.HoistRedundantTensorSubsetsOp
    ):
        pass

    def visit_HoistRedundantVectorTransfersOp(
        self, op: transform_structured.HoistRedundantVectorTransfersOp
    ):
        pass

    def visit_InterchangeOp(self, op: transform_structured.InterchangeOp):
        pass

    def visit_LowerPackOp(self, op: transform_structured.LowerPackOp):
        pass

    def visit_LowerUnPackOp(self, op: transform_structured.LowerUnPackOp):
        pass

    def visit_MaskedVectorizeOp(self, op: transform_structured.MaskedVectorizeOp):
        pass

    def visit_MatchOp(self, op: transform_structured.MatchOp):
        pass

    def visit_MultiTileSizesOp(self, op: transform_structured.MultiTileSizesOp):
        pass

    def visit_PackGreedilyOp(self, op: transform_structured.PackGreedilyOp):
        pass

    def visit_PackOp(self, op: transform_structured.PackOp):
        pass

    def visit_PackTransposeOp(self, op: transform_structured.PackTransposeOp):
        pass

    def visit_PadOp(self, op: transform_structured.PadOp):
        pass

    def visit_PromoteOp(self, op: transform_structured.PromoteOp):
        pass

    def visit_ReplaceOp(self, op: transform_structured.ReplaceOp):
        pass

    def visit_RewriteInDestinationPassingStyleOp(
        self, op: transform_structured.RewriteInDestinationPassingStyleOp
    ):
        pass

    def visit_ScalarizeOp(self, op: transform_structured.ScalarizeOp):
        pass

    def visit_SplitOp(self, op: transform_structured.SplitOp):
        pass

    def visit_SplitReductionOp(self, op: transform_structured.SplitReductionOp):
        pass

    def visit_TileOp(self, op: transform_structured.TileOp):
        pass

    def visit_TileReductionUsingForallOp(
        self, op: transform_structured.TileReductionUsingForallOp
    ):
        pass

    def visit_TileReductionUsingScfOp(
        self, op: transform_structured.TileReductionUsingScfOp
    ):
        pass

    def visit_TileToForallOp(self, op: transform_structured.TileToForallOp):
        pass

    def visit_TileToScfForOp(self, op: transform_structured.TileToScfForOp):
        pass

    def visit_VectorizeOp(self, op: transform_structured.VectorizeOp):
        pass


class VectorDialectVisitor(DialectVisitor):
    def visit_BitCastOp(self, op: vector.BitCastOp):
        pass

    def visit_BroadcastOp(self, op: vector.BroadcastOp):
        pass

    def visit_CompressStoreOp(self, op: vector.CompressStoreOp):
        pass

    def visit_ConstantMaskOp(self, op: vector.ConstantMaskOp):
        pass

    def visit_ContractionOp(self, op: vector.ContractionOp):
        pass

    def visit_CreateMaskOp(self, op: vector.CreateMaskOp):
        pass

    def visit_ExpandLoadOp(self, op: vector.ExpandLoadOp):
        pass

    def visit_ExtractElementOp(self, op: vector.ExtractElementOp):
        pass

    def visit_ExtractOp(self, op: vector.ExtractOp):
        pass

    def visit_ExtractStridedSliceOp(self, op: vector.ExtractStridedSliceOp):
        pass

    def visit_FMAOp(self, op: vector.FMAOp):
        pass

    def visit_FlatTransposeOp(self, op: vector.FlatTransposeOp):
        pass

    def visit_GatherOp(self, op: vector.GatherOp):
        pass

    def visit_InsertElementOp(self, op: vector.InsertElementOp):
        pass

    def visit_InsertOp(self, op: vector.InsertOp):
        pass

    def visit_InsertStridedSliceOp(self, op: vector.InsertStridedSliceOp):
        pass

    def visit_LoadOp(self, op: vector.LoadOp):
        pass

    def visit_MaskOp(self, op: vector.MaskOp):
        pass

    def visit_MaskedLoadOp(self, op: vector.MaskedLoadOp):
        pass

    def visit_MaskedStoreOp(self, op: vector.MaskedStoreOp):
        pass

    def visit_MatmulOp(self, op: vector.MatmulOp):
        pass

    def visit_MultiDimReductionOp(self, op: vector.MultiDimReductionOp):
        pass

    def visit_OuterProductOp(self, op: vector.OuterProductOp):
        pass

    def visit_PrintOp(self, op: vector.PrintOp):
        pass

    def visit_ReductionOp(self, op: vector.ReductionOp):
        pass

    def visit_ReshapeOp(self, op: vector.ReshapeOp):
        pass

    def visit_ScalableExtractOp(self, op: vector.ScalableExtractOp):
        pass

    def visit_ScalableInsertOp(self, op: vector.ScalableInsertOp):
        pass

    def visit_ScanOp(self, op: vector.ScanOp):
        pass

    def visit_ScatterOp(self, op: vector.ScatterOp):
        pass

    def visit_ShapeCastOp(self, op: vector.ShapeCastOp):
        pass

    def visit_ShuffleOp(self, op: vector.ShuffleOp):
        pass

    def visit_SplatOp(self, op: vector.SplatOp):
        pass

    def visit_StoreOp(self, op: vector.StoreOp):
        pass

    def visit_TransferReadOp(self, op: vector.TransferReadOp):
        pass

    def visit_TransferWriteOp(self, op: vector.TransferWriteOp):
        pass

    def visit_TransposeOp(self, op: vector.TransposeOp):
        pass

    def visit_TypeCastOp(self, op: vector.TypeCastOp):
        pass

    def visit_VectorScaleOp(self, op: vector.VectorScaleOp):
        pass

    def visit_WarpExecuteOnLane0Op(self, op: vector.WarpExecuteOnLane0Op):
        pass

    def visit_YieldOp(self, op: vector.YieldOp):
        pass
