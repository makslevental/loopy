from .._mlir.dialects._ods_common import _cext as cext
from .._mlir.dialects._ods_common import (
    extend_opview_class,
    get_op_result_or_value,
    get_op_results_or_values,
    segmented_accessor,
    get_default_loc_context,
)

_ods_ir = cext.ir

_ods_ext_module = None

import builtins


from .._mlir.dialects._transform_ops_gen import _Dialect


@cext.register_operation(_Dialect)
@extend_opview_class(_ods_ext_module)
class ApplyBufferOptimizationsOp(_ods_ir.OpView):
    OPERATION_NAME = "transform.apply_buffer_optimizations"

    _ODS_REGIONS = (0, True)

    @builtins.property
    def target(self):
        return self.operation.operands[0]


# @cext.register_operation(_Dialect)
@extend_opview_class(_ods_ext_module)
class ApplyPatternsOp(_ods_ir.OpView):
    OPERATION_NAME = "transform.apply_patterns"

    _ODS_REGIONS = (0, True)

    def __init__(
        self,
        target,
        *,
        additional_patterns=None,
        bubble_collapse=None,
        bubble_expand=None,
        bubble_pack_un_pack=None,
        canonicalization=None,
        cse=None,
        erase_unnecessary_tensor_operands=None,
        expand_memref_strided_metadata=None,
        fold_memref_aliases=None,
        fold_reassociative_reshapes=None,
        fold_tensor_empty_extract=None,
        licm=None,
        linalg_elementwise_greedy_fusion=None,
        lower_transfer_op_permutations=None,
        lower_vector_masks=None,
        rank_reducing_linalg=None,
        rank_reducing_linalg_via_reshapes=None,
        rank_reducing_vector=None,
        swap_padding_elide_conditional=None,
        swapping_patterns=None,
        tiling_canonicalization=None,
        unroll_vectors_gpu_mma_sync=None,
        unroll_vectors_gpu_wmma=None,
        loc=None,
        ip=None,
    ):
        operands = []
        results = []
        attributes = {}
        regions = None
        operands.append(get_op_result_or_value(target))
        _ods_context = get_default_loc_context(loc)
        if bool(additional_patterns):
            attributes["additional_patterns"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(bubble_collapse):
            attributes["bubble_collapse"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(bubble_expand):
            attributes["bubble_expand"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(bubble_pack_un_pack):
            attributes["bubble_pack_un_pack"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(canonicalization):
            attributes["canonicalization"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(cse):
            attributes["cse"] = _ods_ir.UnitAttr.get(get_default_loc_context(loc))
        if bool(erase_unnecessary_tensor_operands):
            attributes["erase_unnecessary_tensor_operands"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(expand_memref_strided_metadata):
            attributes["expand_memref_strided_metadata"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(fold_memref_aliases):
            attributes["fold_memref_aliases"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(fold_reassociative_reshapes):
            attributes["fold_reassociative_reshapes"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(fold_tensor_empty_extract):
            attributes["fold_tensor_empty_extract"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(licm):
            attributes["licm"] = _ods_ir.UnitAttr.get(get_default_loc_context(loc))
        if bool(linalg_elementwise_greedy_fusion):
            attributes["linalg_elementwise_greedy_fusion"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(lower_transfer_op_permutations):
            attributes["lower_transfer_op_permutations"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(lower_vector_masks):
            attributes["lower_vector_masks"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(rank_reducing_linalg):
            attributes["rank_reducing_linalg"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(rank_reducing_linalg_via_reshapes):
            attributes["rank_reducing_linalg_via_reshapes"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(rank_reducing_vector):
            attributes["rank_reducing_vector"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(swap_padding_elide_conditional):
            attributes["swap_padding_elide_conditional"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(swapping_patterns):
            attributes["swapping_patterns"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(tiling_canonicalization):
            attributes["tiling_canonicalization"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(unroll_vectors_gpu_mma_sync):
            attributes["unroll_vectors_gpu_mma_sync"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(unroll_vectors_gpu_wmma):
            attributes["unroll_vectors_gpu_wmma"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
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
    def target(self):
        return self.operation.operands[0]

    @builtins.property
    def additional_patterns(self):
        return "additional_patterns" in self.operation.attributes

    @additional_patterns.setter
    def additional_patterns(self, value):
        if bool(value):
            self.operation.attributes["additional_patterns"] = _ods_ir.UnitAttr.get()
        elif "additional_patterns" in self.operation.attributes:
            del self.operation.attributes["additional_patterns"]

    @additional_patterns.deleter
    def additional_patterns(self):
        del self.operation.attributes["additional_patterns"]

    @builtins.property
    def bubble_collapse(self):
        return "bubble_collapse" in self.operation.attributes

    @bubble_collapse.setter
    def bubble_collapse(self, value):
        if bool(value):
            self.operation.attributes["bubble_collapse"] = _ods_ir.UnitAttr.get()
        elif "bubble_collapse" in self.operation.attributes:
            del self.operation.attributes["bubble_collapse"]

    @bubble_collapse.deleter
    def bubble_collapse(self):
        del self.operation.attributes["bubble_collapse"]

    @builtins.property
    def bubble_expand(self):
        return "bubble_expand" in self.operation.attributes

    @bubble_expand.setter
    def bubble_expand(self, value):
        if bool(value):
            self.operation.attributes["bubble_expand"] = _ods_ir.UnitAttr.get()
        elif "bubble_expand" in self.operation.attributes:
            del self.operation.attributes["bubble_expand"]

    @bubble_expand.deleter
    def bubble_expand(self):
        del self.operation.attributes["bubble_expand"]

    @builtins.property
    def bubble_pack_un_pack(self):
        return "bubble_pack_un_pack" in self.operation.attributes

    @bubble_pack_un_pack.setter
    def bubble_pack_un_pack(self, value):
        if bool(value):
            self.operation.attributes["bubble_pack_un_pack"] = _ods_ir.UnitAttr.get()
        elif "bubble_pack_un_pack" in self.operation.attributes:
            del self.operation.attributes["bubble_pack_un_pack"]

    @bubble_pack_un_pack.deleter
    def bubble_pack_un_pack(self):
        del self.operation.attributes["bubble_pack_un_pack"]

    @builtins.property
    def canonicalization(self):
        return "canonicalization" in self.operation.attributes

    @canonicalization.setter
    def canonicalization(self, value):
        if bool(value):
            self.operation.attributes["canonicalization"] = _ods_ir.UnitAttr.get()
        elif "canonicalization" in self.operation.attributes:
            del self.operation.attributes["canonicalization"]

    @canonicalization.deleter
    def canonicalization(self):
        del self.operation.attributes["canonicalization"]

    @builtins.property
    def cse(self):
        return "cse" in self.operation.attributes

    @cse.setter
    def cse(self, value):
        if bool(value):
            self.operation.attributes["cse"] = _ods_ir.UnitAttr.get()
        elif "cse" in self.operation.attributes:
            del self.operation.attributes["cse"]

    @cse.deleter
    def cse(self):
        del self.operation.attributes["cse"]

    @builtins.property
    def erase_unnecessary_tensor_operands(self):
        return "erase_unnecessary_tensor_operands" in self.operation.attributes

    @erase_unnecessary_tensor_operands.setter
    def erase_unnecessary_tensor_operands(self, value):
        if bool(value):
            self.operation.attributes[
                "erase_unnecessary_tensor_operands"
            ] = _ods_ir.UnitAttr.get()
        elif "erase_unnecessary_tensor_operands" in self.operation.attributes:
            del self.operation.attributes["erase_unnecessary_tensor_operands"]

    @erase_unnecessary_tensor_operands.deleter
    def erase_unnecessary_tensor_operands(self):
        del self.operation.attributes["erase_unnecessary_tensor_operands"]

    @builtins.property
    def expand_memref_strided_metadata(self):
        return "expand_memref_strided_metadata" in self.operation.attributes

    @expand_memref_strided_metadata.setter
    def expand_memref_strided_metadata(self, value):
        if bool(value):
            self.operation.attributes[
                "expand_memref_strided_metadata"
            ] = _ods_ir.UnitAttr.get()
        elif "expand_memref_strided_metadata" in self.operation.attributes:
            del self.operation.attributes["expand_memref_strided_metadata"]

    @expand_memref_strided_metadata.deleter
    def expand_memref_strided_metadata(self):
        del self.operation.attributes["expand_memref_strided_metadata"]

    @builtins.property
    def fold_memref_aliases(self):
        return "fold_memref_aliases" in self.operation.attributes

    @fold_memref_aliases.setter
    def fold_memref_aliases(self, value):
        if bool(value):
            self.operation.attributes["fold_memref_aliases"] = _ods_ir.UnitAttr.get()
        elif "fold_memref_aliases" in self.operation.attributes:
            del self.operation.attributes["fold_memref_aliases"]

    @fold_memref_aliases.deleter
    def fold_memref_aliases(self):
        del self.operation.attributes["fold_memref_aliases"]

    @builtins.property
    def fold_reassociative_reshapes(self):
        return "fold_reassociative_reshapes" in self.operation.attributes

    @fold_reassociative_reshapes.setter
    def fold_reassociative_reshapes(self, value):
        if bool(value):
            self.operation.attributes[
                "fold_reassociative_reshapes"
            ] = _ods_ir.UnitAttr.get()
        elif "fold_reassociative_reshapes" in self.operation.attributes:
            del self.operation.attributes["fold_reassociative_reshapes"]

    @fold_reassociative_reshapes.deleter
    def fold_reassociative_reshapes(self):
        del self.operation.attributes["fold_reassociative_reshapes"]

    @builtins.property
    def fold_tensor_empty_extract(self):
        return "fold_tensor_empty_extract" in self.operation.attributes

    @fold_tensor_empty_extract.setter
    def fold_tensor_empty_extract(self, value):
        if bool(value):
            self.operation.attributes[
                "fold_tensor_empty_extract"
            ] = _ods_ir.UnitAttr.get()
        elif "fold_tensor_empty_extract" in self.operation.attributes:
            del self.operation.attributes["fold_tensor_empty_extract"]

    @fold_tensor_empty_extract.deleter
    def fold_tensor_empty_extract(self):
        del self.operation.attributes["fold_tensor_empty_extract"]

    @builtins.property
    def licm(self):
        return "licm" in self.operation.attributes

    @licm.setter
    def licm(self, value):
        if bool(value):
            self.operation.attributes["licm"] = _ods_ir.UnitAttr.get()
        elif "licm" in self.operation.attributes:
            del self.operation.attributes["licm"]

    @licm.deleter
    def licm(self):
        del self.operation.attributes["licm"]

    @builtins.property
    def linalg_elementwise_greedy_fusion(self):
        return "linalg_elementwise_greedy_fusion" in self.operation.attributes

    @linalg_elementwise_greedy_fusion.setter
    def linalg_elementwise_greedy_fusion(self, value):
        if bool(value):
            self.operation.attributes[
                "linalg_elementwise_greedy_fusion"
            ] = _ods_ir.UnitAttr.get()
        elif "linalg_elementwise_greedy_fusion" in self.operation.attributes:
            del self.operation.attributes["linalg_elementwise_greedy_fusion"]

    @linalg_elementwise_greedy_fusion.deleter
    def linalg_elementwise_greedy_fusion(self):
        del self.operation.attributes["linalg_elementwise_greedy_fusion"]

    @builtins.property
    def lower_transfer_op_permutations(self):
        return "lower_transfer_op_permutations" in self.operation.attributes

    @lower_transfer_op_permutations.setter
    def lower_transfer_op_permutations(self, value):
        if bool(value):
            self.operation.attributes[
                "lower_transfer_op_permutations"
            ] = _ods_ir.UnitAttr.get()
        elif "lower_transfer_op_permutations" in self.operation.attributes:
            del self.operation.attributes["lower_transfer_op_permutations"]

    @lower_transfer_op_permutations.deleter
    def lower_transfer_op_permutations(self):
        del self.operation.attributes["lower_transfer_op_permutations"]

    @builtins.property
    def lower_vector_masks(self):
        return "lower_vector_masks" in self.operation.attributes

    @lower_vector_masks.setter
    def lower_vector_masks(self, value):
        if bool(value):
            self.operation.attributes["lower_vector_masks"] = _ods_ir.UnitAttr.get()
        elif "lower_vector_masks" in self.operation.attributes:
            del self.operation.attributes["lower_vector_masks"]

    @lower_vector_masks.deleter
    def lower_vector_masks(self):
        del self.operation.attributes["lower_vector_masks"]

    @builtins.property
    def rank_reducing_linalg(self):
        return "rank_reducing_linalg" in self.operation.attributes

    @rank_reducing_linalg.setter
    def rank_reducing_linalg(self, value):
        if bool(value):
            self.operation.attributes["rank_reducing_linalg"] = _ods_ir.UnitAttr.get()
        elif "rank_reducing_linalg" in self.operation.attributes:
            del self.operation.attributes["rank_reducing_linalg"]

    @rank_reducing_linalg.deleter
    def rank_reducing_linalg(self):
        del self.operation.attributes["rank_reducing_linalg"]

    @builtins.property
    def rank_reducing_linalg_via_reshapes(self):
        return "rank_reducing_linalg_via_reshapes" in self.operation.attributes

    @rank_reducing_linalg_via_reshapes.setter
    def rank_reducing_linalg_via_reshapes(self, value):
        if bool(value):
            self.operation.attributes[
                "rank_reducing_linalg_via_reshapes"
            ] = _ods_ir.UnitAttr.get()
        elif "rank_reducing_linalg_via_reshapes" in self.operation.attributes:
            del self.operation.attributes["rank_reducing_linalg_via_reshapes"]

    @rank_reducing_linalg_via_reshapes.deleter
    def rank_reducing_linalg_via_reshapes(self):
        del self.operation.attributes["rank_reducing_linalg_via_reshapes"]

    @builtins.property
    def rank_reducing_vector(self):
        return "rank_reducing_vector" in self.operation.attributes

    @rank_reducing_vector.setter
    def rank_reducing_vector(self, value):
        if bool(value):
            self.operation.attributes["rank_reducing_vector"] = _ods_ir.UnitAttr.get()
        elif "rank_reducing_vector" in self.operation.attributes:
            del self.operation.attributes["rank_reducing_vector"]

    @rank_reducing_vector.deleter
    def rank_reducing_vector(self):
        del self.operation.attributes["rank_reducing_vector"]

    @builtins.property
    def swap_padding_elide_conditional(self):
        return "swap_padding_elide_conditional" in self.operation.attributes

    @swap_padding_elide_conditional.setter
    def swap_padding_elide_conditional(self, value):
        if bool(value):
            self.operation.attributes[
                "swap_padding_elide_conditional"
            ] = _ods_ir.UnitAttr.get()
        elif "swap_padding_elide_conditional" in self.operation.attributes:
            del self.operation.attributes["swap_padding_elide_conditional"]

    @swap_padding_elide_conditional.deleter
    def swap_padding_elide_conditional(self):
        del self.operation.attributes["swap_padding_elide_conditional"]

    @builtins.property
    def swapping_patterns(self):
        return "swapping_patterns" in self.operation.attributes

    @swapping_patterns.setter
    def swapping_patterns(self, value):
        if bool(value):
            self.operation.attributes["swapping_patterns"] = _ods_ir.UnitAttr.get()
        elif "swapping_patterns" in self.operation.attributes:
            del self.operation.attributes["swapping_patterns"]

    @swapping_patterns.deleter
    def swapping_patterns(self):
        del self.operation.attributes["swapping_patterns"]

    @builtins.property
    def tiling_canonicalization(self):
        return "tiling_canonicalization" in self.operation.attributes

    @tiling_canonicalization.setter
    def tiling_canonicalization(self, value):
        if bool(value):
            self.operation.attributes[
                "tiling_canonicalization"
            ] = _ods_ir.UnitAttr.get()
        elif "tiling_canonicalization" in self.operation.attributes:
            del self.operation.attributes["tiling_canonicalization"]

    @tiling_canonicalization.deleter
    def tiling_canonicalization(self):
        del self.operation.attributes["tiling_canonicalization"]

    @builtins.property
    def unroll_vectors_gpu_mma_sync(self):
        return "unroll_vectors_gpu_mma_sync" in self.operation.attributes

    @unroll_vectors_gpu_mma_sync.setter
    def unroll_vectors_gpu_mma_sync(self, value):
        if bool(value):
            self.operation.attributes[
                "unroll_vectors_gpu_mma_sync"
            ] = _ods_ir.UnitAttr.get()
        elif "unroll_vectors_gpu_mma_sync" in self.operation.attributes:
            del self.operation.attributes["unroll_vectors_gpu_mma_sync"]

    @unroll_vectors_gpu_mma_sync.deleter
    def unroll_vectors_gpu_mma_sync(self):
        del self.operation.attributes["unroll_vectors_gpu_mma_sync"]

    @builtins.property
    def unroll_vectors_gpu_wmma(self):
        return "unroll_vectors_gpu_wmma" in self.operation.attributes

    @unroll_vectors_gpu_wmma.setter
    def unroll_vectors_gpu_wmma(self, value):
        if bool(value):
            self.operation.attributes[
                "unroll_vectors_gpu_wmma"
            ] = _ods_ir.UnitAttr.get()
        elif "unroll_vectors_gpu_wmma" in self.operation.attributes:
            del self.operation.attributes["unroll_vectors_gpu_wmma"]

    @unroll_vectors_gpu_wmma.deleter
    def unroll_vectors_gpu_wmma(self):
        del self.operation.attributes["unroll_vectors_gpu_wmma"]


@cext.register_operation(_Dialect)
@extend_opview_class(_ods_ext_module)
class HoistStaticAllocOp(_ods_ir.OpView):
    OPERATION_NAME = "transform.hoist_static_alloc"

    _ODS_REGIONS = (0, True)

    def __init__(self, target, *, loc=None, ip=None):
        operands = []
        results = []
        attributes = {}
        regions = None
        operands.append(get_op_result_or_value(target))
        _ods_context = get_default_loc_context(loc)
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
    def target(self):
        return self.operation.operands[0]


@cext.register_operation(_Dialect)
@extend_opview_class(_ods_ext_module)
class ShareForallOperandsOp(_ods_ir.OpView):
    OPERATION_NAME = "transform.share_forall_operands"

    _ODS_REGIONS = (0, True)

    def __init__(self, result, forall_op, *, share_operands=None, loc=None, ip=None):
        operands = []
        results = []
        attributes = {}
        regions = None
        operands.append(get_op_result_or_value(forall_op))
        _ods_context = get_default_loc_context(loc)
        if share_operands is not None:
            attributes["share_operands"] = (
                share_operands
                if (
                    issubclass(type(share_operands), _ods_ir.Attribute)
                    or not _ods_ir.AttrBuilder.contains("DenseI64ArrayAttr")
                )
                else _ods_ir.AttrBuilder.get("DenseI64ArrayAttr")(
                    share_operands, context=_ods_context
                )
            )
        results.append(result)
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
    def forall_op(self):
        return self.operation.operands[0]

    @builtins.property
    def result(self):
        return self.operation.results[0]


@cext.register_operation(_Dialect)
@extend_opview_class(_ods_ext_module)
class TransformBufferizeOp(_ods_ir.OpView):
    OPERATION_NAME = "transform.bufferize"

    _ODS_REGIONS = (0, True)

    def __init__(
        self,
        result,
        target,
        *,
        target_gpu=None,
        test_analysis_only=None,
        print_conflicts=None,
        loc=None,
        ip=None,
    ):
        operands = []
        results = []
        attributes = {}
        regions = None
        operands.append(get_op_result_or_value(target))
        _ods_context = get_default_loc_context(loc)
        if bool(target_gpu):
            attributes["target_gpu"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(test_analysis_only):
            attributes["test_analysis_only"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        if bool(print_conflicts):
            attributes["print_conflicts"] = _ods_ir.UnitAttr.get(
                get_default_loc_context(loc)
            )
        results.append(result)
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
    def target(self):
        return self.operation.operands[0]

    @builtins.property
    def target_gpu(self):
        return "target_gpu" in self.operation.attributes

    @target_gpu.setter
    def target_gpu(self, value):
        if bool(value):
            self.operation.attributes["target_gpu"] = _ods_ir.UnitAttr.get()
        elif "target_gpu" in self.operation.attributes:
            del self.operation.attributes["target_gpu"]

    @target_gpu.deleter
    def target_gpu(self):
        del self.operation.attributes["target_gpu"]

    @builtins.property
    def test_analysis_only(self):
        return "test_analysis_only" in self.operation.attributes

    @test_analysis_only.setter
    def test_analysis_only(self, value):
        if bool(value):
            self.operation.attributes["test_analysis_only"] = _ods_ir.UnitAttr.get()
        elif "test_analysis_only" in self.operation.attributes:
            del self.operation.attributes["test_analysis_only"]

    @test_analysis_only.deleter
    def test_analysis_only(self):
        del self.operation.attributes["test_analysis_only"]

    @builtins.property
    def print_conflicts(self):
        return "print_conflicts" in self.operation.attributes

    @print_conflicts.setter
    def print_conflicts(self, value):
        if bool(value):
            self.operation.attributes["print_conflicts"] = _ods_ir.UnitAttr.get()
        elif "print_conflicts" in self.operation.attributes:
            del self.operation.attributes["print_conflicts"]

    @print_conflicts.deleter
    def print_conflicts(self):
        del self.operation.attributes["print_conflicts"]

    @builtins.property
    def result(self):
        return self.operation.results[0]


@cext.register_operation(_Dialect)
@extend_opview_class(_ods_ext_module)
class TransformEliminateEmptyTensorsOp(_ods_ir.OpView):
    OPERATION_NAME = "transform.eliminate_empty_tensors"

    _ODS_REGIONS = (0, True)

    def __init__(self, target, *, loc=None, ip=None):
        operands = []
        results = []
        attributes = {}
        regions = None
        operands.append(get_op_result_or_value(target))
        _ods_context = get_default_loc_context(loc)
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
    def target(self):
        return self.operation.operands[0]
