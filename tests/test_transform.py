from textwrap import dedent

import numpy as np
import pytest

from nelli.mlir import arith
from nelli.mlir._mlir.dialects import linalg
from nelli.mlir._mlir.dialects import transform as transform_dialect
from nelli.mlir._mlir.dialects.transform import loop
from nelli.mlir._mlir.ir import InsertionPoint
from nelli.mlir._mlir.runtime import unranked_memref_to_numpy
from nelli.mlir.func import mlir_func
from nelli.mlir.gpu import block_attr
from nelli.mlir.passes import Pipeline
from nelli.mlir.refbackend import (
    LLVMJITBackend,
    elemental_type_to_ctype,
    memref_type_to_np_dtype,
)
from nelli.mlir.scf import scf_range, forall
from nelli.mlir.tensor import TensorValue as Tensor, pad
from nelli.mlir.tensor import (
    parallel_insert_slice,
    extract_slice,
)
from nelli.mlir.transform import (
    sequence,
    match,
    get_parent_for_loop,
    unroll,
    tile_linalg_to_scf_for,
    tile_to_scf_forall,
    tile_to_scf_for,
)
from nelli.mlir.transform.transform import (
    apply_patterns,
    cast,
    share_forall_operands,
    generalize,
    vectorize,
    hoist_redundant_tensor_subsets,
    bufferize,
    lower_vectors,
)

from nelli.mlir.utils import run_pipeline, F32, I64
from nelli.utils import mlir_mod_ctx
from util import check_correct


class TestTiling:
    backend = LLVMJITBackend(shared_libs=[])

    def test_basic_schedule(self):
        with mlir_mod_ctx() as module:
            sequence = transform_dialect.SequenceOp(
                transform_dialect.FailurePropagationMode.PROPAGATE,
                [],
                transform_dialect.OperationType.get("scf.for"),
            )
            with InsertionPoint(sequence.body):
                loop.LoopUnrollOp(sequence.bodyTarget, factor=42)
                transform_dialect.YieldOp()

        correct = dedent(
            """\
        module {
          transform.sequence  failures(propagate) {
          ^bb0(%arg0: !transform.op<"scf.for">):
            transform.loop.unroll %arg0 {factor = 42 : i64} : !transform.op<"scf.for">
          }
        }
        """
        )
        check_correct(correct, module)

    def test_basic_sugar(self):
        with mlir_mod_ctx() as module:

            @sequence(target="scf.for")
            def basic(target, *extra_args):
                loop.LoopUnrollOp(target, factor=42)

        correct = dedent(
            """\
        module {
          transform.sequence  failures(propagate) attributes {transform.target_tag = "basic"} {
          ^bb0(%arg0: !transform.op<"scf.for">):
            transform.loop.unroll %arg0 {factor = 42 : i64} : !transform.op<"scf.for">
          }
        }
        """
        )
        check_correct(correct, module)

    def test_basic_unroll(self):
        with mlir_mod_ctx() as module:

            @mlir_func(range_ctor=scf_range)
            def loop_unroll_op():
                for i in range(0, 42, 5):
                    i + i

            @sequence
            def basic(target, *extra_args):
                m = match(target, ["arith.addi"])
                loop = get_parent_for_loop(m)
                unroll(loop, 4)

        correct = dedent(
            """\
        module {
          func.func @loop_unroll_op() {
            %c0 = arith.constant 0 : index
            %c42 = arith.constant 42 : index
            %c5 = arith.constant 5 : index
            scf.for %arg0 = %c0 to %c42 step %c5 {
              %0 = arith.addi %arg0, %arg0 : index
            }
            return
          }
          transform.sequence  failures(propagate) attributes {transform.target_tag = "basic"} {
          ^bb0(%arg0: !pdl.operation):
            %0 = transform.structured.match ops{["arith.addi"]} in %arg0 : (!pdl.operation) -> !pdl.operation
            %1 = transform.loop.get_parent_for %0 : (!pdl.operation) -> !transform.op<"scf.for">
            transform.loop.unroll %1 {factor = 4 : i64} : !transform.op<"scf.for">
          }
        }
        """
        )
        check_correct(correct, module)

        run_pipeline(
            module,
            Pipeline()
            .transform_dialect_interpreter()
            .transform_dialect_erase_schedule()
            .materialize(),
        )
        correct = dedent(
            """\
        module {
          func.func @loop_unroll_op() {
            %c0 = arith.constant 0 : index
            %c42 = arith.constant 42 : index
            %c5 = arith.constant 5 : index
            %c40 = arith.constant 40 : index
            %c20 = arith.constant 20 : index
            scf.for %arg0 = %c0 to %c40 step %c20 {
              %1 = arith.addi %arg0, %arg0 : index
              %c1 = arith.constant 1 : index
              %2 = arith.muli %c5, %c1 : index
              %3 = arith.addi %arg0, %2 : index
              %4 = arith.addi %3, %3 : index
              %c2 = arith.constant 2 : index
              %5 = arith.muli %c5, %c2 : index
              %6 = arith.addi %arg0, %5 : index
              %7 = arith.addi %6, %6 : index
              %c3 = arith.constant 3 : index
              %8 = arith.muli %c5, %c3 : index
              %9 = arith.addi %arg0, %8 : index
              %10 = arith.addi %9, %9 : index
            }
            %0 = arith.addi %c40, %c40 : index
            return
          }
        }
        """
        )
        check_correct(correct, module)

    def test_basic_tile(self):
        with mlir_mod_ctx() as module:

            @mlir_func
            def pad_tensor_3_4(input_tensor: Tensor[(4, 16), F32], pad_value: F32):
                return pad(input_tensor, [3, 4], [5, 3], pad_value)

            @sequence
            def basic(target, *extra_args):
                m = match(target, ["tensor.pad"])
                tile_to_scf_for(m, tile_sizes=[2, 3])

        correct = dedent(
            """\
        module {
          func.func @pad_tensor_3_4(%arg0: tensor<4x16xf32>, %arg1: f32) -> tensor<12x23xf32> {
            %padded = tensor.pad %arg0 low[3, 4] high[5, 3] {
            ^bb0(%arg2: index, %arg3: index):
              tensor.yield %arg1 : f32
            } : tensor<4x16xf32> to tensor<12x23xf32>
            return %padded : tensor<12x23xf32>
          }
          transform.sequence  failures(propagate) attributes {transform.target_tag = "basic"} {
          ^bb0(%arg0: !pdl.operation):
            %0 = transform.structured.match ops{["tensor.pad"]} in %arg0 : (!pdl.operation) -> !pdl.operation
            %tiled_linalg_op, %loops:2 = transform.structured.tile_to_scf_for %0[2, 3] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
          }
        }
        """
        )
        check_correct(correct, module)
        run_pipeline(
            module,
            Pipeline()
            .transform_dialect_interpreter()
            .transform_dialect_erase_schedule()
            .canonicalize()
            .materialize(),
        )
        correct = dedent(
            """\
        #map = affine_map<(d0) -> (-d0 + 23, 3)>
        #map1 = affine_map<(d0) -> (-d0 + 3, 0)>
        #map2 = affine_map<(d0) -> (0, d0 - 3)>
        #map3 = affine_map<(d0) -> (4, d0)>
        #map4 = affine_map<(d0) -> (0, d0 - 1)>
        #map5 = affine_map<(d0, d1) -> (d0 - d1)>
        #map6 = affine_map<(d0, d1, d2) -> (-d0 - d1 + d2 + 2)>
        #map7 = affine_map<(d0) -> (-d0 + 4, 0)>
        #map8 = affine_map<(d0) -> (0, d0 - 4)>
        #map9 = affine_map<(d0) -> (16, d0)>
        #map10 = affine_map<(d0, d1) -> (0, d0 + d1 - 4)>
        #map11 = affine_map<(d0, d1, d2, d3) -> (-d0 + d1 - d2 + d3)>
        module {
          func.func @pad_tensor_3_4(%arg0: tensor<4x16xf32>, %arg1: f32) -> tensor<12x23xf32> {
            %c23 = arith.constant 23 : index
            %c12 = arith.constant 12 : index
            %c0 = arith.constant 0 : index
            %c3 = arith.constant 3 : index
            %c2 = arith.constant 2 : index
            %0 = tensor.empty() : tensor<12x23xf32>
            %1 = scf.for %arg2 = %c0 to %c12 step %c2 iter_args(%arg3 = %0) -> (tensor<12x23xf32>) {
              %2 = scf.for %arg4 = %c0 to %c23 step %c3 iter_args(%arg5 = %arg3) -> (tensor<12x23xf32>) {
                %3 = affine.min #map(%arg4)
                %4 = affine.max #map1(%arg2)
                %5 = affine.max #map2(%arg2)
                %6 = affine.min #map3(%5)
                %7 = affine.max #map4(%arg2)
                %8 = affine.min #map3(%7)
                %9 = affine.apply #map5(%8, %6)
                %10 = arith.cmpi eq, %9, %c0 : index
                %11 = affine.apply #map6(%4, %8, %6)
                %12 = affine.max #map7(%arg4)
                %13 = affine.max #map8(%arg4)
                %14 = affine.min #map9(%13)
                %15 = affine.max #map10(%3, %arg4)
                %16 = affine.min #map9(%15)
                %17 = affine.apply #map5(%16, %14)
                %18 = arith.cmpi eq, %17, %c0 : index
                %19 = arith.ori %18, %10 : i1
                %20 = affine.apply #map11(%12, %3, %16, %14)
                %21 = scf.if %19 -> (tensor<?x?xf32>) {
                  %generated = tensor.generate %3 {
                  ^bb0(%arg6: index, %arg7: index):
                    tensor.yield %arg1 : f32
                  } : tensor<2x?xf32>
                  %cast_0 = tensor.cast %generated : tensor<2x?xf32> to tensor<?x?xf32>
                  scf.yield %cast_0 : tensor<?x?xf32>
                } else {
                  %extracted_slice = tensor.extract_slice %arg0[%6, %14] [%9, %17] [1, 1] : tensor<4x16xf32> to tensor<?x?xf32>
                  %padded = tensor.pad %extracted_slice low[%4, %12] high[%11, %20] {
                  ^bb0(%arg6: index, %arg7: index):
                    tensor.yield %arg1 : f32
                  } : tensor<?x?xf32> to tensor<?x?xf32>
                  scf.yield %padded : tensor<?x?xf32>
                }
                %cast = tensor.cast %21 : tensor<?x?xf32> to tensor<2x?xf32>
                %inserted_slice = tensor.insert_slice %cast into %arg5[%arg2, %arg4] [2, %3] [1, 1] : tensor<2x?xf32> into tensor<12x23xf32>
                scf.yield %inserted_slice : tensor<12x23xf32>
              }
              scf.yield %2 : tensor<12x23xf32>
            }
            return %1 : tensor<12x23xf32>
          }
        }
        """
        )
        check_correct(correct, module)

    def test_linalg_tile(self):
        with mlir_mod_ctx() as module:

            @mlir_func
            def matmul(
                arg0: Tensor[(4, 16), F32],
                arg1: Tensor[(16, 8), F32],
                out: Tensor[(4, 8), F32],
            ):
                return linalg.matmul(arg0, arg1, outs=[out])

            @sequence
            def basic(target, *extra_args):
                m = match(target, ["linalg.matmul"])
                tile_linalg_to_scf_for(m, sizes=[2, 3])

        correct = dedent(
            """\
        module {
          func.func @matmul(%arg0: tensor<4x16xf32>, %arg1: tensor<16x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
            %0 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<4x16xf32>, tensor<16x8xf32>) outs(%arg2 : tensor<4x8xf32>) -> tensor<4x8xf32>
            return %0 : tensor<4x8xf32>
          }
          transform.sequence  failures(propagate) attributes {transform.target_tag = "basic"} {
          ^bb0(%arg0: !pdl.operation):
            %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!pdl.operation) -> !pdl.operation
            %tiled_linalg_op, %loops:2 = transform.structured.tile %0[2, 3] : (!pdl.operation) -> (!pdl.operation, !transform.op<"scf.for">, !transform.op<"scf.for">)
          }
        }
        """
        )
        check_correct(correct, module)

        run_pipeline(
            module,
            Pipeline()
            .transform_dialect_interpreter()
            .transform_dialect_erase_schedule()
            .materialize(),
        )

        # print(module)
        correct = dedent(
            """\
        #map = affine_map<(d0) -> (3, -d0 + 8)>
        #map1 = affine_map<(d0) -> (d0 - 1)>
        module {
          func.func @matmul(%arg0: tensor<4x16xf32>, %arg1: tensor<16x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
            %c2 = arith.constant 2 : index
            %c3 = arith.constant 3 : index
            %c0 = arith.constant 0 : index
            %c0_0 = arith.constant 0 : index
            %c4 = arith.constant 4 : index
            %0 = scf.for %arg3 = %c0_0 to %c4 step %c2 iter_args(%arg4 = %arg2) -> (tensor<4x8xf32>) {
              %c0_1 = arith.constant 0 : index
              %c8 = arith.constant 8 : index
              %1 = scf.for %arg5 = %c0_1 to %c8 step %c3 iter_args(%arg6 = %arg4) -> (tensor<4x8xf32>) {
                %c8_2 = arith.constant 8 : index
                %2 = affine.min #map(%arg5)
                %c0_3 = arith.constant 0 : index
                %c16 = arith.constant 16 : index
                %3 = affine.apply #map1(%2)
                %4 = affine.apply #map1(%2)
                %5 = affine.apply #map1(%2)
                %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [2, 16] [1, 1] : tensor<4x16xf32> to tensor<2x16xf32>
                %extracted_slice_4 = tensor.extract_slice %arg1[0, %arg5] [16, %2] [1, 1] : tensor<16x8xf32> to tensor<16x?xf32>
                %extracted_slice_5 = tensor.extract_slice %arg6[%arg3, %arg5] [2, %2] [1, 1] : tensor<4x8xf32> to tensor<2x?xf32>
                %6 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%extracted_slice, %extracted_slice_4 : tensor<2x16xf32>, tensor<16x?xf32>) outs(%extracted_slice_5 : tensor<2x?xf32>) -> tensor<2x?xf32>
                %7 = affine.apply #map1(%2)
                %8 = affine.apply #map1(%2)
                %inserted_slice = tensor.insert_slice %6 into %arg6[%arg3, %arg5] [2, %2] [1, 1] : tensor<2x?xf32> into tensor<4x8xf32>
                scf.yield %inserted_slice : tensor<4x8xf32>
              }
              scf.yield %1 : tensor<4x8xf32>
            }
            return %0 : tensor<4x8xf32>
          }
        }
        """
        )
        check_correct(correct, module)

    def test_simple_matmul_tile_runtime(self):
        with mlir_mod_ctx() as module:

            @mlir_func
            def matmul(
                arg0: Tensor[(4, 16), F32],
                arg1: Tensor[(16, 8), F32],
                out: Tensor[(4, 8), F32],
            ):
                return linalg.matmul(arg0, arg1, outs=[out])

            @sequence
            def basic(target, *extra_args):
                m = match(target, ["linalg.matmul"])
                tile_linalg_to_scf_for(m, sizes=[2, 3])

        module = self.backend.compile(
            module,
            kernel_name="matmul",
            pipeline=Pipeline()
            .bufferize()
            .transform_dialect_interpreter()
            .transform_dialect_erase_schedule()
            .FUNC()
            .convert_linalg_to_loops()
            .linalg_bufferize()
            .convert_scf_to_cf()
            .convert_linalg_to_loops()
            .linalg_bufferize()
            .CNUF()
            .arith_bufferize()
            .FUNC()
            .tensor_bufferize()
            .CNUF()
            .func_bufferize()
            .FUNC()
            .finalizing_bufferize()
            .CNUF()
            .refbackend_munge_calling_conventions()
            .convert_linalg_to_llvm()
            .expand_strided_metadata()
            .lower_affine()
            .convert_arith_to_llvm()
            .convert_scf_to_cf()
            .finalize_memref_to_llvm()
            .convert_func_to_llvm()
            .reconcile_unrealized_casts(),
        )

        result = None

        def callback(*args):
            nonlocal result
            result = tuple(
                [
                    arg
                    if type in elemental_type_to_ctype
                    else unranked_memref_to_numpy(arg, memref_type_to_np_dtype[type])
                    for arg, type in zip(args, invoker.ret_types)
                ]
            )
            assert len(args) == 1
            result = result[0]

        invoker = self.backend.load(module, consume_return_func=callback)
        A = np.random.randint(low=0, high=10, size=(4, 16)).astype(np.float32)
        B = np.random.randint(low=0, high=10, size=(16, 8)).astype(np.float32)
        C = np.zeros((4, 8)).astype(np.float32)
        invoker.matmul(A, B, C)
        assert np.allclose(A @ B, result)

    def test_simple_matmul_tile_foreach_thread(self):
        with mlir_mod_ctx() as module:

            @mlir_func
            def matmul(
                arg0: Tensor[(4, 16), F32],
                arg1: Tensor[(16, 8), F32],
                out: Tensor[(4, 8), F32],
            ):
                return linalg.matmul(arg0, arg1, outs=[out])

            @sequence
            def basic(target, *extra_args):
                m = match(target, ["linalg.matmul"])
                tile_to_scf_forall(m, tile_sizes=[2, 3])

        correct = dedent(
            """\
        module {
          func.func @matmul(%arg0: tensor<4x16xf32>, %arg1: tensor<16x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
            %0 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<4x16xf32>, tensor<16x8xf32>) outs(%arg2 : tensor<4x8xf32>) -> tensor<4x8xf32>
            return %0 : tensor<4x8xf32>
          }
          transform.sequence  failures(propagate) attributes {transform.target_tag = "basic"} {
          ^bb0(%arg0: !pdl.operation):
            %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!pdl.operation) -> !pdl.operation
            %forall_op, %tiled_op = transform.structured.tile_to_forall_op %0   tile_sizes [2, 3] : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
          }
        }
        """
        )
        check_correct(correct, module)
        run_pipeline(
            module,
            Pipeline()
            .transform_dialect_interpreter()
            .transform_dialect_erase_schedule()
            .materialize(),
        )
        correct = dedent(
            """\
        #map = affine_map<(d0) -> (d0 * 2)>
        #map1 = affine_map<(d0) -> (d0 * 3)>
        #map2 = affine_map<(d0) -> (d0 * -3 + 8)>
        #map3 = affine_map<(d0) -> (d0 * -3 + 8, 3)>
        #map4 = affine_map<(d0) -> (d0 - 1)>
        module {
          func.func @matmul(%arg0: tensor<4x16xf32>, %arg1: tensor<16x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
            %c2 = arith.constant 2 : index
            %c3 = arith.constant 3 : index
            %0 = scf.forall (%arg3, %arg4) in (2, 3) shared_outs(%arg5 = %arg2) -> (tensor<4x8xf32>) {
              %1 = affine.apply #map(%arg3)
              %2 = affine.apply #map1(%arg4)
              %3 = affine.apply #map2(%arg4)
              %4 = affine.min #map3(%arg4)
              %5 = affine.apply #map4(%4)
              %6 = affine.apply #map(%arg3)
              %7 = affine.apply #map1(%arg4)
              %8 = affine.apply #map4(%4)
              %9 = affine.apply #map(%arg3)
              %10 = affine.apply #map1(%arg4)
              %11 = affine.apply #map4(%4)
              %extracted_slice = tensor.extract_slice %arg0[%6, 0] [2, 16] [1, 1] : tensor<4x16xf32> to tensor<2x16xf32>
              %extracted_slice_0 = tensor.extract_slice %arg1[0, %7] [16, %4] [1, 1] : tensor<16x8xf32> to tensor<16x?xf32>
              %extracted_slice_1 = tensor.extract_slice %arg5[%9, %10] [2, %4] [1, 1] : tensor<4x8xf32> to tensor<2x?xf32>
              %12 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%extracted_slice, %extracted_slice_0 : tensor<2x16xf32>, tensor<16x?xf32>) outs(%extracted_slice_1 : tensor<2x?xf32>) -> tensor<2x?xf32>
              %13 = affine.apply #map4(%4)
              %14 = affine.apply #map(%arg3)
              %15 = affine.apply #map1(%arg4)
              %16 = affine.apply #map4(%4)
              scf.forall.in_parallel {
                tensor.parallel_insert_slice %12 into %arg5[%14, %15] [2, %4] [1, 1] : tensor<2x?xf32> into tensor<4x8xf32>
              }
            }
            return %0 : tensor<4x8xf32>
          }
        }
        """
        )
        check_correct(correct, module)

    def test_gpu_foreach_thread(self):
        with mlir_mod_ctx() as module:
            module = module.parse(
                dedent(
                    """\
            module {
              func.func @matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
                %0 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                                  outs(%C : tensor<?x?xf32>) -> (tensor<?x?xf32>)
                return %0 : tensor<?x?xf32>
              }

              transform.sequence failures(propagate) {
              ^bb1(%arg1: !transform.any_op):
                %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
                %1:2 = transform.structured.tile_to_forall_op %0 num_threads [10, 20] (mapping = [ #gpu.thread<y>, #gpu.thread<x> ] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
              }
            }
            """
                )
            )

        run_pipeline(
            module,
            Pipeline()
            .transform_dialect_interpreter()
            .transform_dialect_erase_schedule()
            .materialize(),
        )

        correct = dedent(
            """\
        #map = affine_map<()[s0] -> (s0 ceildiv 10)>
        #map1 = affine_map<(d0)[s0] -> (d0 * (s0 ceildiv 10))>
        #map2 = affine_map<()[s0] -> (-s0 + (s0 ceildiv 10) * 10)>
        #map3 = affine_map<(d0)[s0] -> (-(d0 * (s0 ceildiv 10)) + s0)>
        #map4 = affine_map<(d0)[s0] -> (-(d0 * (s0 ceildiv 10)) + s0, s0 ceildiv 10)>
        #map5 = affine_map<(d0) -> (0, d0)>
        #map6 = affine_map<()[s0] -> (s0 ceildiv 20)>
        #map7 = affine_map<(d0)[s0] -> (d0 * (s0 ceildiv 20))>
        #map8 = affine_map<()[s0] -> (-s0 + (s0 ceildiv 20) * 20)>
        #map9 = affine_map<(d0)[s0] -> (-(d0 * (s0 ceildiv 20)) + s0)>
        #map10 = affine_map<(d0)[s0] -> (-(d0 * (s0 ceildiv 20)) + s0, s0 ceildiv 20)>
        #map11 = affine_map<(d0) -> (d0 - 1)>
        #map12 = affine_map<()[s0] -> (s0 - 1)>
        module {
          func.func @matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
            %c0 = arith.constant 0 : index
            %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
            %c1 = arith.constant 1 : index
            %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
            %c0_1 = arith.constant 0 : index
            %dim_2 = tensor.dim %arg1, %c0_1 : tensor<?x?xf32>
            %c1_3 = arith.constant 1 : index
            %dim_4 = tensor.dim %arg1, %c1_3 : tensor<?x?xf32>
            %c0_5 = arith.constant 0 : index
            %dim_6 = tensor.dim %arg2, %c0_5 : tensor<?x?xf32>
            %c1_7 = arith.constant 1 : index
            %dim_8 = tensor.dim %arg2, %c1_7 : tensor<?x?xf32>
            %c10 = arith.constant 10 : index
            %c20 = arith.constant 20 : index
            %0 = scf.forall (%arg3, %arg4) in (10, 20) shared_outs(%arg5 = %arg2) -> (tensor<?x?xf32>) {
              %1 = affine.apply #map()[%dim]
              %2 = affine.apply #map1(%arg3)[%dim]
              %3 = affine.apply #map2()[%dim]
              %4 = affine.apply #map3(%arg3)[%dim]
              %5 = affine.min #map4(%arg3)[%dim]
              %6 = affine.max #map5(%5)
              %7 = affine.apply #map6()[%dim_4]
              %8 = affine.apply #map7(%arg4)[%dim_4]
              %9 = affine.apply #map8()[%dim_4]
              %10 = affine.apply #map9(%arg4)[%dim_4]
              %11 = affine.min #map10(%arg4)[%dim_4]
              %12 = affine.max #map5(%11)
              %13 = affine.apply #map11(%6)
              %14 = affine.apply #map11(%12)
              %15 = affine.apply #map12()[%dim_0]
              %16 = affine.apply #map1(%arg3)[%dim]
              %17 = affine.apply #map11(%6)
              %18 = affine.apply #map12()[%dim_0]
              %19 = affine.apply #map12()[%dim_0]
              %20 = affine.apply #map7(%arg4)[%dim_4]
              %21 = affine.apply #map11(%12)
              %22 = affine.apply #map1(%arg3)[%dim]
              %23 = affine.apply #map11(%6)
              %24 = affine.apply #map7(%arg4)[%dim_4]
              %25 = affine.apply #map11(%12)
              %extracted_slice = tensor.extract_slice %arg0[%16, 0] [%6, %dim_0] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
              %extracted_slice_9 = tensor.extract_slice %arg1[0, %20] [%dim_0, %12] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
              %extracted_slice_10 = tensor.extract_slice %arg5[%22, %24] [%6, %12] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
              %26 = linalg.matmul ins(%extracted_slice, %extracted_slice_9 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_10 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %27 = affine.apply #map11(%6)
              %28 = affine.apply #map11(%12)
              %29 = affine.apply #map12()[%dim_0]
              %30 = affine.apply #map1(%arg3)[%dim]
              %31 = affine.apply #map11(%6)
              %32 = affine.apply #map7(%arg4)[%dim_4]
              %33 = affine.apply #map11(%12)
              scf.forall.in_parallel {
                tensor.parallel_insert_slice %26 into %arg5[%30, %32] [%6, %12] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
              }
            } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
            return %0 : tensor<?x?xf32>
          }
        }
        """
        )
        check_correct(correct, module)

    def test_contraction_matmul(self):
        with mlir_mod_ctx() as module:
            module = module.parse(
                dedent(
                    """\
            func.func @contraction_matmul(%A: memref<10x10xf32>, %B: memref<10x10xf32>, %C: memref<10x10xf32>) {
              linalg.matmul ins(%A, %B: memref<10x10xf32>, memref<10x10xf32>)
                        outs(%C: memref<10x10xf32>)
              return
            }

            transform.sequence failures(propagate) {
            ^bb1(%arg1: !transform.any_op):
              %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
              %1 = get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
              %2 = transform.structured.vectorize %1  { disable_multi_reduction_to_contract_patterns } : (!transform.any_op) -> !transform.any_op
            }
            """
                )
            )
        run_pipeline(
            module,
            Pipeline()
            .transform_dialect_interpreter()
            .transform_dialect_erase_schedule()
            .materialize(),
        )

        correct = dedent(
            """\
        module {
          func.func @contraction_matmul(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>, %arg2: memref<10x10xf32>) {
            %c0 = arith.constant 0 : index
            %cst = arith.constant 0.000000e+00 : f32
            %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<10x10xf32>, vector<10x10xf32>
            %1 = vector.broadcast %0 : vector<10x10xf32> to vector<10x10x10xf32>
            %2 = vector.transpose %1, [1, 0, 2] : vector<10x10x10xf32> to vector<10x10x10xf32>
            %3 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<10x10xf32>, vector<10x10xf32>
            %4 = vector.broadcast %3 : vector<10x10xf32> to vector<10x10x10xf32>
            %5 = vector.transpose %4, [0, 2, 1] : vector<10x10x10xf32> to vector<10x10x10xf32>
            %6 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<10x10xf32>, vector<10x10xf32>
            %7 = arith.mulf %2, %5 : vector<10x10x10xf32>
            %8 = vector.multi_reduction <add>, %7, %6 [2] : vector<10x10x10xf32> to vector<10x10xf32>
            vector.transfer_write %8, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<10x10xf32>, memref<10x10xf32>
            return
          }
        }
        """
        )
        check_correct(correct, module)

    def test_contraction_matmul_runtime(self):
        with mlir_mod_ctx() as module:
            module = module.parse(
                dedent(
                    """\
            func.func @contraction_matmul(%A: memref<10x10xf32>, %B: memref<10x10xf32>, %C: memref<10x10xf32>) {
              linalg.matmul ins(%A, %B: memref<10x10xf32>, memref<10x10xf32>)
                        outs(%C: memref<10x10xf32>)
              return
            }

            transform.sequence failures(propagate) {
            ^bb1(%arg1: !transform.any_op):
              %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
              %1 = get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
              %2 = transform.structured.vectorize %1 : (!transform.any_op) -> !transform.any_op
            }
            """
                )
            )

        module = self.backend.compile(
            module,
            kernel_name="contraction_matmul",
            pipeline=Pipeline()
            .transform_dialect_interpreter()
            .transform_dialect_erase_schedule()
            .bufferize()
            .FUNC()
            .convert_vector_to_scf(full_unroll=True)
            .CNUF()
            .convert_vector_to_llvm()
            .finalize_memref_to_llvm()
            .lower_to_llvm(),
        )

        invoker = self.backend.load(module)
        A = np.random.randint(low=0, high=10, size=(10, 10)).astype(np.float32)
        B = np.random.randint(low=0, high=4, size=(10, 10)).astype(np.float32)
        C = np.zeros((10, 10)).astype(np.float32)
        invoker.contraction_matmul(A, B, C)
        assert np.allclose(A @ B, C)

    def test_common_extension_with_pdl_patterns(self):
        with mlir_mod_ctx() as module:
            module = module.parse(
                dedent(
                    """\
            func.func @select_cmp_eq_select(%arg0: i64, %arg1: i64) -> i64 {
              %0 = arith.cmpi eq, %arg0, %arg1 : i64
              %1 = arith.select %0, %arg0, %arg1 : i64
              return %1 : i64
            }

            transform.with_pdl_patterns {
            ^bb0(%arg0: !transform.any_op):
              transform.sequence %arg0 : !transform.any_op failures(propagate) {
              ^bb1(%arg1: !transform.any_op):
                %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
                transform.apply_patterns to %0 {
                    transform.apply_patterns.canonicalization
                  } : !transform.any_op
              }
            }
            """
                )
            )

        module = self.backend.compile(
            module,
            pipeline=Pipeline()
            .transform_dialect_interpreter()
            .transform_dialect_erase_schedule(),
        )

        correct = dedent(
            """\
        module {
          func.func @select_cmp_eq_select(%arg0: i64, %arg1: i64) -> i64 {
            return %arg1 : i64
          }
        }
        """
        )
        check_correct(correct, module)

    def test_common_extension_sequence(self):
        with mlir_mod_ctx() as module:
            module = module.parse(
                dedent(
                    """\
            func.func @select_cmp_eq_select(%arg0: i64, %arg1: i64) -> i64 {
              %0 = arith.cmpi eq, %arg0, %arg1 : i64
              %1 = arith.select %0, %arg0, %arg1 : i64
              return %1 : i64
            }

            transform.sequence  failures(propagate) {
            ^bb0(%arg0: !transform.any_op):
              %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
              transform.apply_patterns to %0 {
                transform.apply_patterns.canonicalization
              } : !transform.any_op
            }
            """
                )
            )

        module = self.backend.compile(
            module,
            pipeline=Pipeline()
            .transform_dialect_interpreter()
            .transform_dialect_erase_schedule(),
        )

        correct = dedent(
            """\
        module {
          func.func @select_cmp_eq_select(%arg0: i64, %arg1: i64) -> i64 {
            return %arg1 : i64
          }
        }
        """
        )
        check_correct(correct, module)

    def test_common_extension_sugar(self):
        with mlir_mod_ctx() as module:

            @mlir_func
            def select_cmp_eq_select(arg0: I64, arg1: I64):
                a = arg0 == arg1
                b = arith.select(a, arg0, arg1)
                return b

            @sequence
            def basic(target, *extra_args):
                m = match(target, ["func.func"])
                apply_patterns(m, canonicalization=True)

        correct = dedent(
            """\
        module {
          func.func @select_cmp_eq_select(%arg0: i64, %arg1: i64) -> i64 {
            %0 = arith.cmpi eq, %arg0, %arg1 : i64
            %1 = arith.select %0, %arg0, %arg1 : i64
            return %1 : i64
          }
          transform.sequence  failures(propagate) attributes {transform.target_tag = "basic"} {
          ^bb0(%arg0: !transform.any_op):
            %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
            apply_patterns %0 {canonicalization} : (!transform.any_op) -> ()
          }
        }
        """
        )
        check_correct(correct, module)
        module = self.backend.compile(
            module,
            pipeline=Pipeline()
            .transform_dialect_interpreter()
            .transform_dialect_erase_schedule(),
        )

        correct = dedent(
            """\
        module {
          func.func @select_cmp_eq_select(%arg0: i64, %arg1: i64) -> i64 {
            return %arg1 : i64
          }
        }
        """
        )
        check_correct(correct, module)

    def test_common_ext_2(self):
        src = dedent(
            """\
        #map0 = affine_map<(d0, d1) -> (d0, d1)>
        #map2 = affine_map<(d0) -> (d0 * 4)>

        func.func @promote() -> (tensor<16x128xf32>) {
          %c0 = arith.constant 0 : index
          %f0 = arith.constant 0.000000e+00 : f32
          %c16 = arith.constant 16 : index
          %c32 = arith.constant 32 : index

          %empty = tensor.empty() : tensor<16x128xf32>
          %filled = linalg.fill ins(%f0 : f32) outs(%empty : tensor<16x128xf32>) -> tensor<16x128xf32>

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
          return %10 : tensor<16x128xf32>
        }

        transform.sequence failures(propagate) {
        ^bb1(%arg1: !transform.any_op):
          %0 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
          %1 = transform.cast %0 : !transform.any_op to !transform.op<"scf.forall">
          transform.share_forall_operands %1 share_operands = [0] : (!transform.op<"scf.forall">) -> !transform.op<"scf.forall">
        }
        """
        )
        with mlir_mod_ctx(src) as module:
            pass

        module = self.backend.compile(
            module,
            pipeline=Pipeline()
            .transform_dialect_interpreter()
            .transform_dialect_erase_schedule(),
        )
        # all it does is replace filled with arg2 in the generic?
        correct = dedent(
            """\
        #map = affine_map<(d0) -> (d0 * 4)>
        #map1 = affine_map<(d0, d1) -> (d0, d1)>
        module {
          func.func @promote() -> tensor<16x128xf32> {
            %c0 = arith.constant 0 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c16 = arith.constant 16 : index
            %c32 = arith.constant 32 : index
            %0 = tensor.empty() : tensor<16x128xf32>
            %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x128xf32>) -> tensor<16x128xf32>
            %2 = scf.forall (%arg0, %arg1) in (%c16, %c32) shared_outs(%arg2 = %1) -> (tensor<16x128xf32>) {
              %3 = affine.apply #map(%arg1)
              %extracted_slice = tensor.extract_slice %arg2[%arg0, %3] [1, 4] [1, 1] : tensor<16x128xf32> to tensor<1x4xf32>
              %extracted_slice_0 = tensor.extract_slice %arg2[%arg0, %3] [1, 4] [1, 1] : tensor<16x128xf32> to tensor<1x4xf32>
              %4 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice : tensor<1x4xf32>) outs(%extracted_slice_0 : tensor<1x4xf32>) {
              ^bb0(%in: f32, %out: f32):
                %5 = arith.addf %in, %in : f32
                linalg.yield %5 : f32
              } -> tensor<1x4xf32>
              scf.forall.in_parallel {
                tensor.parallel_insert_slice %4 into %arg2[%arg0, %3] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<16x128xf32>
              }
            }
            return %2 : tensor<16x128xf32>
          }
        }
        """
        )
        check_correct(correct, module)

    def test_common_ext_2_sugar(self):
        with mlir_mod_ctx() as module:

            @mlir_func
            def promote():
                f0 = arith.constant(0.0, type=F32)

                empty = Tensor.empty([16, 128], F32)
                filled = linalg.fill(f0, outs=[empty])

                @forall((16, 32), shared_outs=[filled])
                def forall_(ivs, shared_outs):
                    i, j = ivs
                    jj = 4 * j
                    arg2 = shared_outs[0]

                    extracted_slice = extract_slice(
                        filled,
                        offsets=[i, jj],
                        static_sizes=[1, 4],
                        static_strides=[1, 1],
                    )
                    extracted_slice_2 = extract_slice(
                        arg2,
                        offsets=[i, jj],
                        static_sizes=[1, 4],
                        static_strides=[1, 1],
                    )
                    v13 = linalg.elemwise_binary(
                        extracted_slice, extracted_slice, outs=[extracted_slice_2]
                    )
                    extracted_slice = parallel_insert_slice(
                        v13,
                        arg2,
                        offsets=[i, jj],
                        static_sizes=[1, 4],
                        static_strides=[1, 1],
                    )
                    return extracted_slice

            @sequence
            def basic(target, *extra_args):
                m = match(target, ["scf.forall"])
                k = cast("scf.forall", m)
                share_forall_operands(k, share_operands=[0])

        correct = dedent(
            """\
        module {
          func.func @promote() {
            %cst = arith.constant 0.000000e+00 : f32
            %0 = tensor.empty() : tensor<16x128xf32>
            %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x128xf32>) -> tensor<16x128xf32>
            %2 = scf.forall (%arg0, %arg1) in (16, 32) shared_outs(%arg2 = %1) -> (tensor<16x128xf32>) {
              %c4 = arith.constant 4 : index
              %3 = arith.muli %c4, %arg1 : index
              %extracted_slice = tensor.extract_slice %1[%arg0, %3] [1, 4] [1, 1] : tensor<16x128xf32> to tensor<1x4xf32>
              %extracted_slice_0 = tensor.extract_slice %arg2[%arg0, %3] [1, 4] [1, 1] : tensor<16x128xf32> to tensor<1x4xf32>
              %4 = linalg.elemwise_binary {cast = #linalg.type_fn<cast_signed>, fun = #linalg.binary_fn<add>} ins(%extracted_slice, %extracted_slice : tensor<1x4xf32>, tensor<1x4xf32>) outs(%extracted_slice_0 : tensor<1x4xf32>) -> tensor<1x4xf32>
              scf.forall.in_parallel {
                tensor.parallel_insert_slice %4 into %arg2[%arg0, %3] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<16x128xf32>
              }
            }
            return
          }
          transform.sequence  failures(propagate) attributes {transform.target_tag = "basic"} {
          ^bb0(%arg0: !transform.any_op):
            %0 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
            %1 = cast %0 : !transform.any_op to !transform.op<"scf.forall">
            %2 = share_forall_operands %1 share_operands = [0] : (!transform.op<"scf.forall">) -> !transform.op<"scf.forall">
          }
        }
        """
        )
        check_correct(correct, module)

        module = self.backend.compile(
            module,
            pipeline=Pipeline()
            .transform_dialect_interpreter()
            .transform_dialect_erase_schedule(),
        )
        correct = dedent(
            """\
        module {
          func.func @promote() {
            %cst = arith.constant 0.000000e+00 : f32
            %0 = tensor.empty() : tensor<16x128xf32>
            %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x128xf32>) -> tensor<16x128xf32>
            %2 = scf.forall (%arg0, %arg1) in (16, 32) shared_outs(%arg2 = %1) -> (tensor<16x128xf32>) {
              %c4 = arith.constant 4 : index
              %3 = arith.muli %c4, %arg1 : index
              %extracted_slice = tensor.extract_slice %arg2[%arg0, %3] [1, 4] [1, 1] : tensor<16x128xf32> to tensor<1x4xf32>
              %extracted_slice_0 = tensor.extract_slice %arg2[%arg0, %3] [1, 4] [1, 1] : tensor<16x128xf32> to tensor<1x4xf32>
              %4 = linalg.elemwise_binary {cast = #linalg.type_fn<cast_signed>, fun = #linalg.binary_fn<add>} ins(%extracted_slice, %extracted_slice : tensor<1x4xf32>, tensor<1x4xf32>) outs(%extracted_slice_0 : tensor<1x4xf32>) -> tensor<1x4xf32>
              scf.forall.in_parallel {
                tensor.parallel_insert_slice %4 into %arg2[%arg0, %3] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<16x128xf32>
              }
            }
            return
          }
        }
        """
        )
        check_correct(correct, module)

    @pytest.mark.xfail()
    def conv_2d_nhwc_hwcf_codegen_spec(self):
        with mlir_mod_ctx() as module:
            N, H, W, C = 1, 41, 140, 1
            KH, KW, F = 1, 140, 128
            OH, OW = 41, 1

            @mlir_func
            def conv_2d_nhwc_hwcf(
                input: Tensor[(N, H, W, C), F32],
                kernel: Tensor[(KH, KW, C, F), F32],
                output: Tensor[(N, OH, OW, F), F32],
            ):
                return linalg.conv_2d_nhwc_hwcf(input, kernel, outs=[output])

            @sequence
            def basic(variant_op, *extra_args):
                named_conv = match(variant_op, ["linalg.conv_2d_nhwc_hwcf"])
                conv = generalize(named_conv)
                func = match(variant_op, ["func.func"])
                apply_patterns(func, rank_reducing_linalg=True)
                apply_patterns(
                    variant_op, canonicalization=True, tiling_canonicalization=True
                )

                # Step 1. Tile to forall and sequential scf.for.
                forall_l1, conv_l1 = tile_to_scf_forall(
                    conv,
                    tile_sizes=[1],
                    mapping={
                        0: block_attr("x"),
                    },
                )
                apply_patterns(
                    variant_op,
                    canonicalization=True,
                    cse=True,
                    tiling_canonicalization=True,
                )

                # Step 2. Tile to sequential scf.for
                matmul_l2, *loops_l1_3 = tile_to_scf_for(conv_l1, tile_sizes=[1, 4, 4])
                apply_patterns(variant_op, canonicalization=True, cse=True, licm=True)

                # Step 3. Vectorize
                func_v = match(variant_op, ["func.func"])
                apply_patterns(func_v, rank_reducing_linalg=True)
                func_v_2 = vectorize(func_v)
                apply_patterns(func_v_2, canonicalization=True, cse=True, licm=True)
                hoist_redundant_tensor_subsets(func_v_2)

                # Bufferization
                apply_patterns(variant_op, canonicalization=True, cse=True, licm=True)
                bufferize(variant_op)

            @sequence
            def final(variant_op, *extra_args):
                func_e = match(variant_op, ["func.func"])
                lower_vectors(
                    func_e,
                    contraction_lowering="outerproduct",
                    transpose_lowering="shuffle",
                )

        module = self.backend.compile(
            module,
            pipeline=Pipeline()
            .transform_dialect_interpreter(debug_transform_root_tag="basic")
            .bufferize()
            .transform_dialect_interpreter(debug_transform_root_tag="final")
            .transform_dialect_erase_schedule(),
        )
        print(module)

    def test_matmul(self):
        with mlir_mod_ctx() as module:
            M, N, K = 3, 5, 3

            @mlir_func
            def matmul(
                A: Tensor[(M, N), F32],
                B: Tensor[(N, K), F32],
                C: Tensor[(M, K), F32],
            ):
                return linalg.matmul(A, B, outs=[C])

            @sequence
            def basic(variant_op, *extra_args):
                matmul = match(variant_op, ["linalg.matmul"])

                # Step 1. Tile to forall with tile_sizes [2].
                forall, tiled_generic = tile_to_scf_forall(
                    matmul, tile_sizes=[2], mapping={0: block_attr("x")}
                )
                apply_patterns(
                    variant_op,
                    canonicalization=True,
                    tiling_canonicalization=True,
                    cse=True,
                )
                bufferize(variant_op)

        module = self.backend.compile(
            module,
            pipeline=Pipeline()
            .transform_dialect_interpreter()
            .transform_dialect_erase_schedule()
            .bufferize(),
        )
        print(module)

    @pytest.mark.xfail()
    def test_reductions_codegen_spec2(self):
        src = dedent(
            """\
            !in_tensor_t = tensor<8x64xf32>
            !out_tensor_t = tensor<8xf32>

            func.func @reduce(%arg : !in_tensor_t) -> (!out_tensor_t) {
              %cst = arith.constant -0.000000e+00 : f32

              %0 = tensor.empty() : !out_tensor_t
              %1 = linalg.fill ins(%cst : f32) outs(%0 : !out_tensor_t) ->   !out_tensor_t
              %2 = linalg.generic {
                indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                 affine_map<(d0, d1) -> (d0)>],
                iterator_types = ["parallel", "reduction"]}
                ins(%arg : !in_tensor_t) outs(%1 : !out_tensor_t) {
                  ^bb0(%arg3: f32, %arg4: f32):
                    %3 = arith.addf %arg3, %arg4 : f32
                    linalg.yield %3 : f32
                  } -> !out_tensor_t
              return %2 : !out_tensor_t
            }

            transform.sequence failures(propagate) attributes {transform.target_tag = "basic"} {
            ^bb1(%variant_op: !transform.any_op):
              %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op

              // Step 1. Split the reduction to get meatier (size(red) / 2)-way parallelism.
              // ===========================================================================
              %0 = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op
              %init_or_alloc_op, %more_parallel_fill_op, %more_parallel_op, %combiner_op =
                transform.structured.split_reduction %0
                  { split_factor = 2, insert_split_dimension = 1 }

              // Step 2. First level of tiling + fusion parallelizes to blocks.
              // ===========================================================================
              %forall_grid, %grid_combiner_op =
                transform.structured.tile_to_forall_op %combiner_op tile_sizes [1]
                  ( mapping = [#gpu.block<x>] )
              %not_combiner = transform.merge_handles %fill, %more_parallel_fill_op, %more_parallel_op : !transform.any_op
              transform.structured.fuse_into_containing_op %not_combiner into %forall_grid

              // Step 3. Second level of tiling + fusion parallelizes to threads.
              // ===========================================================================
              %fill_1d = transform.structured.match ops{["linalg.fill"]} filter_result_type = tensor<1xf32> in %variant_op
                : (!transform.any_op) -> !transform.any_op
              %forall_block_combiner_op, %block_combiner_op =
                transform.structured.tile_to_forall_op %grid_combiner_op tile_sizes [1]
                ( mapping = [#gpu.thread<z>] )
              transform.structured.fuse_into_containing_op %fill_1d into %forall_block_combiner_op

              // Canonicalizations.
              transform.apply_patterns %variant_op
                { canonicalization, tiling_canonicalization, licm, cse } : (!transform.any_op) -> ()

              %fill_2d = transform.structured.match ops{["linalg.fill"]} filter_result_type = tensor<1x2xf32> in %variant_op
                : (!transform.any_op) -> !transform.any_op
              %grid_more_parallel_op = transform.structured.match ops{["linalg.generic"]}
                attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]} in %variant_op
                  : (!transform.any_op) -> !transform.any_op
              %forall_block_more_parallel_op, %block_more_parallel_op =
                transform.structured.tile_to_forall_op %grid_more_parallel_op tile_sizes [1, 1]
                ( mapping = [#gpu.thread<z>, #gpu.thread<y>] )
              transform.structured.fuse_into_containing_op %fill_2d into %forall_block_more_parallel_op

              // Step 4. Rank-reduce and vectorize.
              // ===========================================================================
              %func = transform.structured.match ops{["func.func"]} in %variant_op
                : (!transform.any_op) -> !transform.any_op
              transform.apply_patterns %func {  rank_reducing_linalg, rank_reducing_vector } : (!transform.any_op) -> ()
              %func_3 = transform.structured.vectorize %func

              // Step 5. Bufferize and drop HAL decriptor from memref ops.
              // ===========================================================================
              transform.apply_patterns %func_3 { fold_reassociative_reshapes } : (!transform.any_op) -> ()
              transform.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
              %variant_op_3 = transform.bufferize { target_gpu } %variant_op : (!transform.any_op) -> !transform.any_op

              // Step 6. Post-bufferization mapping to blocks and threads.
              // ===========================================================================
              %func_4 = transform.structured.match ops{["func.func"]} in %variant_op_3
                : (!transform.any_op) -> !transform.any_op
              %gpu_launch = transform.gpu.map_forall_to_blocks %func_4
                grid_dims = [2] {generate_gpu_launch}
              transform.gpu.map_nested_forall_to_threads %gpu_launch
                block_dims = [32, 2, 1]

              %func_5 = transform.structured.match ops{["func.func"]} in %variant_op_3
                : (!transform.any_op) -> !transform.any_op
              transform.apply_patterns %func_5 { rank_reducing_linalg, rank_reducing_vector, fold_memref_aliases, cse, canonicalize } : (!transform.any_op) -> ()
            }

            transform.sequence failures(propagate) attributes {transform.target_tag = "final"} {
            ^bb1(%variant_op: !transform.any_op):
              %if_op = transform.structured.match ops{["scf.if"]} in %variant_op
                : (!transform.any_op) -> !transform.any_op
              transform.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 32 }
              %func_5 = transform.structured.match ops{["func.func"]} in %variant_op
                : (!transform.any_op) -> !transform.any_op
              transform.vector.warp_distribute %func_5 : (!transform.any_op) -> ()
              transform.apply_patterns %variant_op
                { canonicalization, tiling_canonicalization, licm, cse } : (!transform.any_op) -> ()

              // Hoist static allocs to allow multi-buffering to proceed.
              transform.hoist_static_alloc %func_5 : (!transform.any_op) -> ()
              %allocs = transform.structured.match ops{["memref.alloc"]} in %func_5
                : (!transform.any_op) -> !transform.op<"memref.alloc">
              %mb_allocs = transform.memref.multibuffer %allocs {factor = 5 : i64, skip_analysis }
                : (!transform.op<"memref.alloc">) -> !transform.any_op
              transform.apply_patterns %func_5 { canonicalization, cse, licm, fold_memref_aliases, tiling_canonicalization } : (!transform.any_op) -> ()

              // Step 5. Lower vectors.
              // ============================================================================
              transform.vector.lower_vectors %func_5 multireduction_lowering = "innerreduction"
              // Late canonicalizations and cleanups.

            }
        """
        )

        with mlir_mod_ctx(src) as module:
            pass

        module = self.backend.compile(
            module,
            kernel_name="reduce",
            pipeline=Pipeline()
            .symbol_dce()
            .transform_dialect_interpreter(debug_transform_root_tag="basic")
            .lower_affine()
            .cse()
            .canonicalize()
            .transform_dialect_interpreter(debug_transform_root_tag="final")
            .transform_dialect_erase_schedule()
            .lower_affine()
            .func_bufferize()
            .canonicalize()
            .cse()
            .gpu_kernel_outlining()
            .convert_scf_to_cf()
            .convert_to_nvvm()
            .GPU()
            .strip_debuginfo()
            .convert_gpu_to_nvvm()
            .UPG()
            .convert_arith_to_llvm()
            .canonicalize()
            .cse()
            .GPU()
            .gpu_to_cubin(chip="sm_70")
            .UPG()
            .FUNC()
            .canonicalize()
            .cse()
            .CNUF()
            # .gpu_to_llvm()
            ,
            # enable_ir_printing=True
        )
        print(module)
