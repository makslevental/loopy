from textwrap import dedent

import numpy as np
import pytest

from nelli import F32
from nelli.mlir import arith
from nelli.mlir._mlir.dialects import linalg
from nelli.mlir._mlir.ir import InsertionPoint
from nelli.mlir._mlir.runtime import unranked_memref_to_numpy
from nelli.mlir.func import mlir_func
from nelli.mlir.passes import Pipeline
from nelli.mlir.refbackend import (
    LLVMJITBackend,
    elemental_type_to_ctype,
    memref_type_to_np_dtype,
)
from nelli.mlir.scf import range as scf_range, endfor as scf_endfor
from nelli.mlir.tensor import TensorValue as Tensor, pad
from nelli.mlir.transform import (
    loop_ext,
    transform_dialect,
    sequence,
    match,
    get_parent_for_loop,
    unroll,
    tile_linalg_to_scf_for,
    tile_to_scf_foreach_thread,
    tile_to_scf_for,
)
from nelli.mlir.utils import run_pipeline_with_repro_report
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
                loop_ext.LoopUnrollOp(sequence.bodyTarget, factor=42)
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
                loop_ext.LoopUnrollOp(target, factor=42)

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

    def test_basic_unroll(self):
        with mlir_mod_ctx() as module:

            @mlir_func(range_ctor=scf_range, endfor=scf_endfor)
            def loop_unroll_op():
                for i in range(0, 42, 5):
                    v = arith.add(i, i)

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
          transform.sequence  failures(propagate) {
          ^bb0(%arg0: !pdl.operation):
            %0 = transform.structured.match ops{["arith.addi"]} in %arg0 : (!pdl.operation) -> !pdl.operation
            %1 = transform.loop.get_parent_for %0 : (!pdl.operation) -> !transform.op<"scf.for">
            transform.loop.unroll %1 {factor = 4 : i64} : !transform.op<"scf.for">
          }
        }
        """
        )
        check_correct(correct, module)

        run_pipeline_with_repro_report(
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
                tiled = tile_to_scf_for(m, sizes=[2, 3])

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
          transform.sequence  failures(propagate) {
          ^bb0(%arg0: !pdl.operation):
            %0 = transform.structured.match ops{["tensor.pad"]} in %arg0 : (!pdl.operation) -> !pdl.operation
            %tiled_linalg_op, %loops:2 = transform.structured.tile_to_scf_for %0[2, 3]
          }
        }
        """
        )
        check_correct(correct, module)
        run_pipeline_with_repro_report(
            module,
            Pipeline()
            .transform_dialect_interpreter()
            .transform_dialect_erase_schedule()
            .materialize(),
        )
        correct = dedent(
            """\
        #map = affine_map<(d0) -> (3, -d0 + 23)>
        #map1 = affine_map<(d0, d1) -> (d0 - d1)>
        #map2 = affine_map<(d0, d1) -> (0, d1)>
        #map3 = affine_map<(d0, d1) -> (d0, 0)>
        #map4 = affine_map<(d0, d1) -> (d0, 4)>
        #map5 = affine_map<(d0, d1) -> (d0 + d1)>
        #map6 = affine_map<(d0, d1) -> (d0, 16)>
        module {
          func.func @pad_tensor_3_4(%arg0: tensor<4x16xf32>, %arg1: f32) -> tensor<12x23xf32> {
            %0 = tensor.empty() : tensor<12x23xf32>
            %c0 = arith.constant 0 : index
            %c4 = arith.constant 4 : index
            %c12 = arith.constant 12 : index
            %c1 = arith.constant 1 : index
            %c16 = arith.constant 16 : index
            %c23 = arith.constant 23 : index
            %c0_0 = arith.constant 0 : index
            %c1_1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c3 = arith.constant 3 : index
            %1 = scf.for %arg2 = %c0_0 to %c12 step %c2 iter_args(%arg3 = %0) -> (tensor<12x23xf32>) {
              %2 = scf.for %arg4 = %c0_0 to %c23 step %c3 iter_args(%arg5 = %arg3) -> (tensor<12x23xf32>) {
                %3 = affine.min #map(%arg4)
                %c0_2 = arith.constant 0 : index
                %c3_3 = arith.constant 3 : index
                %c5 = arith.constant 5 : index
                %c0_4 = arith.constant 0 : index
                %c4_5 = arith.constant 4 : index
                %4 = affine.apply #map1(%c3_3, %arg2)
                %5 = affine.max #map2(%c0_2, %4)
                %6 = affine.apply #map1(%arg2, %c3_3)
                %7 = affine.max #map3(%6, %c0_2)
                %8 = affine.min #map4(%7, %c4_5)
                %9 = affine.apply #map1(%arg2, %c3_3)
                %10 = affine.apply #map5(%9, %c2)
                %11 = affine.max #map3(%10, %c0_2)
                %12 = affine.min #map4(%11, %c4_5)
                %13 = affine.apply #map1(%12, %8)
                %14 = arith.cmpi eq, %13, %c0_2 : index
                %15 = affine.apply #map1(%c2, %13)
                %16 = affine.apply #map1(%15, %5)
                %c4_6 = arith.constant 4 : index
                %c3_7 = arith.constant 3 : index
                %c1_8 = arith.constant 1 : index
                %c16_9 = arith.constant 16 : index
                %17 = affine.apply #map1(%c4_6, %arg4)
                %18 = affine.max #map2(%c0_2, %17)
                %19 = affine.apply #map1(%arg4, %c4_6)
                %20 = affine.max #map3(%19, %c0_2)
                %21 = affine.min #map6(%20, %c16_9)
                %22 = affine.apply #map1(%arg4, %c4_6)
                %23 = affine.apply #map5(%22, %3)
                %24 = affine.max #map3(%23, %c0_2)
                %25 = affine.min #map6(%24, %c16_9)
                %26 = affine.apply #map1(%25, %21)
                %27 = arith.cmpi eq, %26, %c0_2 : index
                %28 = arith.ori %27, %14 : i1
                %29 = affine.apply #map1(%3, %26)
                %30 = affine.apply #map1(%29, %18)
                %31 = scf.if %28 -> (tensor<?x?xf32>) {
                  %generated = tensor.generate %c2, %3 {
                  ^bb0(%arg6: index, %arg7: index):
                    tensor.yield %arg1 : f32
                  } : tensor<?x?xf32>
                  %cast = tensor.cast %generated : tensor<?x?xf32> to tensor<?x?xf32>
                  scf.yield %cast : tensor<?x?xf32>
                } else {
                  %extracted_slice = tensor.extract_slice %arg0[%8, %21] [%13, %26] [1, 1] : tensor<4x16xf32> to tensor<?x?xf32>
                  %padded = tensor.pad %extracted_slice low[%5, %18] high[%16, %30] {
                  ^bb0(%arg6: index, %arg7: index):
                    tensor.yield %arg1 : f32
                  } : tensor<?x?xf32> to tensor<?x?xf32>
                  %cast = tensor.cast %padded : tensor<?x?xf32> to tensor<?x?xf32>
                  scf.yield %cast : tensor<?x?xf32>
                }
                %inserted_slice = tensor.insert_slice %31 into %arg5[%arg2, %arg4] [%c2, %3] [1, 1] : tensor<?x?xf32> into tensor<12x23xf32>
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
                tiled = tile_linalg_to_scf_for(m, sizes=[2, 3])

        correct = dedent(
            """\
        module {
          func.func @matmul(%arg0: tensor<4x16xf32>, %arg1: tensor<16x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
            %0 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<4x16xf32>, tensor<16x8xf32>) outs(%arg2 : tensor<4x8xf32>) -> tensor<4x8xf32>
            return %0 : tensor<4x8xf32>
          }
          transform.sequence  failures(propagate) {
          ^bb0(%arg0: !pdl.operation):
            %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!pdl.operation) -> !pdl.operation
            %tiled_linalg_op, %loops:2 = transform.structured.tile %0[2, 3] : (!pdl.operation) -> (!pdl.operation, !transform.op<"scf.for">, !transform.op<"scf.for">)
          }
        }
        """
        )
        check_correct(correct, module)

        run_pipeline_with_repro_report(
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
                tiled = tile_linalg_to_scf_for(m, sizes=[2, 3])

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
                tiled = tile_to_scf_foreach_thread(m, sizes=[2, 3])

        correct = dedent(
            """\
        module {
          func.func @matmul(%arg0: tensor<4x16xf32>, %arg1: tensor<16x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
            %0 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : tensor<4x16xf32>, tensor<16x8xf32>) outs(%arg2 : tensor<4x8xf32>) -> tensor<4x8xf32>
            return %0 : tensor<4x8xf32>
          }
          transform.sequence  failures(propagate) {
          ^bb0(%arg0: !pdl.operation):
            %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!pdl.operation) -> !pdl.operation
            %forall_op, %tiled_op = transform.structured.tile_to_forall_op %0   tile_sizes [2, 3]
          }
        }
        """
        )
        check_correct(correct, module)
        run_pipeline_with_repro_report(
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
              ^bb1(%arg1: !pdl.operation):
                %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
                %1:2 = transform.structured.tile_to_forall_op %0 num_threads [10, 20] (mapping = [ #gpu.thread<y>, #gpu.thread<x> ] )
              }
            }
            """
                )
            )

        run_pipeline_with_repro_report(
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
            ^bb1(%arg1: !pdl.operation):
              %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
              %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
              %2 = transform.structured.vectorize %1  { disable_multi_reduction_to_contract_patterns }
            } 
            """
                )
            )
        run_pipeline_with_repro_report(
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
            ^bb1(%arg1: !pdl.operation):
              %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
              %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
              %2 = transform.structured.vectorize %1
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
