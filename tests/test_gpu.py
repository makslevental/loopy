import platform
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from nelli.mlir import spirv
from nelli.mlir._mlir import _mlir_libs
from nelli.mlir.arith import constant
from nelli.mlir.func import declare, mlir_func
from nelli.mlir.gpu import (
    Module as GPUModule,
    block_id_x,
    block_id_y,
    set_container_module,
    host_register,
    gpu_launch,
    all_reduce,
)
from nelli.mlir.memref import (
    MemRefValue as MemRef,
    UnrankedMemRefValue as UnrankedMemRef,
    cast,
)
from nelli.mlir.passes import Pipeline
from nelli.mlir.refbackend import LLVMJITBackend
from nelli.mlir.spirv import set_module_target_env
from nelli.mlir.utils import F32, I32
from nelli.utils import shlib_ext, mlir_mod_ctx
from util import check_correct

c_runner_utils_lib_path = (
    Path(_mlir_libs.__file__).parent / f"libmlir_c_runner_utils.{shlib_ext()}"
)
assert c_runner_utils_lib_path.exists()
runner_utils_lib_path = (
    Path(_mlir_libs.__file__).parent / f"libmlir_runner_utils.{shlib_ext()}"
)
assert runner_utils_lib_path.exists()

vulkan_wrapper_library_path = (
    Path(_mlir_libs.__file__).parent / f"libvulkan-runtime-wrappers.{shlib_ext()}"
)
assert vulkan_wrapper_library_path.exists()

shared_libs = [
    str(vulkan_wrapper_library_path),
    str(runner_utils_lib_path),
    str(c_runner_utils_lib_path),
]

if platform.system() == "Linux" and platform.processor() == "x86_64":
    cuda_wrapper_library_path = (
        Path(_mlir_libs.__file__).parent / f"libmlir_cuda_runtime.{shlib_ext()}"
    )
    assert cuda_wrapper_library_path.exists()
    shared_libs.append(str(cuda_wrapper_library_path))


class TestGPU:
    backend = LLVMJITBackend(shared_libs=shared_libs)

    def test_nested_func(self):
        scale = 1
        M, N, K = 4 * scale, 16 * scale, 8 * scale

        with mlir_mod_ctx() as module:

            class MyClass1(GPUModule):
                def mat_product_kernel(
                    self,
                    A: MemRef[(M, N), F32],
                    B: MemRef[(N, K), F32],
                    C: MemRef[(M, K), F32],
                ):
                    x = block_id_x()
                    y = block_id_y()
                    a = A[x, y]
                    b = B[x, y]
                    C[x, y] = a * b
                    return

            MyClass1(
                func_attributes={
                    "spirv.entry_point_abi": spirv.entry_point_abi(
                        workgroup_size=[1, 1, 1]
                    )
                }
            )

        correct = dedent(
            """\
        module {
          gpu.module @MyClass1 {
            gpu.func @method(%arg0: memref<4x16xf32>, %arg1: memref<16x8xf32>, %arg2: memref<4x8xf32>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
              %0 = gpu.block_id  x
              %1 = gpu.block_id  y
              %2 = memref.load %arg0[%0, %1] : memref<4x16xf32>
              %3 = memref.load %arg1[%0, %1] : memref<16x8xf32>
              %4 = arith.mulf %2, %3 : f32
              memref.store %4, %arg2[%0, %1] : memref<4x8xf32>
              gpu.return
            }
          }
        }
        """
        )
        # print(MyClass1)
        check_correct(correct, module)

    def test_calling_nested_func(self):
        scale = 1
        M, N, K = 4 * scale, 16 * scale, 8 * scale

        with mlir_mod_ctx() as module:

            class MyClass1(GPUModule):
                def kernel(
                    self,
                    A: MemRef[(M, N), F32],
                    B: MemRef[(N, K), F32],
                    C: MemRef[(M, K), F32],
                ):
                    x = block_id_x()
                    y = block_id_y()
                    a = A[x, y]
                    b = B[x, y]
                    C[x, y] = a * b
                    return

            m = MyClass1(
                func_attributes={
                    "spirv.entry_point_abi": spirv.entry_point_abi(
                        workgroup_size=[1, 1, 1]
                    ),
                }
            )

            @mlir_func
            def main(
                A: MemRef[(M, N), F32],
                B: MemRef[(N, K), F32],
                C: MemRef[(M, K), F32],
            ):
                m.kernel(A, B, C, grid_size=[4, 4, 1], block_size=[1, 1, 1])

        module = set_container_module(module)
        module = self.backend.compile(module, Pipeline().canonicalize())

        correct = dedent(
            """\
        module attributes {gpu.container_module} {
          gpu.module @MyClass1 {
            gpu.func @kernel(%arg0: memref<4x16xf32>, %arg1: memref<16x8xf32>, %arg2: memref<4x8xf32>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
              %0 = gpu.block_id  x
              %1 = gpu.block_id  y
              %2 = memref.load %arg0[%0, %1] : memref<4x16xf32>
              %3 = memref.load %arg1[%0, %1] : memref<16x8xf32>
              %4 = arith.mulf %2, %3 : f32
              memref.store %4, %arg2[%0, %1] : memref<4x8xf32>
              gpu.return
            }
          }
          func.func @main(%arg0: memref<4x16xf32>, %arg1: memref<16x8xf32>, %arg2: memref<4x8xf32>) {
            %c4 = arith.constant 4 : index
            %c1 = arith.constant 1 : index
            %0 = gpu.launch_func async @MyClass1::@kernel blocks in (%c4, %c4, %c1) threads in (%c1, %c1, %c1) args(%arg0 : memref<4x16xf32>, %arg1 : memref<16x8xf32>, %arg2 : memref<4x8xf32>)
            return
          }
        }
        """
        )
        check_correct(correct, module)

    @pytest.mark.xfail()
    def _test_calling_nested_func_run(self):
        # M, N, K = 4 * scale, 16 * scale, 8 * scale
        M, N, K = 4, 4, 4
        unranked_memref = UnrankedMemRef[F32]

        with mlir_mod_ctx() as module:
            print_memref_32 = declare("printMemrefF32", [unranked_memref])

            class MyClass2(GPUModule):

                # wtf if you name this `kernel` it doesn't work?
                def matmul(
                    self,
                    A: MemRef[(M, N), F32],
                    B: MemRef[(N, K), F32],
                    C: MemRef[(M, K), F32],
                ):
                    x = block_id_x()
                    y = block_id_y()
                    a = A[x, y]
                    b = B[x, y]
                    C[x, y] = a * b
                    return

            m = MyClass2(
                func_attributes={
                    "spirv.entry_point_abi": spirv.entry_point_abi(
                        workgroup_size=[1, 1, 1]
                    ),
                }
            )

            @mlir_func
            def main(
                A: MemRef[(M, N), F32],
                B: MemRef[(N, K), F32],
                C: MemRef[(M, K), F32],
            ):
                m.matmul(A, B, C, grid_size=[4, 4, 1], block_size=[1, 1, 1])
                C_cast = cast(C, unranked_memref.mlir_type)
                print_memref_32(C_cast)

        module = set_container_module(module)
        module = set_module_target_env(module)

        module = self.backend.compile(
            module,
            Pipeline()
            .gpu_kernel_outlining()
            .fold_memref_alias_ops()
            .canonicalize()
            .convert_gpu_to_spirv()
            .canonicalize()
            .SPIRV()
            .spirv_lower_abi_attrs()
            .spirv_update_vce()
            .VRIPS()
            .convert_gpu_launch_to_vulkan_launch()
            .finalize_memref_to_llvm()
            .FUNC()
            .llvm_request_c_wrappers()
            .CNUF()
            .convert_func_to_llvm(index_bitwidth=64)
            .reconcile_unrealized_casts()
            .launch_func_to_vulkan(),
        )

        A = np.random.randint(low=0, high=10, size=(4, 4)).astype(np.float32)
        B = np.random.randint(low=0, high=10, size=(4, 4)).astype(np.float32)
        C = np.zeros((4, 4)).astype(np.float32)
        invoker = self.backend.load(module)
        invoker.main(A, B, C)
        assert np.allclose(A * B, C)

    @pytest.mark.xfail()
    def test_cuda(self, capfd):
        src = dedent(
            """\
        func.func @main() {
          %data = memref.alloc() : memref<2x6xi32>
          %sum = memref.alloc() : memref<2xi32>
          %cst0 = arith.constant 0 : i32
          %cst1 = arith.constant 1 : i32
          %cst2 = arith.constant 2 : i32
          %cst4 = arith.constant 4 : i32
          %cst8 = arith.constant 8 : i32
          %cst16 = arith.constant 16 : i32

          %cst3 = arith.constant 3 : i32
          %cst6 = arith.constant 6 : i32
          %cst7 = arith.constant 7 : i32
          %cst10 = arith.constant 10 : i32
          %cst11 = arith.constant 11 : i32

          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %c3 = arith.constant 3 : index
          %c4 = arith.constant 4 : index
          %c5 = arith.constant 5 : index
          %c6 = arith.constant 6 : index

          %cast_data = memref.cast %data : memref<2x6xi32> to memref<*xi32>
          gpu.host_register %cast_data : memref<*xi32>
          %cast_sum = memref.cast %sum : memref<2xi32> to memref<*xi32>
          gpu.host_register %cast_sum : memref<*xi32>

          memref.store %cst0, %data[%c0, %c0] : memref<2x6xi32>
          memref.store %cst1, %data[%c0, %c1] : memref<2x6xi32>
          memref.store %cst2, %data[%c0, %c2] : memref<2x6xi32>
          memref.store %cst4, %data[%c0, %c3] : memref<2x6xi32>
          memref.store %cst8, %data[%c0, %c4] : memref<2x6xi32>
          memref.store %cst16, %data[%c0, %c5] : memref<2x6xi32>

          memref.store %cst2, %data[%c1, %c0] : memref<2x6xi32>
          memref.store %cst3, %data[%c1, %c1] : memref<2x6xi32>
          memref.store %cst6, %data[%c1, %c2] : memref<2x6xi32>
          memref.store %cst7, %data[%c1, %c3] : memref<2x6xi32>
          memref.store %cst10, %data[%c1, %c4] : memref<2x6xi32>
          memref.store %cst11, %data[%c1, %c5] : memref<2x6xi32>

          gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c2, %grid_y = %c1, %grid_z = %c1)
                     threads(%tx, %ty, %tz) in (%block_x = %c6, %block_y = %c1, %block_z = %c1) {
            %val = memref.load %data[%bx, %tx] : memref<2x6xi32>
            %reduced = gpu.all_reduce and %val uniform {} : (i32) -> (i32)
            memref.store %reduced, %sum[%bx] : memref<2xi32>
            gpu.terminator
          }

          call @printMemrefI32(%cast_sum) : (memref<*xi32>) -> ()

          return
        }

        func.func private @printMemrefI32(memref<*xi32>)
        """
        )

        with mlir_mod_ctx(src) as module:
            pass

        module = self.backend.compile(
            module,
            Pipeline()
            .gpu_kernel_outlining()
            .GPU()
            .strip_debuginfo()
            .convert_gpu_to_nvvm()
            .gpu_to_cubin()
            .UPG()
            .gpu_to_llvm(),
        )

        invoker = self.backend.load(module)
        invoker.main()

        correct = dedent(
            """\
        Unranked Memref base@ = 0x5612f9fb6570 rank = 1 offset = 0 sizes = [2] strides = [1] data =
        [0,  2]
        """
        )
        out, err = capfd.readouterr()
        check_correct(correct, out)

    @pytest.mark.xfail()
    def test_cuda_sugar(self, capfd):
        unranked_memref = UnrankedMemRef[I32]
        with mlir_mod_ctx() as module:
            print_memref_i32 = declare("printMemrefI32", [unranked_memref])

            @mlir_func(rewrite_ast_=False, rewrite_bytecode_=False)
            def main():
                data = MemRef.alloc((2, 6), I32)
                sum = MemRef.alloc((2,), I32)

                power_csts = [constant(0, type=I32)] + [
                    constant(2 ** i, type=I32) for i in range(5)
                ]
                odd_csts = [
                    constant(3, type=I32),
                    constant(6, type=I32),
                    constant(7, type=I32),
                    constant(10, type=I32),
                    constant(11, type=I32),
                ]
                cast_data = cast(data, unranked_memref.mlir_type)
                host_register(cast_data)
                cast_sum = cast(sum, unranked_memref.mlir_type)
                host_register(cast_sum)

                for i in range(6):
                    data[0, i] = power_csts[i]

                data[1, 0] = power_csts[2]
                for i in range(0, 5):
                    data[1, i + 1] = odd_csts[i]

                @gpu_launch(grid_size=[2, 1, 1], block_size=[6, 1, 1])
                def kernel(block_ids, thread_ids):
                    bx, by, bz = block_ids
                    tx, ty, tz = thread_ids

                    val = data[bx, tx]
                    reduced = all_reduce("and", val, uniform=True)
                    sum[bx] = reduced

                print_memref_i32(cast_sum)

        module = self.backend.compile(
            module,
            Pipeline()
            .gpu_kernel_outlining()
            .GPU()
            .strip_debuginfo()
            .convert_gpu_to_nvvm()
            .gpu_to_cubin()
            .UPG()
            .gpu_to_llvm(),
        )

        invoker = self.backend.load(module)
        invoker.main()

        correct = dedent(
            """\
        Unranked Memref base@ = 0x5612f9fb6570 rank = 1 offset = 0 sizes = [2] strides = [1] data =
        [0,  2]
        """
        )
        out, err = capfd.readouterr()
        check_correct(correct, out)

    @pytest.mark.xfail()
    def test_wmma_matmul_f16(self, capfd):
        src = dedent(
            """\
        func.func @main() {
          %0 = memref.alloc() : memref<16x16xf16>
          %22 = memref.alloc() : memref<16x16xf16>
          %1 = memref.alloc() : memref<16x16xf32>

          %f1 = arith.constant 1.0e+00 : f16
          %f0 = arith.constant 0.0e+00 : f16
          %c0 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c32 = arith.constant 32 : index
          %c1 = arith.constant 1 : index

          // Intialize the Input matrix with the column index in each row.
          scf.for %arg0 = %c0 to %c16 step %c1 {
            scf.for %arg1 = %c0 to %c16 step %c1 {
              %2 = arith.index_cast %arg1 : index to i16
              %3 = arith.sitofp %2 : i16 to f16
              memref.store %3, %0[%arg0, %arg1] : memref<16x16xf16>
            }
          }
          // Intialize the accumulator matrix with zeros.
          scf.for %arg0 = %c0 to %c16 step %c1 {
            scf.for %arg1 = %c0 to %c16 step %c1 {
              memref.store %f0, %22[%arg0, %arg1] : memref<16x16xf16>
            }
          }

          %2 = memref.cast %0 : memref<16x16xf16> to memref<*xf16>
          %33 = memref.cast %22 : memref<16x16xf16> to memref<*xf16>
          %3 = memref.cast %1 : memref<16x16xf32> to memref<*xf32>
          gpu.host_register %2 : memref<*xf16>
          gpu.host_register %33 : memref<*xf16>

          gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
                     threads(%tx, %ty, %tz) in (%block_x = %c32, %block_y = %c1, %block_z = %c1) {
            %A = gpu.subgroup_mma_load_matrix %0[%c0, %c0] {leadDimension = 16 : index, transpose} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %B = gpu.subgroup_mma_load_matrix %0[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %C = gpu.subgroup_mma_load_matrix %22[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "COp">

            %R = gpu.subgroup_mma_compute %A, %B, %C {a_transpose} : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">

            gpu.subgroup_mma_store_matrix %R, %0[%c0, %c0] {leadDimension = 16 : index}: !gpu.mma_matrix<16x16xf16, "COp">, memref<16x16xf16>
            gpu.terminator
          }

          // Convert the results from f16 to f32 for printing.
          scf.for %arg0 = %c0 to %c16 step %c1 {
            scf.for %arg1 = %c0 to %c16 step %c1 {
              %6 = memref.load %0[%arg0, %arg1] : memref<16x16xf16>
              %7 = arith.extf %6 : f16 to f32
              memref.store %7, %1[%arg0, %arg1] : memref<16x16xf32>
            }
          }

          // Print the memref after computation.
          call @printMemrefF32(%3) : (memref<*xf32>) -> ()
          return
        }

        func.func private @printMemrefF32(memref<*xf32>)
        """
        )
        with mlir_mod_ctx(src) as module:
            pass

        module = self.backend.compile(
            module,
            Pipeline()
            .gpu_kernel_outlining()
            .GPU()
            .strip_debuginfo()
            .convert_gpu_to_nvvm()
            .gpu_to_cubin(chip="sm_70")
            .UPG()
            .convert_scf_to_cf()
            .gpu_to_llvm(),
        )

        invoker = self.backend.load(module)
        invoker.main()

        dedent(
            """\
        Unranked Memref base@ = 0x55f3b214e760 rank = 2 offset = 0 sizes = [16, 16] strides = [16, 1] data =
        [[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
         [0,   16,   32,   48,   64,   80,   96,   112,   128,   144,   160,   176,   192,   208,   224,   240],
         [0,   32,   64,   96,   128,   160,   192,   224,   256,   288,   320,   352,   384,   416,   448,   480],
         [0,   48,   96,   144,   192,   240,   288,   336,   384,   432,   480,   528,   576,   624,   672,   720],
         [0,   64,   128,   192,   256,   320,   384,   448,   512,   576,   640,   704,   768,   832,   896,   960],
         [0,   80,   160,   240,   320,   400,   480,   560,   640,   720,   800,   880,   960,   1040,   1120,   1200],
         [0,   96,   192,   288,   384,   480,   576,   672,   768,   864,   960,   1056,   1152,   1248,   1344,   1440],
         [0,   112,   224,   336,   448,   560,   672,   784,   896,   1008,   1120,   1232,   1344,   1456,   1568,   1680],
         [0,   128,   256,   384,   512,   640,   768,   896,   1024,   1152,   1280,   1408,   1536,   1664,   1792,   1920],
         [0,   144,   288,   432,   576,   720,   864,   1008,   1152,   1296,   1440,   1584,   1728,   1872,   2016,   2160],
         [0,   160,   320,   480,   640,   800,   960,   1120,   1280,   1440,   1600,   1760,   1920,   2080,   2240,   2400],
         [0,   176,   352,   528,   704,   880,   1056,   1232,   1408,   1584,   1760,   1936,   2112,   2288,   2464,   2640],
         [0,   192,   384,   576,   768,   960,   1152,   1344,   1536,   1728,   1920,   2112,   2304,   2496,   2688,   2880],
         [0,   208,   416,   624,   832,   1040,   1248,   1456,   1664,   1872,   2080,   2288,   2496,   2704,   2912,   3120],
         [0,   224,   448,   672,   896,   1120,   1344,   1568,   1792,   2016,   2240,   2464,   2688,   2912,   3136,   3360],
         [0,   240,   480,   720,   960,   1200,   1440,   1680,   1920,   2160,   2400,   2640,   2880,   3120,   3360,   3600]]
        """
        )
        out, err = capfd.readouterr()
        # check_correct(correct, out)
