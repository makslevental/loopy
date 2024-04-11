from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
from scipy import signal

from nelli.mlir.utils import F32, I64, enable_debug
from nelli.mlir._mlir import _mlir_libs
from nelli.mlir._mlir.dialects import linalg
from nelli.mlir.arith import constant
from nelli.mlir.func import mlir_func, declare, call_func, visibility_attr
from nelli.mlir.gpu import block_attr, thread_attr
from nelli.mlir.memref import (
    MemRefValue as MemRef,
    UnrankedMemRefValue as UnrankedMemRef,
    cast,
)
from nelli.mlir.passes import Pipeline
from nelli.mlir.refbackend import (
    LLVMJITBackend,
)
from nelli.mlir.scf import scf_range
from nelli.mlir.tensor import TensorValue as Tensor
from nelli.mlir.transform import (
    sequence,
    match,
    tile_to_scf_forall,
    map_nested_foreach_to_threads,
)
from nelli.mlir.transform.transform import map_foreach_to_blocks
from nelli.utils import mlir_mod_ctx
from nelli.utils import shlib_ext
from test_nns import read_model_ir

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


class TestVulkan:
    backend = LLVMJITBackend(
        shared_libs=[
            str(vulkan_wrapper_library_path),
            str(runner_utils_lib_path),
            str(c_runner_utils_lib_path),
        ]
    )
    golden_src = dedent(
        """\
    module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>} {
      gpu.module @kernels {
        gpu.func @kernel_mul(%arg0 : memref<4x4xf32>, %arg1 : memref<4x4xf32>, %arg2 : memref<4x4xf32>)
          kernel attributes { spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
          %x = gpu.block_id x
          %y = gpu.block_id y
          %1 = memref.load %arg0[%x, %y] : memref<4x4xf32>
          %2 = memref.load %arg1[%x, %y] : memref<4x4xf32>
          %3 = arith.mulf %1, %2 : f32
          memref.store %3, %arg2[%x, %y] : memref<4x4xf32>
          gpu.return
        }
      }

      func.func @main(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>, %arg2: memref<4x4xf32>) {
        %cst1 = arith.constant 1 : index
        %cst4 = arith.constant 4 : index
        gpu.launch_func @kernels::@kernel_mul blocks in (%cst4, %cst4, %cst1) threads in(%cst1, %cst1, %cst1) args(%arg0 : memref<4x4xf32>, %arg1 : memref<4x4xf32>, %arg2 : memref<4x4xf32>)
        %arg6 = memref.cast %arg2 : memref<4x4xf32> to memref<*xf32>
        call @printMemrefF32(%arg6) : (memref<*xf32>) -> ()
        return
      }
      func.func private @printMemrefF32(%ptr : memref<*xf32>)
    }
    """
    )

    def test_basic(self):
        with mlir_mod_ctx(self.golden_src) as module:
            pass

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

    def test_pipeline(self):
        with mlir_mod_ctx(self.golden_src) as module:
            pass

        module = self.backend.compile(
            module,
            Pipeline().gpu_kernel_outlining().fold_memref_alias_ops().canonicalize()
            ##########
            .set_spirv_capabilities(client_api="vulkan")
            .GPU()
            .set_spirv_abi_attrs()
            .UPG()
            ##########
            # .nelli_map_memref_spirv_storage_class(client_api="vulkan")
            # .convert_gpux_to_spirv(map_memory_space=True).canonicalize()
            .convert_gpu_to_spirv()
            ##########
            .SPIRV().spirv_lower_abi_attrs().spirv_update_vce().VRIPS()
            ##########
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
    def linalg(self):
        scale = 1
        M, N, K = 4 * scale, 16 * scale, 8 * scale
        dynamic_2d_memref = MemRef[(-1, -1), F32]
        unranked_memref = UnrankedMemRef[F32]
        with mlir_mod_ctx() as module:
            print_memref_32 = declare("printMemrefF32", [unranked_memref])
            fill_resource_2d_float = declare(
                "fillResource2DFloat", [dynamic_2d_memref, F32]
            )

            @mlir_func(attributes={"visibility": visibility_attr("public")})
            def matmul(
                A: MemRef[(M, N), F32],
                B: MemRef[(N, K), F32],
                C: MemRef[(M, K), F32],
            ):
                zero = constant(0.0, type=F32)
                one = constant(1.0, type=F32)
                two = constant(2.0, type=F32)

                A_cast = cast(A, dynamic_2d_memref.mlir_type)
                B_cast = cast(B, dynamic_2d_memref.mlir_type)
                C_cast = cast(C, dynamic_2d_memref.mlir_type)

                fill_resource_2d_float(A_cast, one)
                fill_resource_2d_float(B_cast, two)
                fill_resource_2d_float(C_cast, zero)

                A_cast = cast(A, unranked_memref.mlir_type)
                B_cast = cast(B, unranked_memref.mlir_type)
                C_cast = cast(C, unranked_memref.mlir_type)

                print_memref_32(A_cast)
                print_memref_32(B_cast)
                print_memref_32(C_cast)

                linalg.matmul(A, B, outs=[C])

                print_memref_32(C_cast)
                # return out

        module = self.backend.compile(
            module,
            kernel_name="matmul",
            pipeline=Pipeline().bufferize().FUNC().convert_linalg_to_parallel_loops()
            # .scf_parallel_loop_tiling(parallel_loop_tile_sizes=(tile_x, tile_y))
            .gpu_map_parallel_loops()
            .convert_parallel_loops_to_gpu()
            .canonicalize()
            .lower_affine()
            .CNUF()
            .gpu_launch_sink_index_computations()
            .gpu_kernel_outlining()
            .fold_memref_alias_ops()
            .canonicalize()
            ##########
            .set_spirv_capabilities(client_api="vulkan")
            .GPU()
            .set_spirv_abi_attrs()
            .UPG()
            ##########
            # .nelli_map_memref_spirv_storage_class(client_api="vulkan")
            .convert_gpux_to_spirv(map_memory_space=True).canonicalize()
            ##########
            .SPIRV().spirv_lower_abi_attrs().spirv_update_vce().VRIPS()
            ##########
            .convert_gpu_launch_to_vulkan_launch()
            .finalize_memref_to_llvm()
            .FUNC()
            .llvm_request_c_wrappers()
            .CNUF()
            .convert_func_to_llvm(index_bitwidth=64)
            .reconcile_unrealized_casts()
            .launch_func_to_vulkan(),
            enable_ir_printing=False,
        )
        # A = np.random.randint(low=0, high=10, size=(M, N)).astype(np.float32)
        # B = np.random.randint(low=0, high=10, size=(N, K)).astype(np.float32)
        # C = np.zeros((M, K)).astype(np.float32)
        A = np.empty((M, N)).astype(np.float32)
        B = np.empty((N, K)).astype(np.float32)
        C = np.empty((M, K)).astype(np.float32)
        invoker = self.backend.load(module)
        invoker.matmul(A, B, C)
        assert np.allclose(A @ B, C)

    @pytest.mark.xfail()
    def _test_conv2d(self):
        tile_x, tile_y = 1, 1
        h = w = 12
        k = 3
        with mlir_mod_ctx() as module:

            @mlir_func
            def conv2d(
                input: MemRef[(h, w), F32],
                kernel: MemRef[(k, k), F32],
                output: MemRef[(h - 2, w - 2), F32],
            ):
                linalg.conv_2d(input, kernel, outs=[output])
                return None

        module = self.backend.compile(
            module,
            kernel_name="conv2d",
            pipeline=Pipeline()
            .bufferize()
            .FUNC()
            .convert_linalg_to_parallel_loops()
            .scf_parallel_loop_tiling(parallel_loop_tile_sizes=(tile_x, tile_y))
            .gpu_map_parallel_loops()
            .convert_parallel_loops_to_gpu()
            .canonicalize()
            .lower_affine()
            .CNUF()
            .gpu_launch_sink_index_computations()
            .gpu_kernel_outlining()
            .fold_memref_alias_ops()
            .canonicalize()
            ##########
            .set_spirv_capabilities(client_api="vulkan")
            .GPU()
            .set_spirv_abi_attrs()
            .UPG()
            ##########
            # .nelli_map_memref_spirv_storage_class(client_api="vulkan")
            .convert_gpux_to_spirv(map_memory_space=True).canonicalize()
            ##########
            .SPIRV().spirv_lower_abi_attrs().spirv_update_vce().VRIPS()
            ##########
            .convert_gpu_launch_to_vulkan_launch()
            .finalize_memref_to_llvm()
            .FUNC()
            .llvm_request_c_wrappers()
            .CNUF()
            .convert_func_to_llvm(index_bitwidth=64)
            .reconcile_unrealized_casts()
            .launch_func_to_vulkan(),
            enable_ir_printing=False,
        )
        invoker = self.backend.load(module)
        input = np.random.randint(low=0, high=10, size=(h, w)).astype(np.float32)
        kernel = np.random.randint(low=0, high=4, size=(3, 3)).astype(np.float32)
        output = np.zeros((h - 2, w - 2)).astype(np.float32)
        invoker.conv2d(input, kernel, output)
        correct = signal.correlate(input.squeeze(), kernel.squeeze(), mode="valid")
        assert np.allclose(output.squeeze(), correct)

    @pytest.mark.xfail()
    def test_conv_2d_nhwc_hwcf(self):
        with mlir_mod_ctx() as module:
            N, H, W = 1, 66, 66
            C_i, C_o = 1, 3
            K = 3

            @mlir_func
            def conv_2d_nhwc_hwcf(
                input: Tensor[(N, C_i, H, W), F32],
                kernel: Tensor[(C_o, C_i, K, K), F32],
                output: Tensor[(N, C_o, H - 2, W - 2), F32],
            ):
                return linalg.conv_2d_nchw_fchw(input, kernel, outs=[output])

            @sequence
            def tile_outer(target, *extra_args):
                m = match(target, ["linalg.conv_2d_nchw_fchw"])
                tiled = tile_to_scf_forall(
                    m,
                    tile_sizes=[0, 1, 8, 8],
                    mapping={
                        0: block_attr("x"),
                        1: block_attr("y"),
                        2: block_attr("z"),
                    },
                )

            @sequence
            def tile_inner(target, *extra_args):
                m = match(target, ["linalg.conv_2d_nchw_fchw"])
                tiled = tile_to_scf_forall(
                    m,
                    tile_sizes=[0, 1, 1, 1],
                    mapping={
                        0: thread_attr("x"),
                        1: thread_attr("y"),
                        2: thread_attr("z"),
                    },
                )

            @sequence
            def map_foreach(target, *extra_args):
                m = match(target, ["func.func"])
                gpu_launch = map_foreach_to_blocks(
                    m, grid_dims=[3, 8, 8], generate_gpu_launch=True
                )
                map_nested_foreach_to_threads(gpu_launch, block_dims=[1, 8, 8])

        with enable_debug():
            module = self.backend.compile(
                module,
                kernel_name="conv_2d_nhwc_hwcf",
                pipeline=Pipeline()
                .transform_dialect_interpreter(debug_transform_root_tag="tile_outer")
                .transform_dialect_interpreter(debug_transform_root_tag="tile_inner")
                .FUNC()
                # .buffer_hoisting()
                # .buffer_loop_hoisting()
                # # .linalg_transform_patterns(decompose_convolutions=True)
                .linalg_transform_patterns(conv2d_im2col=True)
                .empty_tensor_to_alloc_tensor()
                .CNUF()
                .one_shot_bufferize(
                    allow_return_allocs=True, bufferize_function_boundaries=True
                )
                .transform_dialect_interpreter(debug_transform_root_tag="map_foreach")
                .transform_dialect_erase_schedule()
                # ############
                # .FUNC()
                # # .gpu_map_parallel_loops()
                # # .convert_parallel_loops_to_gpu()
                # .canonicalize()
                # .lower_affine()
                # .CNUF()
                # #########
                # .gpu_launch_sink_index_computations()
                # .gpu_kernel_outlining()
                # .fold_memref_alias_ops()
                # .canonicalize()
                # #########
                # .set_spirv_capabilities(client_api="vulkan")
                # .GPU()
                # .set_spirv_abi_attrs()
                # .UPG()
                #########
                # .convert_gpux_to_spirv(map_memory_space=True).canonicalize()
                #########
                # .SPIRV().spirv_lower_abi_attrs().spirv_update_vce().VRIPS()
                #########
                # .convert_gpu_launch_to_vulkan_launch()
                # .bufferize()
                # .finalize_memref_to_llvm()
                # .FUNC()
                # .llvm_request_c_wrappers()
                # .CNUF()
                # .convert_func_to_llvm(index_bitwidth=64)
                # .reconcile_unrealized_casts()
                # .launch_func_to_vulkan()
                ,
                enable_ir_printing=False,
            )
            print(module)

    def test_conv_2d_nhwc_hwcf_parallel(self):
        with mlir_mod_ctx() as module:

            @mlir_func
            def conv_2d_nhwc_hwcf(
                input: Tensor[(1, 225, 225, 3), F32],
                kernel: Tensor[(3, 3, 3, 32), F32],
                output: Tensor[(1, 112, 112, 32), F32],
            ):
                return linalg.conv_2d_nhwc_hwcf(input, kernel, outs=[output])

            @sequence
            def basic(target, *extra_args):
                m = match(target, ["linalg.conv_2d_nhwc_hwcf"])
                tiled = tile_to_scf_forall(m, tile_sizes=[0, 1, 8, 8, 1])

        module = self.backend.compile(
            module,
            kernel_name="conv_2d_nhwc_hwcf",
            pipeline=Pipeline()
            .transform_dialect_interpreter()
            .transform_dialect_erase_schedule()
            .FUNC()
            .linalg_transform_patterns(decompose_convolutions=True)
            .CNUF(),
            enable_ir_printing=False,
        )
        # print(module)

    @pytest.mark.xfail()
    def _test_nn(self):
        scale = 1
        tile_x, tile_y = 1, 1
        M, N, K = 4 * scale, 16 * scale, 8 * scale
        with mlir_mod_ctx(read_model_ir("resnet18")) as module:
            pass

        module = self.backend.compile(
            module,
            kernel_name="forward",
            pipeline=Pipeline()
            .FUNC()
            .refbackend_generalize_tensor_pad()
            .linalg_transform_patterns(generalize_pad_tensor=True)
            .linalg_fuse_elementwise_ops()
            .CNUF()
            .bufferize()
            .FUNC()
            .convert_linalg_to_parallel_loops()
            .scf_parallel_loop_tiling(parallel_loop_tile_sizes=(tile_x, tile_y))
            .gpu_map_parallel_loops()
            .convert_parallel_loops_to_gpu()
            .canonicalize()
            .lower_affine()
            .CNUF()
            .gpu_launch_sink_index_computations()
            .gpu_kernel_outlining()
            .fold_memref_alias_ops()
            .canonicalize()
            ##########
            .set_spirv_capabilities(client_api="vulkan")
            .GPU()
            .set_spirv_abi_attrs()
            .UPG()
            # ##########
            # # .nelli_map_memref_spirv_storage_class(client_api="vulkan")
            .convert_gpux_to_spirv(map_memory_space=True).canonicalize()
            # ##########
            .SPIRV().spirv_lower_abi_attrs().spirv_update_vce().VRIPS()
            # ##########
            .convert_gpu_launch_to_vulkan_launch()
            .finalize_memref_to_llvm()
            .FUNC()
            .llvm_request_c_wrappers()
            .CNUF()
            .convert_func_to_llvm(index_bitwidth=64)
            .reconcile_unrealized_casts()
            .launch_func_to_vulkan(),
            enable_ir_printing=False,
        )
        A = np.random.randint(low=0, high=10, size=(M, N)).astype(np.float32)
        B = np.random.randint(low=0, high=10, size=(N, K)).astype(np.float32)
        C = np.zeros((M, K)).astype(np.float32)
        invoker = self.backend.load(module)
        invoker.matmul(A, B, C)
        assert np.allclose(A @ B, C)

    def test_benchmark(self):
        N, C, H, W = 1, 3, 32, 32
        N_RUNS = 10
        param1_type = Tensor[[N, C, H, W], F32]
        res_type = Tensor[[1, 1000], F32].mlir_type

        with mlir_mod_ctx(read_model_ir("resnet18")) as module:
            timer = declare("_mlir_ciface_nanoTime", [], result_annots=[I64])

            @mlir_func(range_ctor=scf_range, attributes={"llvm.emit_c_interface": None})
            def timing_wrapper(
                x: param1_type,
                times: MemRef[[N_RUNS], I64],
            ):
                for i in range(0, N_RUNS):
                    start = timer()
                    out = call_func("forward", [x], [res_type])
                    end = timer()
                    times[i] = end - start

        module = self.backend.compile(
            module,
            Pipeline().sparse_compiler(
                parallelization_strategy="dense-outer-loop",
                vl=16,
                reassociate_fp_reductions=True,
                enable_index_optimizations=True,
            ),
        )

        invoker = self.backend.load(module, opt_level=1)
        A = np.random.randint(low=0, high=10, size=(N, C, H, W)).astype(np.float32)
        times = np.zeros(N_RUNS).astype(np.int64)
        invoker.timing_wrapper(A, times)
        print(times)
        print("avg time", times.mean() / 1e9)
