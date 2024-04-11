import os
import platform
import shutil
import subprocess
from pathlib import Path

mlir_libs_dir = Path(__file__).parent.parent / "nelli/mlir/_mlir/_mlir_libs"
if llvm_install_dir := os.environ.get("LLVM_INSTALL_DIR"):
    llvm_install_dir = Path(llvm_install_dir).absolute()
else:
    llvm_install_dir = (Path(__file__).parent.parent / "llvm_install").absolute()

if platform.system() == "Darwin":
    shlib_ext = "dylib"
elif platform.system() == "Linux":
    shlib_ext = "so"
else:
    raise NotImplementedError(f"unknown platform {platform.system()}")


shlibs = [
    "LTO",
    "MLIR-C",
    "mlir_async_runtime",
    "mlir_c_runner_utils",
    "mlir_float16_utils",
    "mlir_runner_utils",
    "gomp",
    "iomp5",
    "omp",
    "vulkan-runtime-wrappers",
    "mlir_async_runtime",
]

if platform.system() == "Linux" and platform.processor() == "x86_64":
    shlibs += ["mlir_cuda_runtime"]

for shlib in shlibs:
    shlib_name = f"lib{shlib}.{shlib_ext}"
    assert llvm_install_dir.exists(), f"{llvm_install_dir}"
    llvm_install_fp = (llvm_install_dir / "lib" / shlib_name).absolute()
    if not llvm_install_fp.exists():
        continue
    dst_path = mlir_libs_dir / shlib_name
    shutil.copyfile(llvm_install_fp, dst_path)
    if platform.system() == "Linux":
        shutil.copyfile(llvm_install_fp, f"{dst_path}.17git")
        subprocess.run(
            ["patchelf", "--set-rpath", "$ORIGIN", dst_path],
            check=True,
        )
        subprocess.run(
            ["patchelf", "--set-rpath", "$ORIGIN", f"{dst_path}.17git"],
            check=True,
        )
