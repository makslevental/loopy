import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from pip._internal.req import parse_requirements
from setuptools import Extension, setup, find_namespace_packages
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


def get_llvm_url():
    system = platform.system()
    system_suffix = {"Linux": "linux-gnu-ubuntu-22.04", "Darwin": "apple-darwin"}[
        system
    ]
    LIB_ARCH = os.environ.get("LIB_ARCH", platform.machine())
    if LIB_ARCH == "aarch64":
        LIB_ARCH = "arm64"
    # https://github.com/pypa/cibuildwheel/discussions/997#discussioncomment-2037536
    archflags = os.getenv("ARCHFLAGS", "")
    if "arm64" in archflags:
        LIB_ARCH = "arm64"
    LLVM_RELEASE_VERSION = os.environ.get(
        "LLVM_RELEASE_VERSION", "398d68f624d667a17727d346a2139a951a1ebce4"
    )
    assert LLVM_RELEASE_VERSION, "empty LLVM_RELEASE_VERSION"
    # print(f"ARCH {LIB_ARCH}")
    name = f"llvm+mlir+openmp-{sys.version_info.major}.{sys.version_info.minor}-17.0.0-{LIB_ARCH}-{system_suffix}-release"
    url = f"https://github.com/makslevental/llvm-releases/releases/download/llvm-17.0.0-{LLVM_RELEASE_VERSION}/{name}.tar.xz"
    return url


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)  # type: ignore[no-untyped-call]
        ext_build_lib_dir = ext_fullpath.parent.resolve()

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")
        cmake_args = [
            f"-DCMAKE_INSTALL_PREFIX={ext_build_lib_dir}/{PACKAGE_NAME}/mlir",
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DLLVM_ENABLE_LIBEDIT=0",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        llvm_install_dir = os.environ.get("LLVM_INSTALL_DIR", None)
        if llvm_install_dir is not None:
            cmake_args.append(f"-DLLVM_INSTALL_DIR={llvm_install_dir}")
        build_args = []
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if not cmake_generator or cmake_generator == "Ninja":
            try:
                import ninja  # noqa: F401

                ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                cmake_args += [
                    "-GNinja",
                    f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                ]
            except ImportError:
                pass

        if sys.platform.lower().startswith("darwin"):
            cmake_args += ["-DCMAKE_OSX_DEPLOYMENT_TARGET=12"]
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", "--target", "install"] + build_args,
            cwd=build_temp,
            check=True,
        )

        if platform.system() == "Darwin":
            shlib_ext = "dylib"
        elif platform.system() == "Linux":
            shlib_ext = "so"
        else:
            raise NotImplementedError(f"unknown platform {platform.system()}")

        mlir_libs_dir = Path(
            f"{ext_build_lib_dir}/{PACKAGE_NAME}/mlir/_mlir/_mlir_libs"
        )
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
            llvm_install_dir = (Path(".").parent / "llvm_install").absolute()
            assert llvm_install_dir.exists()
            llvm_install_fp = (llvm_install_dir / "lib" / shlib_name).absolute()
            assert llvm_install_fp.exists()
            dst_path = mlir_libs_dir / shlib_name
            shutil.copyfile(llvm_install_fp, dst_path)
            if platform.system() == "Linux":
                shutil.copyfile(llvm_install_fp, f"{dst_path}.17git")
                subprocess.run(
                    ["patchelf", "--set-rpath", "$ORIGIN", dst_path],
                    cwd=build_temp,
                    check=True,
                )
                subprocess.run(
                    ["patchelf", "--set-rpath", "$ORIGIN", f"{dst_path}.17git"],
                    cwd=build_temp,
                    check=True,
                )


PACKAGE_NAME = "nelli"

packages = find_namespace_packages(
    include=[
        PACKAGE_NAME,
        f"{PACKAGE_NAME}.*",
    ],
)

VERSION = "0.0.8"

if len(sys.argv) > 1 and sys.argv[1] == "--version":
    print(VERSION)
elif len(sys.argv) > 1 and sys.argv[1] == "--llvm-url":
    print(get_llvm_url())
else:
    install_reqs = parse_requirements("requirements.txt", session="hack")
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        description="Loop analysis for the polyhedrally challenged.",
        author="Maksim Levental",
        author_email="maksim.levental@gmail.com",
        ext_modules=[CMakeExtension("_mlir")],
        cmdclass={"build_ext": CMakeBuild},
        packages=packages,
        zip_safe=False,
        install_requires=[
            str(ir.requirement)
            for ir in install_reqs
            if "git" not in str(ir.requirement)
        ],
    )
