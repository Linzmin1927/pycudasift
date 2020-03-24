# -*- coding: utf8 -*-

import Cython.Build
import numpy as np
import os
import setuptools
from Cython.Distutils import build_ext
from setuptools import Extension
from distutils.errors import DistutilsFileError

class BuildExt(build_ext):
    user_options = [ 
        ("cuda-version=", None, "Version # for CUDA libraries (default = 7.5)"),
        ("cuda-prefix=", None, 
         "Prefix for the CUDA install directory (default = /usr/local/cuda)"),
        ("cuda-gcc=", None, "Version of GCC to use in nvcc (default=gcc)"),
        ("cuda-arch=", None, "Cuda device architecture (default=sm_52)")] +\
        build_ext.user_options
    def initialize_options(self):
        build_ext.initialize_options(self)
        self.cuda_version = None
        self.cuda_prefix = None
        self.cuda_gcc = None
        self.cuda_arch = None
        
    def finalize_options(self):
        build_ext.finalize_options(self)
        if self.cuda_prefix is None:
            self.cuda_prefix = os.environ.get("CUDA_PREFIX", "/usr/local/cuda")
        if self.cuda_version is None:
            self.cuda_version = os.environ.get("CUDA_VERSION", "75")
        if self.cuda_arch is None:
            self.cuda_arch = os.environ.get("CUDA_ARCH", "sm_52")
        if self.cuda_gcc is None and "CUDA_GCC" in os.environ:
            self.cuda_gcc = os.environ["CUDA_GCC"]
    
    def build_extensions(self):
        self.ensure_finalized()
        #
        # In true distutils-followons fashion, we monkey patch the compiler
        # to use nvcc
        #
        self.compiler.src_extensions.append(".cu")
        self._old_compile = self.compiler._compile
        self._old_compiler_so = self.compiler.compiler_so
        self.compiler._compile = self._cuda_compile
        include_dirs = [
            np.get_include(),
            os.path.dirname(os.path.abspath(__file__)),
            os.path.join(self.cuda_prefix, "include")
        ]
        for extension in self.extensions:
            if extension.include_dirs is None:
                extension.include_dirs = []
            for include_dir in include_dirs:
                if include_dir not in extension.include_dirs:
                    extension.include_dirs.append(include_dir)
            extension.extra_compile_args = ["-g", "--verbose"]
            if extension.extra_link_args is None:
                extension.extra_link_args = []
            extension.extra_link_args += [
                "-L%s" % os.path.join(self.cuda_prefix, "lib64"),
                "-lcudart"]
        build_ext.build_extensions(self)
    
    def _cuda_compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        '''The replacement compiler for building "cu" files.'''
        if os.path.splitext(src)[1] != ".cu":
            return self._old_compile(obj, src, ext, cc_args, extra_postargs, 
                                    pp_opts)

        postargs = [ "-O3", "-lineinfo", "-Xptxas", "-dlcm=ca", "--verbose",
                     "-m64", "-Xcompiler=-fPIC"]
        postargs.append("-arch=%s" % self.cuda_arch)
        if self.cuda_gcc is not None:
            postargs.append("-ccbin=%s" % self.cuda_gcc)
        self.compiler.set_executable(
            'compiler_so', 
            os.path.join(self.cuda_prefix, "bin", "nvcc"))
        try:
            return self._old_compile(obj, src, ext, cc_args, postargs, pp_opts)
        finally:
            self.compiler.compiler_so = self._old_compiler_so

setuptools.setup(
    author = "Mårten Björkman",
    cmdclass = dict(build_ext=BuildExt),
    ext_modules = [
        # TODO - add geomFuncs.cpp after replacing OpenCV's cholesky
        # decomposition with Numpy's
        Extension(name = "cudasift._cudasift",
                  language = "c++",
                  sources = [
                      os.path.join("cudasift", "_cudasift.pyx"),
                      "cudaImage.cu", "cudaSiftH.cu", "matching.cu"])
        ],
    description="Bindings for CudaSift library",
    name="cudasift",
    packages=["cudasift"],
    version="0.1.2")
