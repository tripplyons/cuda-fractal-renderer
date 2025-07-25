# ruff: noqa: F401
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np

with open("src/cuda_fractal_renderer/kernel.cu", "r") as f:
    cuda_code = f.read()

source_module = SourceModule(cuda_code)
add_vectors = source_module.get_function("add_vectors")

if __name__ == "__main__":
    a = gpuarray.to_gpu(np.array([1, 2, 3], dtype=np.float32))
    b = gpuarray.to_gpu(np.array([4, 5, 6], dtype=np.float32))
    c = gpuarray.zeros(3, dtype=np.float32)
    n = np.int32(3)
    add_vectors(a, b, c, n, block=(1, 1, 1), grid=(3, 1))
    print(c.get())
