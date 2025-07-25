__global__ void add_vectors(float *a, float *b, float *c, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}
