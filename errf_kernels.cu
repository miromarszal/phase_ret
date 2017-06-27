#include <cuComplex.h>

__global__ void get_Gj(double *Fj, double *ph, cuDoubleComplex *Gj)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double s, c;
    sincos(ph[idx], &s, &c);
    Gj[idx].x = Fj[idx] * c;
    Gj[idx].y = Fj[idx] * s;
 }

__global__ void mult_complex_1to1(cuDoubleComplex *A, cuDoubleComplex *B, cuDoubleComplex *C)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
    C[idx] = cuCmul(A[idx], B[idx]);
}

__global__ void mult_complex_1ton(cuDoubleComplex *A, cuDoubleComplex *B, cuDoubleComplex *C)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = tid + blockIdx.y * blockDim.x * gridDim.x;
    C[idx] = cuCmul(A[tid], B[idx]);
}

__global__ void get_E_Gwkj(double *Fk, cuDoubleComplex *Gkj, double *E, cuDoubleComplex *Gwkj)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
    double absGkj = cuCabs(Gkj[idx]);
    E[idx] = pow(Fk[idx] - absGkj, 2);
    cuDoubleComplex expGkj = cuCdiv(Gkj[idx], make_cuDoubleComplex(absGkj, 0));
    Gwkj[idx] = cuCsub(cuCmul(make_cuDoubleComplex(Fk[idx], 0), expGkj), Gkj[idx]);
}

__global__ void sum_E(int num, double *E)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double Esum = E[idx];
    for(int s = num-1; s > 0; s--)
        Esum += E[idx + s * blockDim.x * gridDim.x];
    E[idx] = Esum;
}

__global__ void sum_Etot(double *E, double *Ered)
{
    extern __shared__ double share[];
    int tid = threadIdx.x;
    int idx = tid + blockIdx.x * blockDim.x * 2;
    share[tid] = E[idx] + E[idx+blockDim.x];
    __syncthreads();
    for(unsigned int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if(tid < s)
        {
            share[tid] += share[tid+s];
        }
       __syncthreads();
    }
    if(tid == 0)
        Ered[blockIdx.x] = share[0];
}

__global__ void get_dE(int num, cuDoubleComplex *Gj, cuDoubleComplex *Gwjk, double *dE)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double reGwjk = Gwjk[idx].x;
    double imGwjk = -Gwjk[idx].y;
    for(int s = num-1; s > 0; s--)
    {
        reGwjk += Gwjk[idx + s * blockDim.x * gridDim.x].x;
        imGwjk -= Gwjk[idx + s * blockDim.x * gridDim.x].y;
    }
    dE[idx] = (Gj[idx].x * imGwjk + Gj[idx].y * reGwjk) * 2;
}
