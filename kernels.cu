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
    E[idx] = (Fk[idx] - absGkj) * (Fk[idx] - absGkj);
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

__global__ void mult_T(int N, double z, double wl, double *r2, cuDoubleComplex *U)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double s, c;
    cuDoubleComplex u = U[idx];
    double ph = z * sqrt(1 / wl / wl - r2[idx] / N / N) * 2;
    sincospi(ph, &s, &c);
    U[idx].x = u.x * c - u.y * s;
    U[idx].y = u.x * s + u.y * c;
}

__global__ void mult_ph12(int N, double z, double wl, double *r2, cuDoubleComplex *U)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double s, c;
    cuDoubleComplex u = U[idx];
    double ph = 2 * z / wl + r2[idx] / wl / z;
    sincospi(ph, &s, &c);
    U[idx].x = (u.x * s + u.y * c) / N;
    U[idx].y = (-u.x * c + u.y * s) / N;
}

__global__ void mult_ph1(int N, double z, double wl, cuDoubleComplex *U)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double s, c;
    cuDoubleComplex u = U[idx];
    double ph = 2 * z / wl;
    sincospi(ph, &s, &c);
    U[idx].x = -(u.x * s + u.y * c) * N;
    U[idx].y = (u.x * c - u.y * s) * N;
}

__global__ void mult_ph2(int N, double z, double wl, double *r2, cuDoubleComplex *U)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double s, c;
    cuDoubleComplex u = U[idx];
    double ph = r2[idx] / wl / z;
    sincospi(ph, &s, &c);
    U[idx].x = u.x * c - u.y * s;
    U[idx].y = u.x * s + u.y * c;
}

__global__ void fftshift(cuDoubleComplex *U)
{
    int i = threadIdx.x;
    int j = blockIdx.x;
    int idx = i + j * blockDim.x;
    //double a = 1 - 2 * ((i + j) & 1);
    double a = pow(-1.0, (i + j) & 1);
    U[idx].x *= a;
    U[idx].y *= a;
}
