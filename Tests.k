#include <float.h>
#include "Shared.k"

__global__ void sm_average_density(
    const int ncells, const double* Density, const double* Vol, double* ReduceArray)
{
  const int tid = threadIdx.x;
  const int gid = blockIdx.x*blockDim.x+tid;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = 0.0;

  if(gid < ncells) {
    sdata[tid] += Density[gid] * Vol[gid];
  }
  __syncthreads();

  sum_reduce_in_shared<double, NTHREADS>(tid, sdata);
  if (tid == 0) {
    ReduceArray[tid] = sdata[0];
  }
}

__global__ void sm_pressure(
    const int ncells, const double nmatconst, const double* Density, const double* Temperature, const double* Vol, double* Pressure) {

  const int gid = blockIdx.x*blockDim.x+threadIdx.x;

  if(gid >= ncells) {
    return;
  }

  Pressure[gid] = (nmatconst*Density[gid]*Temperature[gid])/Vol[gid];
}

__global__ void cdfm_average_density(
    const int ncells, const int nmats, const double* Density, const double* Vol, double** Densityfrac, double** Volfrac, double* ReduceArray, double* Density_average)
{
  const int tid = threadIdx.x;
  const int gid = blockIdx.x*blockDim.x+tid;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = 0.0;

  if(gid < ncells) {
    for(int m = 0; m < nmats; m++) {
      sdata[tid] += Densityfrac[gid][m] * Volfrac[gid][m];
    }
    Density_average[gid] = sdata[tid] / Vol[gid];
  }
  __syncthreads();

  sum_reduce_in_shared<double, NTHREADS>(tid, sdata);
  if (tid == 0) {
    ReduceArray[tid] = sdata[0];
  }
}

__global__ void cdfm_average_density_with_if(
    const int ncells, const int nmats, const double* Density, const double* Vol, double** Densityfrac, double** Volfrac, double* ReduceArray, double* Density_average)
{
  const int tid = threadIdx.x;
  const int gid = blockIdx.x*blockDim.x+tid;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = 0.0;

  if(gid < ncells) {
    for(int m = 0; m < nmats; m++) {
      if(Volfrac[gid][m] > 0.0) {
        sdata[tid] += Densityfrac[gid][m] * Volfrac[gid][m];
      }
    }
    Density_average[gid] = sdata[tid] / Vol[gid];
  }
  __syncthreads();

  sum_reduce_in_shared<double, NTHREADS>(tid, sdata);
  if (tid == 0) {
    ReduceArray[tid] = sdata[0];
  }
}

__global__ void cdfm_pressure(
    const int ncells, const int nmats, const double* nmatconsts, double** Volfrac, double** Pressurefrac, double** Densityfrac, double** Temperaturefrac)
{
  const int gid = blockIdx.x*blockDim.x+threadIdx.x;

  if(gid >= ncells) {
    return;
  }

  for (int m = 0; m < nmats; m++) {
    if (Volfrac[gid][m] > 0.0) {
      Pressurefrac[gid][m] =
        (nmatconsts[m] * Densityfrac[gid][m] * Temperaturefrac[gid][m]) /
        (Volfrac[gid][m]);
    } else {
      Pressurefrac[gid][m] = 0.0;
    }
  }
}

__global__ void cdfm_average_density_neighbourhood(
    const int ncells, const int nmats, const int* nnbrs, int** nbrs, const double* cen_x, const double* cen_y, double** Volfrac, double** Densityfrac, double** MatDensity_average)
{
  const int gid = blockIdx.x*blockDim.x+threadIdx.x;

  if(gid >= ncells) {
    return;
  }

  double xc[2];
  xc[0] = cen_x[gid];
  xc[1] = cen_y[gid];
  int nn = nnbrs[gid];
  int cnbrs[8];
  double dsqr[8];
  for (int n = 0; n < nn; n++)
    cnbrs[n] = nbrs[gid][n];
  for (int n = 0; n < nn; n++) {
    dsqr[n] = 0.0;
    double ddist = (xc[0] - cen_x[cnbrs[n]]);
    dsqr[n] += ddist * ddist;
  }
  for (int m = 0; m < nmats; m++) {
    if (Volfrac[gid][m] > 0.0) {
      int nnm = 0; // number of nbrs with this material
      for (int n = 0; n < nn; n++) {
        int jc = cnbrs[n];
        if (Volfrac[jc][m] > 0.0) {
          MatDensity_average[gid][m] += Densityfrac[gid][m] / dsqr[n];
          nnm++;
        }
      }
      MatDensity_average[gid][m] /= nnm;
    } else {
      MatDensity_average[gid][m] = 0.0;
    }
  }
}


