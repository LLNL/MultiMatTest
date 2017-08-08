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
    const int ncells, const int nmats, const double* Density, const double* Vol, double** DensityFrac, double** VolFrac, double* ReduceArray, double* Density_average)
{
  const int tid = threadIdx.x;
  const int gid = blockIdx.x*blockDim.x+tid;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = 0.0;

  if(gid < ncells) {
    for(int m = 0; m < nmats; m++) {
      sdata[tid] += DensityFrac[gid][m] * VolFrac[gid][m];
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
    const int ncells, const int nmats, const double* Density, const double* Vol, double** DensityFrac, double** VolFrac, double* ReduceArray, double* Density_average)
{
  const int tid = threadIdx.x;
  const int gid = blockIdx.x*blockDim.x+tid;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = 0.0;

  if(gid < ncells) {
    for(int m = 0; m < nmats; m++) {
      if(VolFrac[gid][m] > 0.0) {
        sdata[tid] += DensityFrac[gid][m] * VolFrac[gid][m];
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
    const int ncells, const int nmats, const double* nmatconsts, double** VolFrac, double** PressureFrac, double** DensityFrac, double** TemperatureFrac)
{
  const int gid = blockIdx.x*blockDim.x+threadIdx.x;

  if(gid >= ncells) {
    return;
  }

  for (int m = 0; m < nmats; m++) {
    if (VolFrac[gid][m] > 0.0) {
      PressureFrac[gid][m] =
        (nmatconsts[m] * DensityFrac[gid][m] * TemperatureFrac[gid][m]) /
        (VolFrac[gid][m]);
    } else {
      PressureFrac[gid][m] = 0.0;
    }
  }
}

__global__ void cdfm_average_density_neighbourhood(
    const int ncells, const int nmats, const double* Density, const double* Vol, double** DensityFrac, double** VolFrac, double* ReduceArray, double* MatDensity_average)
{
    for (int ic = 0; ic < ncells; ic++) {
      double xc[2];
      xc[0] = cen[ic][0];
      xc[1] = cen[ic][1];
      int nn = nnbrs[ic];
      int cnbrs[8];
      double dsqr[8];
      for (int n = 0; n < nn; n++)
        cnbrs[n] = nbrs[ic][n];
      for (int n = 0; n < nn; n++) {
        dsqr[n] = 0.0;
        for (int d = 0; d < 1; d++) {
          double ddist = (xc[d] - cen[cnbrs[n]][d]);
          dsqr[n] += ddist * ddist;
        }
      }
      for (int m = 0; m < nmats; m++) {
        if (Volfrac[ic][m] > 0.0) {
          int nnm = 0; // number of nbrs with this material
          for (int n = 0; n < nn; n++) {
            int jc = cnbrs[n];
            if (Volfrac[jc][m] > 0.0) {
              MatDensity_average[ic][m] += Densityfrac[ic][m] / dsqr[n];
              nnm++;
            }
          }
          MatDensity_average[ic][m] /= nnm;
        } else {
          MatDensity_average[ic][m] = 0.0;
        }
      }
    }
}


