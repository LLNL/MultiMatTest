#include <float.h>
#include "Shared.k"

__global__ void sm_average_density(
    const int ncells, const double* Density, const double* Vol, double* ReduceArray)
{
  const int tid = threadIdx.x;
  const int ic = blockIdx.x*blockDim.x+tid;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = 0.0;

  if(ic < ncells) {
    sdata[tid] += Density[ic] * Vol[ic];
  }
  __syncthreads();

  sum_reduce_in_shared<double, NTHREADS>(tid, sdata);
  if (tid == 0) {
    ReduceArray[tid] = sdata[0];
  }
}

__global__ void sm_pressure(
    const int ncells, const double nmatconst, const double* Density, const double* Temperature, const double* Vol, double* Pressure) {

  const int ic = blockIdx.x*blockDim.x+threadIdx.x;

  if(ic >= ncells) {
    return;
  }

  Pressure[ic] = (nmatconst*Density[ic]*Temperature[ic])/Vol[ic];
}

__global__ void cdfm_average_density(
    const int ncells, const int nmats, const double* Density, const double* Vol, double** Densityfrac, double** Volfrac, double* ReduceArray, double* Density_average)
{
  const int tid = threadIdx.x;
  const int ic = blockIdx.x*blockDim.x+tid;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = 0.0;

  if(ic < ncells) {
    for(int m = 0; m < nmats; m++) {
      sdata[tid] += Densityfrac[ic][m] * Volfrac[ic][m];
    }
    Density_average[ic] = sdata[tid] / Vol[ic];
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
  const int ic = blockIdx.x*blockDim.x+tid;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = 0.0;

  if(ic < ncells) {
    for(int m = 0; m < nmats; m++) {
      if(Volfrac[ic][m] > 0.0) {
        sdata[tid] += Densityfrac[ic][m] * Volfrac[ic][m];
      }
    }
    Density_average[ic] = sdata[tid] / Vol[ic];
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
  const int ic = blockIdx.x*blockDim.x+threadIdx.x;

  if(ic >= ncells) {
    return;
  }

  for (int m = 0; m < nmats; m++) {
    if (Volfrac[ic][m] > 0.0) {
      Pressurefrac[ic][m] =
        (nmatconsts[m] * Densityfrac[ic][m] * Temperaturefrac[ic][m]) /
        (Volfrac[ic][m]);
    } else {
      Pressurefrac[ic][m] = 0.0;
    }
  }
}

__global__ void cdfm_average_density_neighbourhood(
    const int ncells, const int nmats, const int* nnbrs, int** nbrs, const double* cen_x, const double* cen_y, double** Volfrac, double** Densityfrac, double** MatDensity_average)
{
  const int ic = blockIdx.x*blockDim.x+threadIdx.x;

  if(ic >= ncells) {
    return;
  }

  double xc[2];
  xc[0] = cen_x[ic];
  xc[1] = cen_y[ic];
  int nn = nnbrs[ic];
  int cnbrs[8];
  double dsqr[8];
  for (int n = 0; n < nn; n++)
    cnbrs[n] = nbrs[ic][n];
  for (int n = 0; n < nn; n++) {
    dsqr[n] = 0.0;
    // TODO: Fairly sure this was meant to iterate over both dimensions??
    double ddist = (xc[0] - cen_x[cnbrs[n]]);
    dsqr[n] += ddist * ddist;
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

__global__ void mdfm_average_density_zero(
    const int ncells, const int nmats, double* Density_average)
{
  const int ic = blockIdx.x*blockDim.x+threadIdx.x;

  if(ic >= ncells) {
    return;
  }

  Density_average[ic] = 0.0;
}

__global__ void mdfm_average_density(
    const int ncells, const int nmats, double* Density_average, double** Densityfrac, double** Volfrac, double* Vol)
{
  const int gid = blockIdx.x*blockDim.x+threadIdx.x;
  const int ic = gid % ncells; 
  const int m = gid / ncells;

  if(gid >= nmats*ncells) {
    return;
  }

  // TODO: Fix the poor race condition
  Density_average[ic] += Densityfrac[m][ic] * Volfrac[m][ic] / Vol[ic];
}

__global__ void mdfm_average_density_with_if(
    const int ncells, const int nmats, double* Density_average, double** Densityfrac, double** Volfrac, double* Vol)
{
  const int gid = blockIdx.x*blockDim.x+threadIdx.x;
  const int ic = gid % ncells; 
  const int m = gid / ncells;
  if(gid >= nmats*ncells) {
    return;
  }

  if (Volfrac[m][ic] > 0.0) {
    Density_average[ic] += Densityfrac[m][ic] * Volfrac[m][ic] / Vol[ic];
  }
}

__global__ void mdfm_pressure(
    const int ncells, const int nmats, const double* nmatconsts, double** Volfrac, double** Pressurefrac, double** Densityfrac, double** Temperaturefrac)
{
  const int gid = blockIdx.x*blockDim.x+threadIdx.x;
  const int ic = gid % ncells; 
  const int m = gid / ncells;

  if(gid >= nmats*ncells) {
    return;
  }

  if (Volfrac[m][ic] > 0.0) {
    Pressurefrac[m][ic] =
      (nmatconsts[m] * Densityfrac[m][ic] * Temperaturefrac[m][ic]) /
      Volfrac[m][ic];
  } else {
    Pressurefrac[m][ic] = 0.0;
  }
}

__global__ void mdfm_average_density_neighbourhood(
    const int ncells, const int nmats, const int* nnbrs, int** nbrs, const double* cen_x, const double* cen_y, double** Volfrac, double** Densityfrac, double** MatDensity_average)
{
  const int gid = blockIdx.x*blockDim.x+threadIdx.x;
  const int ic = gid % ncells; 
  const int m = gid / ncells;

  if(gid >= ncells*nmats) {
    return;
  }

  if (Volfrac[m][ic] > 0.0) {
    double xc[2];
    xc[0] = cen_x[ic];
    xc[1] = cen_y[ic];
    int nn = nnbrs[ic];
    int cnbrs[8];
    double dsqr[8];
    for (int n = 0; n < nn; n++)
      cnbrs[n] = nbrs[ic][n];
    for (int n = 0; n < nn; n++) {
      dsqr[n] = 0.0;

      // TODO: Fairly sure this was meant to iterate over both dimensions??
      double ddist = (xc[0] - cen_x[cnbrs[n]]);
      dsqr[n] += ddist * ddist;
    }

    int nnm = 0; // number of nbrs with this material
    for (int n = 0; n < nn; n++) {
      int jc = cnbrs[n];
      if (Volfrac[m][jc] > 0.0) {
        MatDensity_average[m][ic] += Densityfrac[m][ic] / dsqr[n];
        nnm++;
      }
    }
    MatDensity_average[m][ic] /= nnm;
  } else {
    MatDensity_average[m][ic] = 0.0;
  }
}

__global__ void cdc_average_density(
    const int ncells, const int* imaterial, const int* nextfrac, double* Densityfrac, double* Volfrac, double* Density, double* Vol)
{
  const int ic = blockIdx.x*blockDim.x+threadIdx.x;

  if(ic >= ncells) {
    return;
  }

  double density_ave = 0.0;
  int ix = imaterial[ic];
  if (ix <= 0) { // material numbers for clean cells start at 1
    for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
      density_ave += Densityfrac[ix] * Volfrac[ix];
    }
    // TODO: Surely this is meant to be Density_avarage?
    Density[ic] = density_ave / Vol[ic];
  }
}

__global__ void cdc_average_density_using_nmats(
    const int ncells, const int* imaterial, const int* nmaterials, double* Densityfrac, double* Volfrac, double* Density, double* Vol)
{
  const int ic = blockIdx.x*blockDim.x+threadIdx.x;

  if(ic >= ncells) {
    return;
  }

  double density_ave = 0.0;
  int mstart = imaterial[ic];
  if (mstart <= 0) { // material numbers for clean cells start at 1
    mstart = -mstart;
    for (int ix = 0; ix < nmaterials[ic]; ix++) {
      density_ave += Densityfrac[mstart + ix] * Volfrac[mstart + ix];
    }
    Density[ic] = density_ave / Vol[ic];
  }
}

__global__ void cdc_average_density_with_fractional(
    const int ncells, const int* imaterial, const int* nextfrac, double* Density, double* Densityfrac, double* Volfrac, double* Density_average, double* Vol)
{
  const int ic = blockIdx.x*blockDim.x+threadIdx.x;

  if(ic >= ncells) {
    return;
  }

  double density_ave = 0.0;
  int ix = imaterial[ic];
  if (ix <= 0) { // material numbers for clean cells start at 1
    for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
      density_ave += Densityfrac[ix] * Volfrac[ix];
    }
    Density_average[ic] = density_ave / Vol[ic];
  } else {
    Density_average[ic] = Density[ic];
  }
}

__global__ void cdc_average_density_with_pure(
    const int ncells, const int* imaterial, const int* nextfrac, double* Density, double* Densityfrac, double* Volfrac, double* Density_average, double* Vol)
{
  const int ic = blockIdx.x*blockDim.x+threadIdx.x;

  if(ic >= ncells) {
    return;
  }

  int ix = imaterial[ic];
  double density_ave = 0.0;
  if (ix <= 0) { // material numbers for clean cells start at 1
    for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
      density_ave += Densityfrac[ix] * Volfrac[ix];
    }
  } else { // Pure cell
    density_ave = Density[ic] * Vol[ic];
  }
  Density_average[ic] = density_ave / Vol[ic];
}

__global__ void cdc_pressure(
    const int ncells, const double* nmatconsts, const int* imaterial, const int* nextfrac, const int* imaterialfrac, double* Density, double* Densityfrac, 
    double* Volfrac, double* Pressurefrac, double* Pressure, double* Vol, double* Temperaturefrac, double* Temperature)
{
  const int ic = blockIdx.x*blockDim.x+threadIdx.x;

  if(ic >= ncells) {
    return;
  }

  int ix = imaterial[ic];
  if (ix <= 0) { // material numbers for clean cells start at 1
    for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
      int m = imaterialfrac[ix];
      Pressurefrac[ix] =
        (nmatconsts[m] * Densityfrac[ix] * Temperaturefrac[ix]) /
        Volfrac[ix];
    }
  } else {
    Pressure[ic] = nmatconsts[ix] * Density[ic] * Temperature[ic] / Vol[ic];
  }
}

__global__ void cdc_mat_density_zero(
    const int ncells, const int nmats, double** MatDensity_average)
{
  const int gid = blockIdx.x*blockDim.x+threadIdx.x;
  const int ic = gid % ncells; 
  const int m = gid / ncells;

  if(gid >= ncells*nmats) {
    return;
  }

  MatDensity_average[ic][m] = 0.0;
}

__global__ void cdc_average_mat_density_neighbourhood(
    const int ncells, const int nmats, const int* imaterial, const int* imaterialfrac, const int* nextfrac, const int* nnbrs, int** nbrs, const double* cen_x, const double* cen_y, double* Volfrac, double* Densityfrac, double** MatDensity_average)
{
  const int ic = blockIdx.x*blockDim.x+threadIdx.x;
  if(ic >= ncells) {
    return;
  }

  double xc[2];
  xc[0] = cen_x[ic];
  xc[1] = cen_y[ic];
  int nn = nnbrs[ic];
  int cnbrs[8];
  double dsqr[8];
  for (int n = 0; n < nn; n++)
    cnbrs[n] = nbrs[ic][n];
  for (int n = 0; n < nn; n++) {
    dsqr[n] = 0.0;
    // TODO: This should be over both dimensions?
    double ddist = (xc[0] - cen_x[cnbrs[n]]);
    dsqr[n] += ddist * ddist;
  }

  int ix = imaterial[ic];
  if (ix <= 0) {
    for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
      int m = imaterialfrac[ix];

      int nnm = 0; // number of nbrs with this material
      for (int n = 0; n < nn; n++) {
        int jc = cnbrs[n];

        int jx = imaterial[jc];
        if (jx <= 0) {
          for (jx = -jx; jx >= 0; jx = nextfrac[jx]) {
            if (imaterialfrac[jx] == m) {
              MatDensity_average[ic][m] += Densityfrac[jx] / dsqr[n];
              nnm++;
              break;
            }
          }
        } else {
          if (imaterialfrac[jx] == m) {
            MatDensity_average[ic][m] += Densityfrac[jx] / dsqr[n];
            nnm++;
          }
        }
      }
      MatDensity_average[ic][m] /= nnm;
    }
  } else {
    int m = imaterialfrac[ix];

    int nnm = 0; // number of nbrs with this material
    for (int n = 0; n < nn; n++) {
      int jc = cnbrs[n];

      int jx = imaterial[jc];
      if (jx <= 0) {
        for (jx = -jx; jx >= 0; jx = nextfrac[jx]) {
          if (imaterialfrac[jx] == m) {
            MatDensity_average[ic][m] += Densityfrac[jx] / dsqr[n];
            nnm++;
            break;
          }
        }
      } else {
        if (imaterialfrac[jx] == m) {
          MatDensity_average[ic][m] += Densityfrac[jx] / dsqr[n];
          nnm++;
        }
      }
    }
    MatDensity_average[ic][m] /= nnm;
  }
}

__global__ void mdc_average_density_zero(
    const int ncells, double* Density)
{
  const int C = blockIdx.x*blockDim.x+threadIdx.x;

  if(C >= ncells) {
    return;
  }

  Density[C] = 0.0;
}

__global__ void mdc_average_density(
    const int ncm, const int m, int** subset2mesh, double** Densityfrac,
    double** Volfrac, double* Density)
{
  const int c = blockIdx.x*blockDim.x+threadIdx.x;

  if(c >= ncm) {
    return;
  }

  int C = subset2mesh[m][c];
  Density[C] += Densityfrac[m][c] * Volfrac[m][c];
}

