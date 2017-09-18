#include "Shared.k"
#include <float.h>

__global__ void sm_average_density(const int ncells, const double *Density,
                                   const double *Vol, double *ReduceArray) {
  const int tid = threadIdx.x;
  const int ic = blockIdx.x * blockDim.x + tid;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = 0.0;

  // Coalesced streaming reads and writes
  if (ic < ncells) {
    sdata[tid] += Density[ic] * Vol[ic];
  }
  __syncthreads();

  sum_reduce_in_shared<double, NTHREADS>(tid, sdata);
  if (tid == 0) {
    ReduceArray[tid] = sdata[0];
  }
}

__global__ void sm_pressure(const int ncells, const double nmatconst,
                            const double *Density, const double *Temperature,
                            const double *Vol, double *Pressure) {

  const int ic = blockIdx.x * blockDim.x + threadIdx.x;

  if (ic >= ncells) {
    return;
  }

  // Coalesced streaming reads and writes
  Pressure[ic] = (nmatconst * Density[ic] * Temperature[ic]) / Vol[ic];
}

__global__ void cdfm_average_density(const int ncells, const int nmats,
                                     const double *Density, const double *Vol,
                                     double **Densityfrac, double **Volfrac,
                                     double *ReduceArray,
                                     double *Density_average) {
  const int tid = threadIdx.x;
  const int ic = blockIdx.x * blockDim.x + tid;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = 0.0;

  if (ic < ncells) {
    for (int m = 0; m < nmats; m++) {
      // Uncoalesced reads, local write
      sdata[tid] += Densityfrac[ic][m] * Volfrac[ic][m];
    }
    // Coalesced write
    Density_average[ic] = sdata[tid] / Vol[ic];
  }
  __syncthreads();

  sum_reduce_in_shared<double, NTHREADS>(tid, sdata);
  if (tid == 0) {
    ReduceArray[tid] = sdata[0];
  }
}

__global__ void
cdfm_average_density_with_if(const int ncells, const int nmats,
                             const double *Density, const double *Vol,
                             double **Densityfrac, double **Volfrac,
                             double *ReduceArray, double *Density_average) {
  const int tid = threadIdx.x;
  const int ic = blockIdx.x * blockDim.x + tid;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = 0.0;

  if (ic < ncells) {
    for (int m = 0; m < nmats; m++) {
      // Uncoalesced access to volfrac
      if (Volfrac[ic][m] > 0.0) {
        sdata[tid] += Densityfrac[ic][m] * Volfrac[ic][m];
      }
    }
    // Coalesced read and write
    Density_average[ic] = sdata[tid] / Vol[ic];
  }
  __syncthreads();

  sum_reduce_in_shared<double, NTHREADS>(tid, sdata);
  if (tid == 0) {
    ReduceArray[tid] = sdata[0];
  }
}

__global__ void cdfm_pressure(const int ncells, const int nmats,
                              const double *nmatconsts, double **Volfrac,
                              double **Pressurefrac, double **Densityfrac,
                              double **Temperaturefrac) {
  const int ic = blockIdx.x * blockDim.x + threadIdx.x;

  if (ic >= ncells) {
    return;
  }

  for (int m = 0; m < nmats; m++) {
    // Uncoalesced read
    if (Volfrac[ic][m] > 0.0) {
      Pressurefrac[ic][m] =
          (nmatconsts[m] * Densityfrac[ic][m] * Temperaturefrac[ic][m]) /
          (Volfrac[ic][m]);
    } else {
      // Uncoalesced write
      Pressurefrac[ic][m] = 0.0;
    }
  }
}

__global__ void cdfm_average_density_neighbourhood(
    const int ncells, const int nmats, const int *nnbrs, int **nbrs,
    const double *cen_x, const double *cen_y, double **Volfrac,
    double **Densityfrac, double **MatDensity_average) {
  const int ic = blockIdx.x * blockDim.x + threadIdx.x;

  if (ic >= ncells) {
    return;
  }

  // Coalesced reads
  double xc[2];
  xc[0] = cen_x[ic];
  xc[1] = cen_y[ic];

  // Coalesced read
  int nn = nnbrs[ic];

  int cnbrs[8];
  double dsqr[8];
  for (int n = 0; n < nn; n++) {
    // Uncoalesced read
    cnbrs[n] = nbrs[ic][n];
  }

  for (int n = 0; n < nn; n++) {
    dsqr[n] = 0.0;
    // TODO: Fairly sure this was meant to iterate over both dimensions??
    double ddist = (xc[0] - cen_x[cnbrs[n]]);
    dsqr[n] += ddist * ddist;
  }

  for (int m = 0; m < nmats; m++) {
    // Uncoalesced read
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
      // Uncoalesced write
      MatDensity_average[ic][m] = 0.0;
    }
  }
}

__global__ void mdfm_average_density_zero(const int ncells, const int nmats,
                                          double *Density_average) {
  const int ic = blockIdx.x * blockDim.x + threadIdx.x;

  if (ic >= ncells) {
    return;
  }

  // Coalesced write
  Density_average[ic] = 0.0;
}

__global__ void mdfm_average_density(const int ncells, const int nmats,
                                     const int m, double *Density_average,
                                     double **Densityfrac, double **Volfrac,
                                     double *Vol) {
  const int ic = blockIdx.x * blockDim.x + threadIdx.x;

  if (ic >= ncells) {
    return;
  }

  // Coalesced reads/writes
  Density_average[ic] += Densityfrac[m][ic] * Volfrac[m][ic] / Vol[ic];
}

__global__ void mdfm_average_density_with_if(const int ncells, const int nmats,
                                             const int m,
                                             double *Density_average,
                                             double **Densityfrac,
                                             double **Volfrac, double *Vol) {
  const int ic = blockIdx.x * blockDim.x + threadIdx.x;
  if (ic >= ncells) {
    return;
  }

  // Coalesced reads
  if (Volfrac[m][ic] > 0.0) {

    // Coalesced sparse writes
    Density_average[ic] += Densityfrac[m][ic] * Volfrac[m][ic] / Vol[ic];
  }
}

__global__ void mdfm_pressure(const int ncells, const int nmats, const int m,
                              const double *nmatconsts, double **Volfrac,
                              double **Pressurefrac, double **Densityfrac,
                              double **Temperaturefrac) {
  const int ic = blockIdx.x * blockDim.x + threadIdx.x;
  if (ic >= ncells) {
    return;
  }

  // Coalesced read
  if (Volfrac[m][ic] > 0.0) {

    // Coalesced sparse writes
    Pressurefrac[m][ic] =
        (nmatconsts[m] * Densityfrac[m][ic] * Temperaturefrac[m][ic]) /
        Volfrac[m][ic];
  } else {

    // Coalesced sparse write
    Pressurefrac[m][ic] = 0.0;
  }
}

__global__ void mdfm_average_density_neighbourhood(
    const int ncells, const int nmats, const int m, const int *nnbrs,
    int **nbrs, const double *cen_x, const double *cen_y, double **Volfrac,
    double **Densityfrac, double **MatDensity_average) {
  const int ic = blockIdx.x * blockDim.x + threadIdx.x;
  if (ic >= ncells) {
    return;
  }

  // Coalesced read
  if (Volfrac[m][ic] > 0.0) {

    // Coalesced read
    double xc[2];
    xc[0] = cen_x[ic];
    xc[1] = cen_y[ic];

    // Coalesced read
    int nn = nnbrs[ic];

    int cnbrs[8];
    double dsqr[8];

    for (int n = 0; n < nn; n++) {
      // Uncoalesced read
      cnbrs[n] = nbrs[ic][n];
    }

    for (int n = 0; n < nn; n++) {
      dsqr[n] = 0.0;

      // TODO: Fairly sure this was meant to iterate over both dimensions??
      // Uncoalesced read
      double ddist = (xc[0] - cen_x[cnbrs[n]]);
      dsqr[n] += ddist * ddist;
    }

    int nnm = 0; // number of nbrs with this material
    for (int n = 0; n < nn; n++) {
      int jc = cnbrs[n];
      // Random uncoalesced read
      if (Volfrac[m][jc] > 0.0) {
        // Coalesced read / write
        MatDensity_average[m][ic] += Densityfrac[m][ic] / dsqr[n];
        nnm++;
      }
    }
    // Coalesced write
    MatDensity_average[m][ic] /= nnm;
  } else {
    // Coalesced write
    MatDensity_average[m][ic] = 0.0;
  }
}

__global__ void ccc_average_density(const int ncells, const int *imaterial,
                                    const int *nextfrac, double *Densityfrac,
                                    double *Volfrac, double *Density,
                                    double *Vol) {
  const int ic = blockIdx.x * blockDim.x + threadIdx.x;
  if (ic >= ncells) {
    return;
  }

  // Coalesced read
  int ix = imaterial[ic];

  if (ix <= 0) { // material numbers for clean cells start at 1
    double density_ave = 0.0;
    for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {

      // Uncoalesced reads
      density_ave += Densityfrac[ix] * Volfrac[ix];
    }

    // TODO: Surely this is meant to be Density_avarage?

    // Coalesced writes
    Density[ic] = density_ave / Vol[ic];
  }
}

__global__ void
ccc_average_density_using_nmats(const int ncells, const int *imaterial,
                                const int *nmaterials, double *Densityfrac,
                                double *Volfrac, double *Density, double *Vol) {
  const int ic = blockIdx.x * blockDim.x + threadIdx.x;
  if (ic >= ncells) {
    return;
  }

  double density_ave = 0.0;

  // Coalesced read
  int mstart = imaterial[ic];
  if (mstart <= 0) { // material numbers for clean cells start at 1
    mstart = -mstart;

    // Coalesced read
    for (int ix = 0; ix < nmaterials[ic]; ix++) {

      // Uncoalesced reads
      density_ave += Densityfrac[mstart + ix] * Volfrac[mstart + ix];
    }

    // Coalesced read / write
    Density[ic] = density_ave / Vol[ic];
  }
}

__global__ void
ccc_average_density_with_fractional(const int ncells, const int *imaterial,
                                    const int *nextfrac, double *Density,
                                    double *Densityfrac, double *Volfrac,
                                    double *Density_average, double *Vol) {
  const int ic = blockIdx.x * blockDim.x + threadIdx.x;

  if (ic >= ncells) {
    return;
  }

  double density_ave = 0.0;

  // Coalesced read
  int ix = imaterial[ic];

  if (ix <= 0) { // material numbers for clean cells start at 1
    for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
      // Uncoalesced reads
      density_ave += Densityfrac[ix] * Volfrac[ix];
    }

    // Coalesced read / write
    Density_average[ic] = density_ave / Vol[ic];
  } else {

    // Coalesced read / write
    Density_average[ic] = Density[ic];
  }
}

__global__ void
ccc_average_density_with_pure(const int ncells, const int *imaterial,
                              const int *nextfrac, double *Density,
                              double *Densityfrac, double *Volfrac,
                              double *Density_average, double *Vol) {
  const int ic = blockIdx.x * blockDim.x + threadIdx.x;

  if (ic >= ncells) {
    return;
  }

  // Coalesced read
  int ix = imaterial[ic];

  double density_ave = 0.0;
  if (ix <= 0) { // material numbers for clean cells start at 1
    for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
      // Uncoalesced reads
      density_ave += Densityfrac[ix] * Volfrac[ix];
    }
  } else { // Pure cell

    // Coalesced reads
    density_ave = Density[ic] * Vol[ic];
  }

  // Coalesced read / write
  Density_average[ic] = density_ave / Vol[ic];
}

__global__ void ccc_pressure(const int ncells, const double *nmatconsts,
                             const int *imaterial, const int *nextfrac,
                             const int *imaterialfrac, double *Density,
                             double *Densityfrac, double *Volfrac,
                             double *Pressurefrac, double *Pressure,
                             double *Vol, double *Temperaturefrac,
                             double *Temperature) {
  const int ic = blockIdx.x * blockDim.x + threadIdx.x;

  if (ic >= ncells) {
    return;
  }

  // Coalesced read
  int ix = imaterial[ic];

  if (ix <= 0) { // material numbers for clean cells start at 1

    // Uncoalesced read
    for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {

      // Uncoalesced reads / writes
      int m = imaterialfrac[ix];
      Pressurefrac[ix] =
          (nmatconsts[m] * Densityfrac[ix] * Temperaturefrac[ix]) / Volfrac[ix];
    }
  } else {
    // Coalesced reads / writes, except nmatconsts
    Pressure[ic] = nmatconsts[ix] * Density[ic] * Temperature[ic] / Vol[ic];
  }
}

__global__ void ccc_mat_density_zero(const int ncells, const int nmats,
                                     double *MatDensity_average) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= ncells * nmats) {
    return;
  }

  MatDensity_average[gid] = 0.0;
}

__global__ void ccc_average_mat_density_neighbourhood(
    const int ncells, const int nmats, const int *imaterial,
    const int *imaterialfrac, const int *nextfrac, const int *nnbrs, int **nbrs,
    const double *cen_x, const double *cen_y, double *Volfrac,
    double *Densityfrac, double **MatDensity_average) {

  const int ic = blockIdx.x * blockDim.x + threadIdx.x;
  if (ic >= ncells) {
    return;
  }

  // Coalesced reads
  double xc[2];
  xc[0] = cen_x[ic];
  xc[1] = cen_y[ic];

  // Coalesced read
  int nn = nnbrs[ic];
  int cnbrs[8];
  double dsqr[8];

  // Uncoalesced read
  for (int n = 0; n < nn; n++) {
    cnbrs[n] = nbrs[ic][n];
  }

  for (int n = 0; n < nn; n++) {
    dsqr[n] = 0.0;
    // TODO: This should be over both dimensions?
    // Uncoalesced read
    double ddist = (xc[0] - cen_x[cnbrs[n]]);
    dsqr[n] += ddist * ddist;
  }

  // Coalesced read
  int ix = imaterial[ic];
  if (ix <= 0) {
    for (ix = -ix; ix >= 0; ix = nextfrac[ix]) {
      // Uncoalesced read
      int m = imaterialfrac[ix];

      int nnm = 0; // number of nbrs with this material
      for (int n = 0; n < nn; n++) {
        int jc = cnbrs[n];

        // Uncoalesced read
        int jx = imaterial[jc];
        if (jx <= 0) {
          // Uncoalesced read
          for (jx = -jx; jx >= 0; jx = nextfrac[jx]) {
            // Uncoalesced read
            if (imaterialfrac[jx] == m) {
              // Uncoalesced read / write
              MatDensity_average[ic][m] += Densityfrac[jx] / dsqr[n];
              nnm++;
              break;
            }
          }
        } else {
          if (imaterialfrac[jx] == m) {
            // Uncoalesced read / write
            MatDensity_average[ic][m] += Densityfrac[jx] / dsqr[n];
            nnm++;
          }
        }
      }
      // Uncoalesced read / write
      MatDensity_average[ic][m] /= nnm;
    }
  } else {
    // Uncoalesced read
    int m = imaterialfrac[ix];

    int nnm = 0; // number of nbrs with this material
    for (int n = 0; n < nn; n++) {
      int jc = cnbrs[n];

      // Uncoalesced read
      int jx = imaterial[jc];
      if (jx <= 0) {
        // Uncoalesced read
        for (jx = -jx; jx >= 0; jx = nextfrac[jx]) {
          // Uncoalesced read
          if (imaterialfrac[jx] == m) {
            // Uncoalesced read / write
            MatDensity_average[ic][m] += Densityfrac[jx] / dsqr[n];
            nnm++;
            break;
          }
        }
      } else {
        if (imaterialfrac[jx] == m) {
          // Uncoalesced read / write
          MatDensity_average[ic][m] += Densityfrac[jx] / dsqr[n];
          nnm++;
        }
      }
    }
    // Uncoalesed read / write
    MatDensity_average[ic][m] /= nnm;
  }
}

__global__ void mcc_average_density_zero(const int ncells, double *Density) {
  const int C = blockIdx.x * blockDim.x + threadIdx.x;

  if (C >= ncells) {
    return;
  }

  // Coalesced writes
  Density[C] = 0.0;
}

__global__ void mcc_average_density_fractional(const int ncm, const int m,
                                               int **subset2mesh,
                                               double **Densityfrac,
                                               double **Volfrac,
                                               double *Density) {

  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncm) {
    return;
  }

  int C = subset2mesh[m][c];
  Density[C] += Densityfrac[m][c] * Volfrac[m][c];
}

__global__ void mcc_average_density_by_cell(const int ncells, const int *matids,
                                            int **mesh2subset, int *nmatscell,
                                            double **Densityfrac,
                                            double **Volfrac, double *Density,
                                            double *Vol) {
  const int C = blockIdx.x * blockDim.x + threadIdx.x;

  if (C >= ncells) {
    return;
  }

  double density_ave = 0.0;
  for (int im = 0; im < nmatscell[C]; im++) {
    int m = matids[4 * C + im];
    int c = mesh2subset[m][C];
    density_ave += Densityfrac[m][c] * Volfrac[m][c];
  }
  Density[C] = density_ave / Vol[C];
}

__global__ void mcc_average_density_by_matrix(
    const int ncells, const int *matids, int **mesh2subset, int *nmatscell,
    double **Densityfrac, double **Volfrac, double *Density, double *Vol) {
  const int C = blockIdx.x * blockDim.x + threadIdx.x;

  if (C >= ncells) {
    return;
  }

  double density_ave = 0.0;
  for (int im = 0; im < nmatscell[C]; im++) {
    int m = matids[4 * C + im];
    int c = mesh2subset[m][C];
    density_ave += Densityfrac[m][c] * Volfrac[m][c];
  }
  Density[C] = density_ave / Vol[C];
}

__global__ void
mcc_pressure_by_material(const int ncm, const int m, const double nmatconst,
                         double **Pressurefrac, double **Densityfrac,
                         double **Temperaturefrac, double **Volfrac) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;

  if (c >= ncm) {
    return;
  }

  Pressurefrac[m][c] =
      (nmatconst * Densityfrac[m][c] * Temperaturefrac[m][c]) / Volfrac[m][c];
}

__global__ void mcc_average_density_by_neighbourhood(
    const int ncm, const int m, int **subset2mesh, int **mesh2subset,
    const double *cen_x, const double *cen_y, const int *nnbrs, int **nbrs,
    double **MatDensity_average, double **Densityfrac) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;

  if (c >= ncm) {
    return;
  }

  int C = subset2mesh[m][c];
  double xc[2];
  xc[0] = cen_x[C];
  xc[1] = cen_y[C];
  int nn = nnbrs[C];
  int cnbrs[9];
  double dsqr[8];
  for (int n = 0; n < nn; n++)
    cnbrs[n] = nbrs[C][n];
  for (int n = 0; n < nn; n++) {
    dsqr[n] = 0.0;
    // TODO: Another case where both dimensions were probably intended to be
    // used
    double ddist = (xc[0] - cen_x[cnbrs[n]]);
    dsqr[n] += ddist * ddist;
  }

  int nnm = 0; // number of nbrs with this material
  for (int n = 0; n < nn; n++) {
    int C_j = cnbrs[n];
    int c_j = mesh2subset[m][C_j];
    if (c_j >= 0) {
      MatDensity_average[m][C] += Densityfrac[m][c_j] / dsqr[n];
      nnm++;
    }
  }
  MatDensity_average[m][C] /= nnm;
}

__global__ void mcc_mat_density_zero(const int ncells, const int nmats,
                                     double *MatDensity_average) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= ncells * nmats) {
    return;
  }

  MatDensity_average[gid] = 0.0;
}
