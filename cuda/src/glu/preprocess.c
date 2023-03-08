#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"
#include "nicslu/nicslu.h"
#include "nicslu/nicslu_util.h"

int my_DumpA(SNicsLU *nicslu, double *ax, unsigned int *ai, unsigned int *ap)
{
    uint__t n, nnz;
    double *ax0;
    unsigned int *ai0, *ap0;
    uint__t *rowperm, *pinv, *piv, oldrow, start, end;
    uint__t i, j, p;

    n = nicslu->n;
    nnz = nicslu->nnz;
    ax0 = nicslu->ax;
    ai0 = nicslu->ai;
    ap0 = nicslu->ap;
    rowperm = nicslu->row_perm;/*row_perm[i]=j-->row i in the permuted matrix is row j in the original matrix*/
    pinv = (uint__t *)nicslu->pivot_inv;/*pivot_inv[i]=j-->column i is the jth pivot column*/
    piv = (uint__t *)nicslu->pivot;

    //generate pivot and pivot_inv for function NicsLU_DumpA
    for (i = 0; i < n; ++i)
    {
        pinv[i] = i;
        piv[i] = i;
    }

    // *ax = (real__t *)malloc(sizeof(real__t)*nnz);
    // *ai = (uint__t *)malloc(sizeof(uint__t)*nnz);
    // *ap = (uint__t *)malloc(sizeof(uint__t)*(n+1));

    ap[0] = 0;

    p = 0;
    for (i=0; i<n; ++i)
    {
        oldrow = rowperm[i];
        start = ap0[oldrow];
        end = ap0[oldrow+1];
        ap[i+1] = ap[i] + end - start;

        for (j=start; j<end; ++j)
        {
            ax[p] = ax0[j];
            ai[p++] = pinv[ai0[j]];
        }
    }

    return 0;
}

int preprocess(SNicsLU* nicslu, uint__t n, uint__t nnz, double* ax, unsigned int* ai, unsigned int* ap)
{
    // nicslu = (SNicsLU *)malloc(sizeof(SNicsLU));
    NicsLU_Initialize(nicslu);

    NicsLU_CreateMatrix(nicslu, n, nnz, ax, ai, ap);
    nicslu->cfgi[0] = 1;
    nicslu->cfgf[1] = 0;

    printf("Preprocessing matrix...\n");

    NicsLU_Analyze(nicslu);
    printf("Preprocessing time: %f ms\n", nicslu->stat[0] * 1000);

    my_DumpA(nicslu, ax, ai, ap);
    //rp = nicslu->col_perm;
    //cp = nicslu->row_perm_inv;
    //piv = nicslu->pivot;
    //rows = nicslu->col_scale_perm;
    //cols = nicslu->row_scale;
    //cscale = nicslu->cscale;

    return 0;
}
