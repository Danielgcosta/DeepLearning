/**************
 * lusystem.c *
 **************/

#include <stdlib.h>
#include "lusystem.h"

namespace MSA
{

int LuDcmp(double **a, int n, int *indx)
{
  int i,imax,j,k;
  double big,dum,sum,temp;
  double *vv;

  imax = 0;

  vv=(double *)malloc(n*sizeof(double));
  for (i=0;i<n;i++)
  {
    big=0.0;
    for (j=0;j<n;j++)
      if ((temp=ABS(a[i][j])) > big) big=temp;
    if (big == 0.0) return 0;
    vv[i]=1.0/big;
  }
  for (j=0;j<n;j++)
  {
    for (i=0;i<j;i++)
    {
      sum=a[i][j];
      for (k=0;k<i;k++) sum -= a[i][k]*a[k][j];
      a[i][j]=sum;
    }
    big=0.0;
    for (i=j;i<n;i++)
    {
      sum=a[i][j];
      for (k=0;k<j;k++)
        sum -= a[i][k]*a[k][j];
      a[i][j]=sum;
      if ( (dum=vv[i]*ABS(sum)) >= big)
      {
        big=dum;
        imax=i;
      }
    }
    if (j != imax)
    {
      for (k=0;k<n;k++)
        SWAP(a[imax][k], a[j][k])
        vv[imax]=vv[j];
    }
    indx[j]=imax;
    if (a[j][j] == 0.0) a[j][j]=1.0e-20;
    if (j != n-1)
    {
      dum=1.0/(a[j][j]);
      for (i=j+1;i<n;i++) a[i][j] *= dum;
    }
  }
  free(vv);
  return 1;
}





void LuBksb(double **a, int n, int *indx, double *b)
{
  int i,ii=0,ip,j;
  double sum;

  for (i=0;i<n;i++)
  {
    ip=indx[i];
    sum=b[ip];
    b[ip]=b[i];
    if (ii) for (j=ii-1;j<i;j++) sum -= a[i][j]*b[j];
    else if (sum) ii=i+1;
    b[i]=sum;
  }
  for (i=n-1;i>=0;i--)
  {
    sum=b[i];
    for (j=i+1;j<n;j++) sum -= a[i][j]*b[j];
    b[i]=sum/a[i][i];
  }
}

}
