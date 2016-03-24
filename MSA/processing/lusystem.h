#ifndef _LUSYSTEM_
#define _LUSYSTEM_

#if defined(__cplusplus)
extern "C" {
#endif



#ifndef ABS
/**
  * Retorna o módulo.
  * @param a Valor do qual se quer obter o módulo.
  * @return Retorna o módulo do valor passado.
  */
#define ABS(a) (((a)<0)?-(a):(a))
#endif

#ifndef SQR
/**
  * Encontra a raiz quadrada do valor passado como parâmetro.
  * @param a Valor do qual será calculada a raiz quadrada.
  * @return Retorna a raiz quadrada do valor passado.
  */
#define SQR(a) ((a)*(a))
#endif

#ifndef MAX
/**
  * Recebe dois valores e retorna o maior deles.
  * @param a Primeiro valor passado à função.
  * @param b Segundo valor passado à função.
  * @return Retorna o maior dos dois valores recebidos.
  */
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif

#ifndef MIN
/**
  * Recebe dois valores e retorna o menor deles.
  * @param a Primeiro valor passado à função.
  * @param b Segundo valor passado à função.
  * @return Retorna o menor dos dois valores recebidos.
  */
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

#ifndef SWAP
/**
  * Recebe duas variáveis e troca os valores, isto é, o valor da primeira passa a ser o da segunda e vice-versa.
  * @param a Primeira variável.
  * @param b Segunda variável.
  */
#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}
#endif

#ifndef _PI_
#define _PI_ 3.1415926535897932384626433832795
#endif

/**
 * Função que decompõe a matriz quadrada 'a' de dimensão 'n' na forma LU.
 * @param a Matriz quadrada a ser decomposta.
 * @param n Número de dimensões da matriz a.
 * @param indx Vetor contendo os x.
 */
int LuDcmp(double **a, int n, int *indx);

/**
 * Função que resolve o sistema ax=b.
 * @param a Matriz quadrada a ser decomposta.
 * @param n Número de dimensões da matriz a.
 * @param indx Vetor contendo os x.
 * @param Vetor contendo os b.
 */
void LuBksb(double **a, int n, int *indx, double *b);

#if defined(__cplusplus)
}
#endif

#endif
