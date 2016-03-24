#ifndef _UTL_H_
#define _UTL_H_

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <limits>


/*
   a..z - [97..122]
   A..Z - [65..90]
   0..9 - [48..57]
 */

namespace MSA
{


/**
 * Valor absoluto de um numero.
 */
#ifndef ABS
#define ABS(a) (((a)<0) ? (-(a)) : (a))
#endif

/**
 * Valor mi­nimo entre dois numeros.
 */
#ifndef MIN
#define MIN(a,b) ((a<b)?(a):(b))
#endif

/**
 * Valor mi­nimo entre dois numeros.
 */
#ifndef MAX
#define MAX(a,b) ((a>b)?(a):(b))
#endif

/**
* Retorna o maior entre dois números.
* @param a Primeiro número.
* @param b Segundo número.
* @return Retorna o maior entre os dois números recebidos.
*/
#define Greater(a, b) ((a>=b)?(a):(b))


/**
* Retorna o menor entre dois números.
* @param a Primeiro número.
* @param b Segundo número.
* @return Retorna o menor entre os dois números recebidos.
*/
#define Minor(a, b) ((a<=b)?(a):(b))



/**
 * Retorna um numero inteiro aleatorio n entre x e y (x <= n <= x).
 * @param x Inteiro x.
 * @param y Inteiro y.
 * @return Numero aleatorio inteiro retornado.
 */
int RandInt(int x,int y);


/**
 * Retorna um numero (ponto flutuante) aleatorio entre 0 e 1.
 * @return Numero aleatorio float retornado.
 */
double RandFloat();


/**
 * Retorna um numero booleano aleatorio.
 * @return Numero aleatorio booleano retornado.
 */
int RandBool();


/**
 * Retorna um numero n float aleatorio -1 < n < 1.
 * @return Numero n float aleatorio -1 < n < 1 retornado.
 */
double RandomClamped();


/**
 * Recebe tres numeros tipo double e restringe o valor do primeiro como
 * estando entre o segundo e o terceiro: min <= arg <= max.
 * @param arg Numero a ter seu valor restringido.
 * @param min Valor mínimo da restricao.
 * @param max Valor maximo da restricao.
 */
void Clamp(double *arg, double min, double max);


/**
 * Recebe tres numeros tipo double e restringe o valor do primeiro como
 * estando entre o segundo e o terceiro: min <= arg <= max.
 * @param arg Numero a ter seu valor restringido.
 * @param min Valor mínimo da restricao.
 * @param max Valor maximo da restricao.
 */
void Clamp(double& arg, double min, double max);


/**
 * Arredonda o valor recebido para o inteiro mais proximo.
 * @param val Valor em ponto flutuante a ter o valor arredondado.
 * @return Retorna o valor do inteiro mais proximo.
 */
int Rounded(double val);


/**
 * Arredonda o valor recebido para o inteiro mais proximo,
 * dependendo de o valor da mantissa ser maior ou menor que o offset.
 * @param val Valor em ponto flutuante a ter o valor arredondado.
 * @param offset Offset a ter seu valor comparado com a mantissa.
 * @return Retorna o valor do inteiro mais proximo, seguindo a restricao.
 */
int RoundUnderOffset(double val, double offset);


/**
  * Retorna a norna de um vetor qualquer.
  */
float lenght( float* vec1, int size );


/**
  * Normaliza o vetor recebido.
  */
float normalize( float* vec1, int size );


/**
  * Encontra o produto interno entre dois vetores.
  */
float inner( float* vec1, float* vec2, int size );


/**
* Obtem uma escala do vetor com todas as componentes dentro de um intervalo pre-definido.
* @param vec Vetor a ter os valores escalonados.
* @param maxScale Valor maximo do escalonamento.
* @param minScale Valor minimo do escalonamento.
*/
void intervalScaleIn( float* vec, float maxScale, float minScale, int size );

}
#endif
