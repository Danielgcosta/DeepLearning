#include "utl.h"

using namespace MSA;

namespace MSA
{

/**
 * Retorna um numero inteiro aleatorio n entre x e y (x <= n <= x).
 * @param x Inteiro x.
 * @param y Inteiro y.
 * @return Numero aleatorio inteiro retornado.
 */
int RandInt(int x,int y)
{
    return rand()%(y-x+1)+x;
}


/**
 * Retorna um numero (ponto flutuante) aleatorio entre 0 e 1.
 * @return Numero aleatorio float retornado.
 */
double RandFloat()
{
    return (rand())/(RAND_MAX+1.0);
}


/**
 * Retorna um numero booleano aleatorio.
 * @return Numero aleatorio booleano retornado.
 */
int RandBool()
{
    if (RandInt(0,1))
        return 1;
    else
        return 0;
}


/**
 * Retorna um numero n float aleatorio -1 < n < 1.
 * @return Numero n float aleatorio -1 < n < 1 retornado.
 */
double RandomClamped()
{
    return RandFloat() - RandFloat();
}


/**
 * Recebe tres numeros tipo double e restringe o valor do primeiro como
 * estando entre o segundo e o terceiro: min <= arg <= max.
 * @param arg Numero a ter seu valor restringido.
 * @param min Valor mínimo da restricao.
 * @param max Valor maximo da restricao.
 */
void Clamp(double *arg, double min, double max)
{
    if (*arg < min)
        *arg = min;
    if (*arg > max)
        *arg = max;
}


/**
 * Recebe tres numeros tipo double e restringe o valor do primeiro como
 * estando entre o segundo e o terceiro: min <= arg <= max.
 * @param arg Numero a ter seu valor restringido.
 * @param min Valor mínimo da restricao.
 * @param max Valor maximo da restricao.
 */
void Clamp(double& arg, double min, double max)
{
    if (arg < min)
        arg = min;
    if (arg > max)
        arg = max;
}


/**
 * Arredonda o valor recebido para o inteiro mais proximo.
 * @param val Valor em ponto flutuante a ter o valor arredondado.
 * @return Retorna o valor do inteiro mais proximo.
 */
int Rounded(double val)
{
    int    integral = (int)val;
    double mantissa = val - integral;
    if (mantissa < 0.5)
        return integral;
    else
        return integral + 1;
}


/**
 * Arredonda o valor recebido para o inteiro mais proximo,
 * dependendo de o valor da mantissa ser maior ou menor que o offset.
 * @param val Valor em ponto flutuante a ter o valor arredondado.
 * @param offset Offset a ter seu valor comparado com a mantissa.
 * @return Retorna o valor do inteiro mais proximo, seguindo a restricao.
 */
int RoundUnderOffset(double val, double offset)
{
    int    integral = (int)val;
    double mantissa = val - integral;
    if (mantissa < offset)
        return integral;
    else
        return integral + 1;
}


/**
  * Retorna a norna de um vetor qualquer.
  */
float lenght( float* vec1, int size )
{
    int i;
    float temp = 0;
    // Casos de retorno:
    if( vec1 == NULL)
        return -1;
    for( i=0 ; i<size ; i++ )
        temp+= (float)( vec1[i]*vec1[i] );
    temp = (float)sqrt(temp);
    return temp;
}


/**
  * Normaliza o vetor recebido.
  */
float normalize( float* vec1, int size )
{
    int i;
    // Casos de retorno:
    if( vec1 == NULL)
        return -1;
    float l = lenght( vec1, size );
    for( i=0 ; i<size ; i++ )
        vec1[i] = vec1[i]/l;
    return l;
}


/**
  * Encontra o produto interno entre dois vetores.
  */
float inner( float* vec1, float* vec2, int size )
{
    int i;
    float temp = 0;
    // Casos de retorno:
    if( (vec1 == NULL) || (vec2 == NULL) )
        return -1;
    for( i=0 ; i<size ; i++ )
        temp+= (float)( vec1[i]*vec2[i] );
    return temp;
}


/**
* Obtem uma escala do vetor com todas as componentes dentro de um intervalo pre-definido.
* @param vec Vetor a ter os valores escalonados.
* @param maxScale Valor maximo do escalonamento.
* @param minScale Valor minimo do escalonamento.
*/
void intervalScaleIn( float* vec, float maxScale, float minScale, int size )
{
	float scale;
	int i;
	float minValue = std::numeric_limits<float>::max();
	float maxValue = std::numeric_limits<float>::min();


	// Obtem o maior e o menor valores dentre das componentes do vetor:
	for( i=0 ; i<size ; i++ )
	{
		if( vec[i] < minValue )
			minValue = vec[i];
		if( vec[i] > maxValue )
			maxValue = vec[i];
	}

	// Escala a ser aplicada:
	scale   = (float)((maxScale - minScale)/(maxValue - minValue));

	// Aplicando a escala ao vetor:
	for( i=0 ; i<size ; i++ )
		vec[i] = (vec[i]-minValue)*scale + minScale;
}

}
