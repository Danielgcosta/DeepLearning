
#ifndef _STL_UTILS_H_
#define _STL_UTILS_H_

#include <vector>

namespace MSA
{

/**
  *  stl_utils.h Contem funcoes uteis na manipulacao de objetos (vetores da STL).
*/


/**
  * Configura a nova capacidade do vetor para que ele seja exatamente do tamanho. Com isso o vetor nao e
  * aumentado de metade do seu tamanho, como a STL normalmente faria. Deve ser utilizada quando se sabe
  * exatamente o tamanho do vetor a ser alocado, otimizando a alocacao de memoria.
  */
template<typename T>
inline static void vectorExactResize( std::vector<T>& v, int newSize )
{
	v.reserve( newSize );
	v.resize( newSize );
}


/**
  * Configura a nova capacidade do vetor para que ele seja exatamente do tamanho. Com isso o vetor nao e
  * aumentado de metade do seu tamanho, como a STL normalmente faria. O parametro "pad" contem o numero de
  * elementos que deverao ser adicionados ao final do vetor, fazendo o novo tamanho ser maior que o atual.
  * Deve ser utilizada quando se sabe exatamente o tamanho do vetor a ser alocado, otimizando a alocacao de memoria.
  */
template<typename T>
inline static void vectorExactResize( std::vector<T>& v, int newSize, T pad )
{
	v.reserve( newSize );
	v.resize( newSize, pad );
}


/**
  * Essa funcao auxiliar requer a duplicacao temporaria do vetor a ser processado. Portanto,
  * a funcao nao e recomendada para vetores muito grandes. Modifica o vetor de forma que ele
  * somente contenha os valores situados entre os índices "first" e "last".
  */
template<typename T>
inline static void vectorClip( std::vector<T>& v, int first, int last )
{
	std::vector<T>( v.begin() + first, v.begin() + last ).swap( v );
}


/**
  * Reconfigura a capacidade do vetor para que seja exatamente do tamanho atual, liberando
  * toda a memoria nao utilizada.
  */
template<typename T>
inline static void vectorTrim( std::vector<T>& v )
{
	std::vector<T>( v ).swap( v );
}


/**
  * Seta a capacidade do vetor para zero, liberando toda a memoria.
  */
template<typename T>
inline static void vectorFreeMemory( std::vector<T>& v )
{
	std::vector<T>().swap( v );
}

}

#endif // _STL_UTILS_H_
