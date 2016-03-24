/*
 * featureUtils.h
 *
 *  Created on: 28/01/2015
 *      Author: aurelio
 */

#ifndef FEATUREUTILS_H_
#define FEATUREUTILS_H_


#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <assert.h>
#include <numeric>
#include <vector>
#include <limits>
#include <complex>

namespace MSA
{

/**
 * Funcao auxiliar. Retorna a maior diferenca entre posicoes relativas de dos vetores.
 */
template <typename Type1>
static double GreaterDiff( Type1* vec1, Type1* vec2, int vecsSize )
{
	double result = fabs( (double)(vec1[0]) - (double)(vec2[0]) );
	for( int i=1 ; i<vecsSize ; i++ )
	{
		double thisResult = fabs( (double)(vec1[i]) - (double)(vec2[i]) );
		if( thisResult > result )
			result = thisResult;
	}
	return result;
}
// Versao 2: ///////////////////////////////////////////////////
template <typename Type1>
static double GreaterDiff( std::vector<Type1> &vec1, Type1* vec2, int vecsSize )
{
	double result = std::abs( (double)(vec1[0]) - (double)(vec2[0]) );
	for( int i=1 ; i<vecsSize ; i++ )
	{
		double thisResult = fabs( (double)(vec1[i]) - (double)(vec2[i]) );
		if( thisResult > result )
			result = thisResult;
	}
	return result;
}
// Versao 3: ///////////////////////////////////////////////////
template <typename Type1>
static double GreaterDiff( Type1* vec1, std::vector<Type1> &vec2, int vecsSize )
{
	double result = fabs( (double)(vec1[0]) - (double)(vec2[0]) );
	for( int i=1 ; i<vecsSize ; i++ )
	{
		double thisResult = fabs( (double)(vec1[i]) - (double)(vec2[i]) );
		if( thisResult > result )
			result = thisResult;
	}
	return result;
}


/**
 * Copia os valores do primeiro vetor no segundo.
 */
template <typename Type1>
static int CopyVector( Type1* vec1, Type1* vec2, int vecsSize )
{
	if( (vec1 == NULL) || (vec2 == NULL) )
		return -1;
	for( int i=0 ; i<vecsSize ; i++ )
		vec2[i] = vec1[i];
	return 0;
}
// Versao 2: ///////////////////////////////////////////////////
template <typename Type1>
static int CopyVector( std::vector<Type1> &vec1, Type1* vec2, int vecsSize )
{
	if( (vec1.size() < (unsigned int)vecsSize) || (vec2 == NULL) )
		return -1;
	for( int i=0 ; i<vecsSize ; i++ )
		vec2[i] = vec1[i];
	return 0;
}
// Versao 3: ///////////////////////////////////////////////////
template <typename Type1>
int CopyVector( Type1* vec1, std::vector<Type1> &vec2, int vecsSize )
{
	if( (vec1 == NULL) || (vec2.size() < (unsigned int)vecsSize) )
		return -1;
	vec2.reserve( vecsSize );
	vec2.resize( vecsSize );
	for( int i=0 ; i<vecsSize ; i++ )
		vec2[i] = vec1[i];
	return 0;
}
// Versao 4: ///////////////////////////////////////////////////
template <typename Type1>
int CopyVector( std::vector<Type1> vec1, std::vector<Type1> &vec2, int vecsSize )
{
	if( (vec1.size() < (unsigned int)vecsSize) || (vec2.size() < (unsigned int)vecsSize) )
		return -1;
	vec2.reserve( vecsSize );
	vec2.resize( vecsSize );
	for( int i=0 ; i<vecsSize ; i++ )
		vec2[i] = vec1[i];
	return 0;
}


/**
 * Recebe uma matriz de vetores e faz com que cada componente tenha (no conjunto) media 0.
 */
template <typename Type1>
static int toMeanZero( Type1** matrix, int nLines, int nColumns )
{
	if( matrix == NULL )
		return -1;

	std::vector<Type1> vec;
	for( int i=0 ; i<nColumns ; i++ )
		vec.push_back( (Type1)0 );

	for( int i=0 ; i<nLines ; i++ )
		for( int j=0 ; j<nColumns ; j++ )
			vec[j] += matrix[i][j];

	for( int j=0 ; j<nColumns ; j++ )
		vec[j] = vec[j]/nLines;

	for( int i=0 ; i<nLines ; i++ )
		for( int j=0 ; j<nColumns ; j++ )
			matrix[i][j] = matrix[i][j] - vec[j];

	return 0;
}


/**
 * Obtem uma versao escalonada do valor dentro de um intervalo pre-definido.
 * @param maxScale Valor maximo da escala a ser aplicada.
 * @param minScale Valor mi�nimo da escala a ser aplicada.
 * @param maxValue Valor maximo dentre que escvalue pode atingir.
 * @param minValue Valor mi�nimo dentre que escvalue pode atingir.
 * @param escvalue Valor de entrada, que e sobreescrito com o valor de sai�da.
 */
template <typename Type1>
static void scaleIn( Type1 maxScale, Type1 minScale, Type1 maxValue, Type1 minValue, Type1& escvalue )
{
	Type1 scale;
	// Escala a ser aplicada:
	scale = (Type1) ( ( maxScale - minScale ) / ( maxValue - minValue ) );
	escvalue = ( escvalue - minValue ) * scale + minScale;
}


/**
 * Recebe uma matriz de vetores e faz com que cada componente tenha (no conjunto) media 0.
 */
template <typename Type1>
static int toInterval( std::vector< std::vector<Type1> >& matrix, Type1 maxScale, Type1 minScale )
{
	if( matrix.size() == 0 )
		return -1;

	// Encontrando valores de mi�nimo e maximo dos voxels:
	Type1 min, max;
	min = std::numeric_limits<Type1>::max();
	max = std::numeric_limits<Type1>::min();
	for( int i=0 ; i<(int)matrix.size() ; i++ )
	{
		for( int j=0 ; j<(int)((matrix[0]).size()) ; j++ )
		{
			if( min > matrix[i][j] )
				min = matrix[i][j];
			if( max < matrix[i][j] )
				max = matrix[i][j];
		}
	}

	// Aplicando a escala:
	for( int i=0 ; i<(int)matrix.size() ; i++ )
	{
		for( int j=0 ; j<(int)((matrix[0]).size()) ; j++ )
		{
			Type1 escvalue = matrix[i][j];
			scaleIn( maxScale, minScale, max, min, escvalue );
			matrix[i][j] = escvalue;
		}
	}

	return 0;
}


/**
 * Recebe uma matriz de vetores e faz com que cada componente tenha (no conjunto) media 0.
 */
template <typename Type1>
static int toMeanZero( std::vector<Type1*> matrix, int nColumns )
{
	if( matrix.size() == 0 )
		return -1;

	int nLines = matrix.size();

	std::vector<Type1> vec;
	for( int i=0 ; i<nColumns ; i++ )
		vec.push_back( (Type1)0 );

	for( int i=0 ; i<nLines ; i++ )
		for( int j=0 ; j<nColumns ; j++ )
			vec[j] += matrix[i][j];

	for( int j=0 ; j<nColumns ; j++ )
		vec[j] = vec[j]/nLines;

	for( int i=0 ; i<nLines ; i++ )
		for( int j=0 ; j<nColumns ; j++ )
			matrix[i][j] = matrix[i][j] - vec[j];

	return 0;
}


/**
 */
static int Save_File( float** features, int nfeatures, int dimfeatures, char* filename )
{
	FILE *f;
	char name[128];

	/* Inserting the correct extensions: */
	sprintf( name, "%s.arff", &(filename[0]) );

	// Abre o arquivo para escrita. Em caso de erro, retorna -3:
	f = fopen(name, "w");
	if ( f == NULL )
		return -1;

	// Cabecalho:
	fprintf(f, "FEATURES LIST\n\n");

	// Numero de vetores:
	fprintf(f, "Number of vectors: %d\n", nfeatures);

	// Dimensão do dado:
	fprintf(f, "Vectors dimension: %d\n", dimfeatures);

	int i, j; /* Temporary counters. */
	for( i=0 ; i<nfeatures ; i++ )
	{
		for( j=0 ; j<dimfeatures ; j++ )
		{
			fprintf(f, "%.7f", features[i][j] );
			if( j < dimfeatures-1 )
				fprintf(f, ", " );
		}
		fprintf(f, "\n" );
	}

	// Fecha o arquvo:
	fclose(f);

	return 0;
}


/**
 */
static int Save_File( std::vector<float*> features, int dimfeatures, char* filename )
{
	FILE *f;
	char name[128];

	int nfeatures = features.size();

	/* Inserting the correct extensions: */
	sprintf( name, "%s.arff", &(filename[0]) );

	// Abre o arquivo para escrita. Em caso de erro, retorna -3:
	f = fopen(name, "w");
	if ( f == NULL )
		return -1;

	// Cabecalho:
	fprintf(f, "FEATURES LIST\n\n");

	// Numero de vetores:
	fprintf(f, "Number of vectors: %d\n", nfeatures);

	// Dimensão do dado:
	fprintf(f, "Vectors dimension: %d\n", dimfeatures);

	int i, j; /* Temporary counters. */
	for( i=0 ; i<nfeatures ; i++ )
	{
		for( j=0 ; j<dimfeatures ; j++ )
		{
			fprintf(f, "%.7f", features[i][j] );
			if( j < dimfeatures-1 )
				fprintf(f, ", " );
		}
		fprintf(f, "\n" );
	}

	// Fecha o arquvo:
	fclose(f);

	return 0;
}


/**
 */
static int Load_File( float*** features, int* nfeatures, int* dimfeatures, char* filename )
{
	FILE *f;
	char name[128];

	/* Inserting the correct extensions: */
	sprintf( name, "%s", &(filename[0]) );

	// Abre o arquivo para escrita. Em caso de erro, retorna -3:
	f = fopen(name, "r");
	if ( f == NULL )
		return -1;

	// Cabecalho:
	int numRead;
	numRead = fscanf(f, "FEATURES LIST\n\n");
	numRead =0;

	// Numero de vetores:
	int tmpNfeatures = 0;
	numRead = fscanf(f, "Number of vectors: %d\n", &tmpNfeatures);
	*nfeatures = tmpNfeatures;

	int tmpDimFeatures = 0;
	numRead = fscanf(f, "Vectors dimension: %d\n", &tmpDimFeatures);
	*dimfeatures = tmpDimFeatures;

	float** tmpfeatures = new float*[tmpNfeatures];
	float* tmpfeaturesVec = new float[tmpDimFeatures*tmpNfeatures];

	int i, j; /* Temporary counters. */
	for( i=0 ; i<tmpNfeatures ; i++ )
	{
		tmpfeatures[i] = &(tmpfeaturesVec[tmpDimFeatures*i]);
		for( j=0 ; j<tmpDimFeatures ; j++ )
		{
			numRead = fscanf(f, "%f", &(tmpfeatures[i][j]) );
			if( j < tmpDimFeatures-1 )
				numRead = fscanf(f, ", " );
		}
		numRead = fscanf(f, "\n" );
	}
	// Retornando os resultados:
	*features = tmpfeatures;
	*nfeatures = tmpNfeatures;
	*dimfeatures = tmpDimFeatures;

	// Fecha o arquvo:
	fclose(f);
	return 0;
}

}

#endif /* FEATUREUTILS_H_ */
