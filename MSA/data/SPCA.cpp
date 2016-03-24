/*
 * File:   SPCA.cpp
 * Author: Aurélio
 *
 * Created on May 13, 2010, 11:50 AM
 */

#include "SPCA.h"


using namespace MSA;

namespace MSA
{


/**
 * Destructor.
 */
SPCA::~SPCA( )
{
	_spca_alphas.clear();
}


/**
     * Constructor.
     */
SPCA::SPCA( )
{
	_spca_nLenght = -1;
}


/**
 * Constructor.
 * @param It receives spatial dimension of the samples.
 */
SPCA::SPCA( int spca_nLenght )
{
	// The vectors lenght;
	_spca_nLenght = spca_nLenght;

	// Creating the alphas vector and the corresponding y's vector:
	_spca_alphas.reserve( spca_nLenght );
	for( int i=0 ; i< _spca_nLenght ; i++ )
	{
		double value = RandFloat();
		_spca_alphas.push_back( value );
	}
}


/**
 * It returns a normalized version of the vector.
 * @param vct Vector to be normalized.
 */
void SPCA::SPCA_Normalize( std::vector<double>& vct )
{
	double norm=0;
	for( int i=0 ; i<(int)vct.size() ; i++ )
		norm += (vct[i] * vct[i]);
	norm = (double)(sqrt(norm));

	for( int i=0 ; i<(int)vct.size() ; i++ )
		vct[i] = vct[i]/norm;
}


/**
 * It calculates the y value over an arbitrary entry vector.
 * @param xj Entry vector.
 * @return Returns the resulting y value.
 */
double SPCA::SPCA_Y( std::vector<double> xj )
{
	double y=0;
	for( int i=0 ; i<_spca_nLenght ; i++ )
		y += static_cast<double>(xj[i]) * _spca_alphas[i];

	return y;
}


/**
 * It calculates the y value over an arbitrary entry vector.
 * @param xj Entry vector.
 * @return Returns the resulting y value.
 */
double SPCA::SPCA_Y( double* xj )
{
	double y=0;
	for( int i=0 ; i<_spca_nLenght ; i++ )
		y += static_cast<double>(xj[i]) * _spca_alphas[i];

	return y;
}


/**
 * It estimates the alpha of the next iteration.
 * @param entries Matrix that contains all the entry vectors.
 */
void SPCA::SPCA_Next_Alpha( std::vector<spcaVector> entries )
{
	std::vector<double> auxVector;
	auxVector.resize(_spca_nLenght);
	auxVector.assign(auxVector.size(), 0 );

	int entriesSize = entries.size();

	for( int i=0 ; i<entriesSize ; i++ )
	{
		spcaVector thisVector = entries[i];
		double y = SPCA_Y( thisVector );

		if( y >= 0 )
		{
			for( int j = 0; j < _spca_nLenght; j++ )
			{
				auxVector[i] += static_cast<double>(thisVector[i]);
			}
		}
	}

	// Normalizing the resulting vector:
	SPCA_Normalize( auxVector );
	// The "auxVector" is the alpha vector of the next iteration:
	for( int j = 0; j < _spca_nLenght; j++ )
		_spca_alphas[j] = auxVector[j];
}


/**
 * It estimates the alpha of the next iteration.
 * @param entries Matrix that contains all the entry vectors.
 * @param entriesSize Number of vector contained into the entries matrix.
 */
void SPCA::SPCA_Next_Alpha( double** entries, int entriesSize )
{
	std::vector<double> auxVector;
	auxVector.resize(_spca_nLenght);
	auxVector.assign(auxVector.size(), 0 );

	for( int i=0 ; i<entriesSize ; i++ )
	{
		double*  thisVector = entries[i];
		double y = SPCA_Y( thisVector );

		if( y >= 0 )
		{
			for( int j = 0; j < _spca_nLenght; j++ )
			{
				auxVector[j] += thisVector[j];
			}
		}
	}

	// Normalizing the vector
	SPCA_Normalize( auxVector );
	for( int j = 0; j < _spca_nLenght; j++ )
		_spca_alphas[j] = auxVector[j];
}


/**
 * Once the Principal Component is estimated (alpha vector), we need find the next principal component.
 * To acomplish this, the effect of the first principal component is removed from the dataset,
 * so as to avoid being found again. The removal of a component from the dataset, is called “deflation”.
 * This is performed by subtracting the component of each datum which is parallel to the principal
 * component from that datum, leaving only an orthogonal component. That is, datum , is deflated by
 * the unit principal component direction, according to:
 *                                 xi' = xi - (alphat.xi).alpha,     where:
 * - xi is an arbitrary entry vector;
 * - at is the trasposed alpha vector;
 * - a is the alpha vector.
 *
 * All the entry vectors from the dataset needs to be transformed.
 * This function executes this transformation.
 */
void SPCA::SPCA_entries_deflation( std::vector<spcaVector>& entries )
{
	int entriesSize = entries.size( );

	for( int i = 0; i < entriesSize; i++ )
	{
		spcaVector thisVector = entries[i];

		for( int j = 0; j < _spca_nLenght; j++ )
			thisVector[i] = thisVector[i] - (thisVector[i] * (double)(_spca_alphas[i]) * (double)(_spca_alphas[i]));
	}
}


/**
 * Once the Principal Component is estimated (alpha vector), we need find the next principal component.
 * To acomplish this, the effect of the first principal component is removed from the dataset,
 * so as to avoid being found again. The removal of a component from the dataset, is called “deflation”.
 * This is performed by subtracting the component of each datum which is parallel to the principal
 * component from that datum, leaving only an orthogonal component. That is, datum , is deflated by
 * the unit principal component direction, according to:
 *                                 xi' = xi - (alphat.xi).alpha,     where:
 * - xi is an arbitrary entry vector;
 * - at is the trasposed alpha vector;
 * - a is the alpha vector.
 *
 * All the entry vectors from the dataset needs to be transformed.
 * This function executes this transformation.
 */
void SPCA::SPCA_entries_deflation( double** entries, int entriesSize )
{
	int i;
	for( i = 0; i < entriesSize; i++ )
	{
		double*  thisVector = entries[i];

		double inner = 0;
		int j;
		for( j = 0; j < _spca_nLenght; j++ )
			inner += thisVector[j] * _spca_alphas[j];

		for( j = 0; j < _spca_nLenght; j++ )
			thisVector[j] = thisVector[j] - (double)(inner * _spca_alphas[j]);
	}
}

}
