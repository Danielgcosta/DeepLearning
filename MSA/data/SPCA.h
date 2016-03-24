/*
 * File:   SPCA.h
 * Author: Aurélio
 *
 * Created on May 13, 2010, 11:50 AM
 */


#ifndef SPCA_H_
#define SPCA_H_


#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "utl.h"
#include "featureUtils.h"

namespace MSA
{

typedef std::vector<double> spcaVector;


/**
 * Simple Principal Component Analysis method.
 * It estimates the PCA components from a set of vectors in a very fast iterative manner.
 * Implementation based on its original paper:
 * "Fast Dimensionality Reduction and Simple PCA",
 * written by Matthew Partridge and Rafael Calvo.
 *
 * The algorithmic complexity of the SPCA algorithm is O(dmn) which is equal
 * to the complexity of Hebbian learning algorithms.
 * However, the paper shows each iteration is much faster.
 */
class SPCA
{
public:
	int _spca_nLenght;
	std::vector<double> _spca_alphas;


	/**
	 * Destructor.
	 */
	~SPCA( );


	/**
	 * Constructor.
	 */
	SPCA( );


	/**
	 * Constructor.
	 * @param It receives spatial dimension of the samples.
	 */
	SPCA( int spca_nLenght );


	/**
	 * It returns a normalized version of the vector.
	 * @param vct Vector to be normalized.
	 */
	void SPCA_Normalize( std::vector<double>& vct );


	/**
	 * It calculates the y value over an arbitrary entry vector.
	 * @param xj Entry vector.
	 * @return Returns the resulting y value.
	 */
	double SPCA_Y( std::vector<double> xj );


	/**
	 * It calculates the y value over an arbitrary entry vector.
	 * @param xj Entry vector.
	 * @return Returns the resulting y value.
	 */
	double SPCA_Y( double* xj );


	/**
	 * It estimates the alpha of the next iteration.
	 * @param entries Matrix that contains all the entry vectors.
	 */
	void SPCA_Next_Alpha( std::vector<spcaVector> entries );


	/**
	 * It estimates the alpha of the next iteration.
	 * @param entries Matrix that contains all the entry vectors.
	 * @param entriesSize Number of vector contained into the entries matrix.
	 */
	void SPCA_Next_Alpha( double** entries, int entriesSize );


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
	void SPCA_entries_deflation( std::vector<spcaVector>& entries );


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
	void SPCA_entries_deflation( double** entries, int entriesSize );

};

}

#endif /* SPCA_H_ */

