/*
 * Matrix.h
 *
 *  Created on: Dec 23, 2014
 *      Author: aurelio
 */

#ifndef DEEPLEARNING_MATRIX_H_
#define DEEPLEARNING_MATRIX_H_

#include <omp.h>

#include <stdio.h>
#include <math.h>
#include <sstream>
#include <string>
#include <vector>

#include "utl.h"


using namespace MSA;



class Matrix
{
public:

    enum OpType
    {
    	ADD=0,
    	SUBTRACT,
    	MULTIPLY,
    	DIVIDE,
    	POWER
    };


	std::vector< std::vector<double> > _weights;
	unsigned int _rows;
	unsigned int _columns;


	Matrix()
	{
		_rows = 0;
		_columns = 0;
	}


	Matrix(	unsigned int irows, unsigned int icolumns )
	{
		_weights.resize( irows, std::vector<double>( icolumns, 0.0 ) );
		_rows = irows;
		_columns = icolumns;
	}


	Matrix( float**inMatrix, int inSize, int nSamples )
	{
		_weights.resize( nSamples, std::vector<double>( inSize, 0.0 ) );
		_rows = nSamples;
		_columns = inSize;

		for( unsigned int i=0 ; i<_rows ; i++ )
			for( unsigned int j=0 ; j<_columns ; j++ )
				set( i, j, inMatrix[i][j] );
	}


	void resize( unsigned int irows, unsigned int icolumns )
	{
		_weights.clear();
		_weights.resize( irows, std::vector<double>( icolumns, 0.0 ) );
		_rows = irows;
		_columns = icolumns;
	}


	~Matrix()
	{
//		_weights.clear();
	}


	unsigned int& rows()
	{
		return _rows;
	}


	unsigned int& columns()
	{
		return _columns;
	}


	void matrixMultiplication( Matrix& b, Matrix& result )
	{
		unsigned int ma = _weights.size();
		unsigned int na = (_weights[0]).size();
		unsigned int mb = (b._weights).size();
		unsigned int nb = (b._weights[0]).size();

		if( na != mb )
			std::cout << "Matrix matrixMultiplication. na != mb. Erro." << std::endl;

		result.resize( ma, nb );
		unsigned int i, j , v;

		#pragma omp parallel shared( result ) private(i,j,v)
		{
			#pragma omp for  schedule(static)
			for (i = 0 ; i < ma; i++ )
				for (j = 0; j < nb; j++)
					for (v = 0; v < na; v++)
						result._weights[i][j] = result._weights[i][j] + _weights[i][v] * (b._weights)[v][j];
		}
	}


	void set( unsigned int i, unsigned int j, double val )
	{
		_weights[i][j] = val;
	}


	double get( unsigned int i, unsigned int j )
	{
		return _weights[i][j];
	}


    void copy( Matrix& result )
    {
    	if( (result.rows() != rows()) || (result.columns() != columns()) )
    		result.resize( _rows, _columns );

    	unsigned int r=0;
    	unsigned int c=0;
		#pragma omp parallel shared( result ) private(r,c)
		{
			#pragma omp for  schedule(static)
				for( r=0 ; r<_rows ; r++ )
					for( c=0 ; c<_columns ; c++ )
						result.set( r, c, get( r, c ) );
		}
    }


	void insert( unsigned int axis, double value )
	{
		if( axis > 1 )
			return;
		if( axis == 0 )
		{
			std::vector< std::vector<double> > tmp_weights( (_rows+1), std::vector<double>( _columns, 0.0 ) );

			for( unsigned int c=0 ; c<_columns ; c++ )
        		tmp_weights[0][c] = value;

	    	for( unsigned int r=0 ; r<_rows ; r++ )
	        	for( unsigned int c=0 ; c<_columns ; c++ )
	        		tmp_weights[r+1][c] = _weights[r][c];
			_rows ++;
			_weights = tmp_weights;
		}

		if( axis == 1 )
		{
			std::vector< std::vector<double> > tmp_weights( _rows, std::vector<double>( (_columns+1), 0.0 ) );

	    	for( unsigned int r=0 ; r<_rows ; r++ )
        		tmp_weights[r][0] = value;

	    	for( unsigned int r=0 ; r<_rows ; r++ )
	        	for( unsigned int c=0 ; c<_columns ; c++ )
	        		tmp_weights[r][c+1] = _weights[r][c];
			_columns++;
			_weights = tmp_weights;
		}
	}


    void transpose( Matrix& result )
    {
    	result.resize( _columns, _rows );

    	unsigned int c=0;
    	unsigned int r=0;
#pragma omp parallel for private(c)
    	for( r=0 ; r<_rows ; r++ )
    	{
        	for( c=0 ; c<_columns ; c++ )
        		result.set( c, r, get( r, c ) );
    	}
    }


    void random()
    {
    	for( unsigned int r=0 ; r<_rows ; r++ )
        	for( unsigned int c=0 ; c<_columns ; c++ )
        		set( r, c, RandFloat() );
    }


    unsigned int dim()
    {
        return rows() * columns();
    }


	void apply( double b, OpType op )
	{
		unsigned int i, j;

		#pragma omp parallel for private(j)
		for( i=0 ; i <_rows ; i++ )
			for( j=0 ; j<_columns ; j++ )
			{
				if( op == ADD )
					_weights[i][j] = _weights[i][j] + b;
				if( op == SUBTRACT )
					_weights[i][j] = _weights[i][j] - b;
				if( op == MULTIPLY )
					_weights[i][j] = _weights[i][j] * b;
				if( op == DIVIDE )
					_weights[i][j] = _weights[i][j] / b;
				if( op == POWER )
					_weights[i][j] = pow( (_weights[i][j]), b );
			}
	}


	void apply( Matrix& b, OpType op, Matrix& mr )
	{
		if( _rows != b.rows() )
			std::cout << "Matrix matrixArithmetic_operations. _rows != b.rows(). Erro op: " << (int)op << std::endl;

		if( _columns != b.columns() )
			std::cout << "Matrix matrixArithmetic_operations. _columns != b.columns(). Erro op: " << (int)op << std::endl;

		mr.resize( _rows, _columns );

		unsigned int i, j;

		#pragma omp parallel shared( mr ) private(i,j)
		{
			#pragma omp for  schedule(static)
			for( i=0 ; i<_rows ; i++ )
				for( j=0 ; j< _columns; j++ )
				{
					if( op == ADD )
						mr.set( i, j, (_weights[i][j] + b.get( i, j )) );
					if( op == SUBTRACT )
						mr.set( i, j, (_weights[i][j] - b.get( i, j )) );
					if( op == MULTIPLY )
						mr.set( i, j, (_weights[i][j] * b.get( i, j )) );
					if( op == DIVIDE )
						mr.set( i, j, (_weights[i][j] / b.get( i, j )) );
					if( op == POWER )
						mr.set( i, j, pow( (_weights[i][j]), b.get( i, j ) ) );
				}
		}
	}


    void add( Matrix& m2, Matrix& result )
    {
    	apply( m2, ADD, result );
    }

    void subtract( Matrix& m2, Matrix& result )
    {
    	apply( m2, SUBTRACT, result );
    }

    void multiply( Matrix& m2, Matrix& result )
    {
    	apply( m2, MULTIPLY, result );
    }

    void divide( Matrix& m2, Matrix& result )
    {
    	apply( m2, DIVIDE, result );
    }

    void multiply( double s, Matrix& result )
    {
    	copy( result );
    	result.apply( s, MULTIPLY );
    }

    void divide( double s, Matrix& result )
    {
    	copy( result );
    	result.apply( s, DIVIDE );
    }


    void power( double exponent, Matrix& result )
    {
    	copy( result );
    	result.apply( exponent, POWER );
    }

    double sum()
    {
        double sum = 0.0;
        for( unsigned int i = 0; i < rows(); i++)
        {
            for( unsigned int j = 0; j < columns(); j++)
            {
                sum += get(i, j);
            }
        }
        return sum;
    }


    /**
     * Print the actual weight values.
     */
    void printWeights( )
    {
        for( unsigned int i = 0; i < rows(); i++)
        {
            for( unsigned int j = 0; j < columns(); j++)
            {
        		std::cout << _weights[i][j] << ", ";
            }
    		std::cout << std::endl;
        }
        return;
    }


};

#endif /* DEEPLEARNING_MATRIX_H_ */
