/* 
 * Arquivo contendo definições de tipos.
 * 
 * File:   DataTypes.h
 * Author: ederperez
 *
 */

#ifndef DATA_TYPES_H
#define	DATA_TYPES_H

#include <vector>
#include "Eigen/Dense"

#ifdef DOUBLE_PRECISION
typedef double FLOAT;
#else
typedef float FLOAT;
#endif

typedef Eigen::Matrix<FLOAT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixNxM;
typedef Eigen::Matrix<FLOAT, Eigen::Dynamic, 1> VectorN;

#endif	// DATA_TYPES_H
