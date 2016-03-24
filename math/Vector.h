/*
 * File:   Vector.h
 * Author: ederperez
 *
 * Definições de vetores/pontos tanto float quanto double.
 *
 * Created on October 28, 2014, 8:09 PM
 */

#ifndef VECTOR_H
#define VECTOR_H

#include "Vector2D.h"
#include "Vector3D.h"
#include "Vector4D.h"

namespace MSA
{
    typedef Vector2D<int>    Vector2Di;
    typedef Vector2D<short>  Vector2Ds;
    typedef Vector2D<float>  Vector2Df;
    typedef Vector2D<double> Vector2Dd;

    typedef Vector3D<int>    Vector3Di;
    typedef Vector3D<short>  Vector3Ds;
    typedef Vector3D<float>  Vector3Df;
    typedef Vector3D<double> Vector3Dd;

    typedef Vector4D<int>    Vector4Di;
    typedef Vector4D<short>  Vector4Ds;
    typedef Vector4D<float>  Vector4Df;
    typedef Vector4D<double> Vector4Dd;
}

#endif  /* VECTOR_H */
