/* 
 * File:   DataGrid3.h
 * Author: Aurélio
 *
 * Created on August 7, 2009, 4:15 PM
 */

#ifndef _DATAGRID3_H
#define	_DATAGRID3_H


#include <vector>
#include <cassert>
#include "stl_utils.h"

namespace MSA
{

template<class T>
class DataGrid3
{
public:
    DataGrid3()
    {
        _numCellsX = 0;
        _numCellsXY = 0;
    }

    DataGrid3( int numCellsX, int numCellsY, int numCellsZ )
    {
        setDimensions( numCellsX, numCellsY, numCellsZ );
    }

    void setDimensions( int numCellsX, int numCellsY, int numCellsZ )
    {
        _numCellsX = numCellsX;
        _numCellsXY = numCellsX * numCellsY;
        _data.resize( numCellsX * numCellsY * numCellsZ );
    }

    inline T operator() ( int x, int y, int z ) const
    {
        return _data[getLinearIndex( x, y, z )];
    }

    inline T& operator() ( int x, int y, int z )
    {
        return _data[getLinearIndex( x, y, z )];
    }

    bool isInside( int x, int y, int z )
    {
        return x >= 0 && y >= 0 && z >= 0 && x < getNumCellsX() && y < getNumCellsY() && z < getNumCellsZ();
    }

    inline int getLinearIndex( int x, int y, int z )
    {
        assert( x >= 0 && y >= 0 && z >= 0 && x < getNumCellsX() && y < getNumCellsY() && z < getNumCellsZ() );
        return x + _numCellsX * y + _numCellsXY * z;
    }

    int getNumCellsX()
    {
        return _numCellsX;
    }

    int getNumCellsY()
    {
        if( _data.empty() )
            return 0;

        return _numCellsXY / _numCellsX;
    }

    int getNumCellsZ()
    {
        if( _data.empty() )
            return 0;

        return (int)_data.size() / _numCellsXY;
    }

    void getDimensions( int& x, int& y, int& z )
    {
        x = getNumCellsX();
        y = getNumCellsY();
        z = getNumCellsZ();
    }

    void clear()
    {
        vectorFreeMemory( _data );
        _numCellsX = 0;
        _numCellsXY = 0;
    }

private:
    int _numCellsX;
    int _numCellsXY;
    std::vector<T> _data;
};

}

#endif	/* _DATAGRID3_H */

