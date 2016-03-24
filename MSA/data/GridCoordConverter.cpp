#include "GridCoordConverter.h"

#include "../../math/Vector.h"

namespace MSA
{

GridCoordConverter::GridCoordConverter( int startInline,
												  int inlineIncrement,
												  int inlineSteps,
												  int startCrossline,
												  int crosslineIncrement,
												  int crosslineSteps,
												  double p1x, double p1y,
												  double p2x, double p2y,
												  double p3x, double p3y )
{
    
    initialize( startInline, inlineIncrement, inlineSteps, startCrossline, crosslineIncrement, crosslineSteps, p1x, p1y, p2x, p2y, p3x, p3y );
}


GridCoordConverter::GridCoordConverter( const GridCoordConverter& orig )
{
    _startInline     = orig._startInline;
	_inlineIncrement = orig._inlineIncrement;
	_inlineSteps     = orig._inlineSteps;

	_startCrossline     = orig._startCrossline;
	_crosslineIncrement = orig._crosslineIncrement;	
	_crosslineSteps     = orig._crosslineSteps;

    _p1x = orig._p1x;
    _p1y = orig._p1y;
    _p2x = orig._p2x;
    _p2y = orig._p2y;
	_p3x = orig._p3x;
    _p3y = orig._p3y;

    _inlineCellVectorX    = orig._inlineCellVectorX;
    _inlineCellVectorY    = orig._inlineCellVectorY;
    _crosslineCellVectorX = orig._crosslineCellVectorX;
    _crosslineCellVectorY = orig._crosslineCellVectorY;

    _indexToMapMatrix = orig._indexToMapMatrix;
    _mapToIndexMatrix = orig._mapToIndexMatrix;
}



GridCoordConverter::~GridCoordConverter()
{
}


void GridCoordConverter::initialize( int startInline,
                                     int inlineIncrement,
                                     int inlineSteps,
                                     int startCrossline,
                                     int crosslineIncrement,
                                     int crosslineSteps,
                                     double p1x, double p1y,
                                     double p2x, double p2y,
                                     double p3x, double p3y )
{
    _p1x  = p1x;
    _p1y  = p1y;
    _p2x  = p2x;
    _p2y  = p2y;
    _p3x  = p3x;
    _p3y  = p3y;

    _startInline		= startInline;
    _inlineIncrement    = inlineIncrement;
    _inlineSteps        = (inlineSteps > 0) ? inlineSteps : 1;
    _inlineCellVectorX = (_p3x - _p1x) / _inlineSteps;
    _inlineCellVectorY = (_p3y - _p1y) / _inlineSteps;

    _startCrossline     = startCrossline;
    _crosslineIncrement = crosslineIncrement;
    _crosslineSteps     = (crosslineSteps > 0) ? crosslineSteps : 1;
    _crosslineCellVectorX = (_p2x - _p1x) / _crosslineSteps;
    _crosslineCellVectorY = (_p2y - _p1y) / _crosslineSteps;

    //ACOX++
    if( _inlineSteps < 2 )
    {
        // Adiciona dois elementos ao grid da inline
    	_inlineSteps = 2;

        //Calcula o vetor normalizado da direção crossline
        Vector3Dd xlVector = Vector3Dd(p2x - p1x, p2y - p1y, 0);
        xlVector.normalize();

        Vector3Dd timeVector = Vector3Dd(0, 0, 1);

        //Calcula o vetor normalizado da direção perpendicular a crossline (produto vetorial de xlVector pelo vetor do tempo (0,0,1))
        Vector3Dd directionVector = xlVector.cross(timeVector);

        if( ( _p3x - _p1x == 0 ) && ( _p3y - _p1y ) == 0 )
        {
        	_inlineCellVectorX = _crosslineCellVectorY;
      	    _inlineCellVectorY = _crosslineCellVectorX;
        }

        _p3x  = _p1x + 2 * _inlineCellVectorX * directionVector.x;
        _p3y  = _p1y + 2 * _inlineCellVectorY * directionVector.y;
    }

    if( _crosslineSteps < 2 )
    {
        // Adiciona dois elementos ao grid da crossline
    	_crosslineSteps = 2;

        //Calcula o vetor normalizado da direção crossline
        Vector3Dd lVector = Vector3Dd(p3x - p1x, p3y - p1y, 0);
        lVector.normalize();

        Vector3Dd timeVector = Vector3Dd(0, 0, 1);

        //Calcula o vetor normalizado da direção perpendicular a crossline (produto vetorial de xlVector pelo vetor do tempo (0,0,1))
        Vector3Dd directionVector = lVector.cross(timeVector);

        if( ( _p2x - _p1x == 0 ) && ( _p2y - _p1y ) == 0 )
        {
            _crosslineCellVectorX = _inlineCellVectorY;
            _crosslineCellVectorY = _inlineCellVectorX;
        }

        _p2x  = _p1x + 2 * _crosslineCellVectorX * directionVector.x;
        _p2y  = _p1y + 2 * _crosslineCellVectorY * directionVector.y;
    }

    // Inicializa as matrizes de transformação
    computeIndexToMapMatrix();
    computeMapToIndexMatrix();
}



void GridCoordConverter::gridToMap( const double inlineNumber, const double crosslineNumber, double& x, double& y )
{
	double inlineIndex, crosslineIndex;

	gridToIndex( inlineIndex, crosslineIndex, inlineNumber, crosslineNumber );

	float inlineX =  _inlineCellVectorX * inlineIndex;
	float crosslineX =  _crosslineCellVectorX * crosslineIndex;
	float inlineY = _inlineCellVectorY * inlineIndex;
	float crosslineY =  _crosslineCellVectorY * crosslineIndex;

	x = _p1x + inlineX + crosslineX;
	y = _p1y + inlineY + crosslineY;
}



void GridCoordConverter::mapToGrid( double& inlineNumber, double& crosslineNumber, const double x, const double y )
{
	double crosslineIndex, inlineIndex;
	mapToIndex( x, y, inlineIndex, crosslineIndex );

	indexToGrid( inlineIndex, crosslineIndex, inlineNumber, crosslineNumber );
}



void GridCoordConverter::indexToGrid( const double inlineIndex, const double crosslineIndex, double& inlineNumber, double& crosslineNumber )
{
    crosslineNumber = _startCrossline + ( _crosslineIncrement * crosslineIndex );
    inlineNumber = _startInline + ( _inlineIncrement * inlineIndex );
}



void GridCoordConverter::gridToIndex( double& inlineIndex, double& crosslineIndex, const double inlineNumber, const double crosslineNumber )
{
	double firstInline = (double)_startInline;
	double firstCrossline = (double)_startCrossline;

	double incrementInline = (double)_inlineIncrement;
	double incrementCrossline = (double)_crosslineIncrement;

	inlineIndex = ( inlineNumber - firstInline )/incrementInline;
	crosslineIndex = ( crosslineNumber - firstCrossline )/incrementCrossline;
}



void GridCoordConverter::mapToIndex( const double x, const double y, double& inlineIndex, double& crosslineIndex )
{
    // Translada o sistema para a origem
    double xCoord = x - _p1x;
    double yCoord = y - _p1y;

    // Tranforma o ponto para o sistema de indice
    transform( xCoord, yCoord, _mapToIndexMatrix, crosslineIndex, inlineIndex );
}



void GridCoordConverter::indexToMap( double& x, double& y, const double inlineIndex, const double crosslineIndex )
{
    x = _p1x + _crosslineCellVectorX * crosslineIndex + _inlineCellVectorX * inlineIndex;
    y = _p1y + _crosslineCellVectorY * crosslineIndex + _inlineCellVectorY * inlineIndex;
}


bool GridCoordConverter::invertMatrix2x2( const double m11, const double m12, const double m21, const double m22, double& im11, double& im12, double& im21, double& im22 )
{
	double ad = m11 * m22;
	double bc = m12 * m21;
	double factor = ad - bc;

	if ( fabs(factor) < 1e-15 )
	{
		return false;
	}

	double scalar = 1.0/factor;

	im11 = scalar * m22;
	im12 = scalar * ( - m12 );
	im21 = scalar * ( - m21 );
	im22 = scalar * m11;

	return true;
}


bool GridCoordConverter::invertMatrix2x2( const Matrix2x2& matrix, Matrix2x2& inverseMatrix )
{
    double matrixDet = matrix._m11 * matrix._m22 - matrix._m12 * matrix._m21 ;

    // Verifica se a matriz pode ser invertida
    if ( fabs(matrixDet) < 1e-15 )
        return false;

    double  inversetMatrixDet = 1.0 / matrixDet;

    // Computa a matriz inversa.
    inverseMatrix._m11 =  inversetMatrixDet * matrix._m22;
    inverseMatrix._m21 = -inversetMatrixDet * matrix._m21;
    inverseMatrix._m12 = -inversetMatrixDet * matrix._m12;
    inverseMatrix._m22 =  inversetMatrixDet * matrix._m11;

    return true;
}



void GridCoordConverter::transform( const double inX, const double inY, const Matrix2x2& matrix, double& outX, double& outY )
{
    outX = matrix._m11 * inX + matrix._m21 * inY;
    outY = matrix._m12 * inX + matrix._m22 * inY;
}



void GridCoordConverter::computeIndexToMapMatrix()
{
    _indexToMapMatrix._m11 = _crosslineCellVectorX;
    _indexToMapMatrix._m12 = _crosslineCellVectorY;
    _indexToMapMatrix._m21 = _inlineCellVectorX;
    _indexToMapMatrix._m22 = _inlineCellVectorY;
}


void GridCoordConverter::computeMapToIndexMatrix()
{  
    // Esta matriz sera a inversa da gridToMap caso ela seja inversível.
    if ( invertMatrix2x2( _indexToMapMatrix, _mapToIndexMatrix ) == false)
    {
        // Se falhar utiliza uma matriz identidade.
        _mapToIndexMatrix._m11 = 1.0;
        _mapToIndexMatrix._m12 = 0.0;
        _mapToIndexMatrix._m21 = 0.0;
        _mapToIndexMatrix._m22 = 1.0;
    }
}

}
