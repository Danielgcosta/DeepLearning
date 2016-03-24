//-----------------------------------------------------------------------------
/**
 * @file GridCoordinateConverter.h
 * Declaracoes da classe GridCoordinateConverter.
 *
 * @author Luciana (lalmeida@tecgraf.puc-rio.br)
 * @date 12/08/2010
 */
//-----------------------------------------------------------------------------

#ifndef GRID_COORDINATE_CONVERTER_H
#define GRID_COORDINATE_CONVERTER_H

//-----------------------------------------------------------------------------

namespace MSA{

class Matrix2x2
{
public:

    /* Construtor
     * @param[in]  m11  elemento na linha 1 coluna 1 da matriz 
     * @param[in]  m12  elemento na linha 1 coluna 2 da matriz 
     * @param[in]  m21  elemento na linha 2 coluna 1 da matriz 
     * @param[in]  m22  elemento na linha 2 coluna 2 da matriz 
     */
    Matrix2x2( const double m11, const double m12, const double m21, const double m22 )
    {
        _m11 = m11;
        _m21 = m21;
        _m12 = m12;
        _m22 = m22;
    }

    /**
     * Construtor padrão.
     */
    Matrix2x2()
    {
        _m11 = 1.0;
        _m21 = 0.0;
        _m12 = 0.0;
        _m22 = 1.0;
    }

    /**
     * Operador de igualdade.
     */
    Matrix2x2& operator=( const Matrix2x2& matrix )
    {
        _m11 = matrix._m11;
        _m21 = matrix._m21;
        _m12 = matrix._m12;
        _m22 = matrix._m22;
        
		return *this;
    }

    /**
     * Destrutor
     */
    ~Matrix2x2()
    {
    }

    /** Elemento na linha 1 coluna 1 da matriz */
    double _m11;

    /** Elemento na linha 1 coluna 2 da matriz */
    double _m12;

    /** Elemento na linha 2 coluna 1 da matriz */
    double _m21;

    /** Elemento na linha 2 coluna 2 da matriz */
    double _m22;
};

//-----------------------------------------------------------------------------

/** @class GridCoordinateConverter
 * @brief Esta classe é responsável pela conversão entre 
 * os sistemas de coordenadas: INDEX(IJK), GRID e MAPA.
 */
class GridCoordConverter
{
    protected:
	/**
 	 * Construtor padrao.
	 */
	GridCoordConverter() {};

public:
    
    /**
     * Construtor.
     */
    GridCoordConverter( int startInline, int inlineIncrement, int inlineSteps, int startCrossline, int crosslineIncrement, int crosslineSteps, double p1x, double p1y, double p2x, double p2y, double p3x, double p3y );
	
    /** 
     * Construtor de cópia 
     */
    GridCoordConverter( const GridCoordConverter& orig );
    
    /**
     * Destrutor.
     */
    ~GridCoordConverter();

    /**
     * Converte de coordenada GRID para MAPA.
     * @param[out] inlineNumber Número da inline do sistema GRID.
     * @param[out] crosslineNumber Número da crossline do sistema GRID.
     * @param[out] x Valor de x do sistema MAPA.
	 * @param[out] y Valor de y do sistema MAPA.
     */
     void gridToMap( const double inlineNumber, const double crosslineNumber, double& x, double& y );

	 /**
     * Converte de coordenada MAPA para GRID.
     * @param[out] inlineNumber Valor da inline do sistema GRID.
     * @param[out] crosslineNumber valor da crossline do sistema GRID.
     * @param[out] x Valor da coordenada x do sistema MAPA.
	 * @param[out] y Valor da coordenada y do sistema MAPA.
     */
     void mapToGrid( double& inlineNumber, double& crosslineNumber, const double x, const double y );

	 /**
     * Converte de coordenada INDEX(IJK) para GRID.
     * @param[out] inlineIndex Valor da inline do sistema INDEX(IJK).
     * @param[out] crosslineIndex Valor da crossline do sistema INDEX(IJK).
     * @param[out] inlineNumber Valor da inline do sistema GRID.
	 * @param[out] crosslineNumber Valor da crossline do sistema GRID
     */
     void indexToGrid( const double inlineIndex, const double crosslineIndex, double& inlineNumber, double& crosslineNumber );

	 /**
     * Converte de coordenada GRID para INDEX(IJK).
     * @param[out] inlineIndex Valor da inline do sistema INDEX(IJK).
     * @param[out] crosslineIndex Valor da crossline do sistema INDEX(IJK).
     * @param[out] inlineNumber Valor da inline do sistema GRID.
	 * @param[out] crosslineNumber Valor da crossline do sistema GRID
     */
     void gridToIndex( double& inlineIndex, double& crosslineIndex, const double inlineNumber, const double crosslineNumber );

 	 /**
     * Converte de coordenada MAPA para INDEX(IJK).
     * @param[out] x Valor da coordenada x do sistema MAPA.
	 * @param[out] y Valor da coordenada y do sistema MAPA.
     * @param[out] inlineIndex Valor da inline do sistema INDEX(IJK).
     * @param[out] crosslineIndex Valor da crossline do sistema INDEX(IJK).
     */
     void mapToIndex( const double x, const double y, double& inlineIndex, double& crosslineIndex );

	 /**
     * Converte de coordenada INDEX(IJK) para MAPA.
     * @param[out] x Valor da coordenada x do sistema MAPA.
	 * @param[out] y Valor da coordenada y do sistema MAPA.
     * @param[out] inlineIndex Valor da inline do sistema INDEX(IJK).
     * @param[out] crosslineIndex Valor da crossline do sistema INDEX(IJK).
     */
     void indexToMap( double& x, double& y, const double inlineIndex, const double crosslineIndex );

 	 /**
     * Calcula a inversa de uma matriz 2x2
     * @param[in]  m11  elemento na linha 1 coluna 1 da matriz a ser invertida
     * @param[in]  m12  elemento na linha 1 coluna 2 da matriz a ser invertida
     * @param[in]  m21  elemento na linha 2 coluna 1 da matriz a ser invertida
     * @param[in]  m22  elemento na linha 2 coluna 2 da matriz a ser invertida
     * @param[out] mi11 elemento na linha 1 coluna 1 da matriz invertida
     * @param[out] mi12 elemento na linha 1 coluna 1 da matriz invertida
     * @param[out] mi21 elemento na linha 1 coluna 1 da matriz invertida
     * @param[out] mi22 elemento na linha 1 coluna 1 da matriz invertida
     * @return retorna false caso a matriz nao possa ser invertida e true se ela tiver sido calculada com sucesso
     */
     bool invertMatrix2x2( const double m11, const double m12, const double m21, const double m22, double& im11, double& im12, double& im21, double& im22 );

     /**
     * Calcula a inversa de uma matriz 2x2
     * @param[in]  matrix Matriz a ser invertida.
     * @param[out] inverseMatrix Matriz a invertida que sera computada.
     * @return retorna false caso a matriz nao possa ser invertida e true se ela tiver sido calculada com sucesso
     */
     bool invertMatrix2x2( const Matrix2x2& matrix, Matrix2x2& inverseMatrix );

     /** 
      * Transforma um ponto pela matriz.
      */
     void transform( const double inX, const double inY, const Matrix2x2& matrix, double& outX, double& outY );

protected:

    /** Deixa precomputada a matriz para converter do sistema Index para o o de Mapa */
    void computeMapToIndexMatrix();

    /** Deixa precomputada a matriz para converter do sistema Mapa para o o de Index */
    void computeIndexToMapMatrix();
    
    void initialize( int startInline, int inlineIncrement, int inlineSteps, int startCrossline, int crosslineIncrement, int crosslineSteps, double p1x, double p1y, double p2x, double p2y, double p3x, double p3y );

	/** Guarda variáveis dos grids de inline e crossline para conversão */	
	/** Inicio da inline */
	int _startInline;
	
	/** Incremento da inline */
	int _inlineIncrement;
	
	/** Passos da inline */
	int _inlineSteps;

	/** Inicio da crossline */
	int _startCrossline;

	/** Incremento da crossline */
	int _crosslineIncrement;
	
	/** Passos da crossline */
	int _crosslineSteps;

    /** Coordenada x de p1 */
    double _p1x;

    /** Coordenada y de p1 */
    double _p1y;

	/** Coordenada x de p2 */
    double _p2x;

	/** Coordenada y de p2 */
    double _p2y;

	/** Coordenada x de p3 */
    double _p3x;

	/** Coordenada y de p3 */
    double _p3y;

    /** Coordenadas x e y do vetor unitário na direção da inline */
    double _inlineCellVectorX;
    double _inlineCellVectorY;

    /** Coordenadas x e y do vetor unitário na direção da crossline */
    double _crosslineCellVectorX;
    double _crosslineCellVectorY;

    /** Matriz pre computada para converter do sistema Index para o sistema Mapa */
    Matrix2x2 _indexToMapMatrix;

    /** Matriz pre computada para converter do sistema Mapa para o sistema Index */
    Matrix2x2 _mapToIndexMatrix;

    
};

}
//-----------------------------------------------------------------------------
#endif
