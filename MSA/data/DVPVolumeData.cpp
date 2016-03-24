
#include <limits>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>

#include "DVPVolumeData.h"
#include "SDKVolume.h"
#include "GridCoordConverter.h"

#include <iostream>
#include <iomanip>

namespace MSA
{

DVPVolumeData::DVPVolumeData( SDKVolume* volume )
{
    int numInline, numCrosslines, numTime;
    numInline = numCrosslines = numTime = 0;
    volume->getDimensoes( numInline, numCrosslines, numTime );
    
    InlineSize = numInline;
    CrosslineSize = numCrosslines;
    TimeSize = numTime;
 
    Vector3Df zInfo;
    volume->grade(
        InLineInf.x, InLineInf.y, InLineInf.z,
        CrossLineInf.x, CrossLineInf.y, CrossLineInf.z,
        zInfo.x, zInfo.y, zInfo.z
    );
    ZInf.x = (int) zInfo.x;
    ZInf.y = (int) zInfo.y;
    ZInf.z = (int) zInfo.z;
    
    std::cout << std::endl << "*** Construtor recebendo SDKVolume ***" << std::endl;
    std::cout << "Inline( inicial, final, incremento) - Num: (" << InLineInf.x << ", " <<  InLineInf.y << ", " << InLineInf.z << ") - " << numInline << std::endl;
    std::cout << "Crossline( inicial, final, incremento): (" << CrossLineInf.x << ", " <<  CrossLineInf.y << ", " << CrossLineInf.z << ") - " << numCrosslines << std::endl;
    std::cout << "Time( inicial, final, incremento): (" << ZInf.x << ", " <<  ZInf.y << ", " << ZInf.z << ") - " << numTime << std::endl;

    // Copia todos os dados do volume.
    float* voxelsDir;
    volume->getVoxels( &voxelsDir );
    m_Data = (void*) voxelsDir;

    double min, max;
    if( getMinMax( min, max ) )
    {
        MinAmpVoxel = min;
        MaxAmpVoxel = max;
    }

    volume->p1( P1.x, P1.y );
    volume->p2( P2.x, P2.y );
    volume->p3( P3.x, P3.y );
    
    std::cout << "P1("<< std::setprecision(15) << P1.x << ", " <<  P1.y << ")" << std::endl;
    std::cout << "P2("<< std::setprecision(15) << P2.x << ", " <<  P2.y << ")" << std::endl;    
    std::cout << "P3("<< std::setprecision(15) << P3.x << ", " <<  P3.y << ")" << std::endl;

    _fileName = std::string( volume->nome() );
    std::cout << "File Name: " << _fileName << std::endl;


    m_type = SoDataSet::FLOAT;
    m_dim = Vector3Di( InlineSize, CrosslineSize, TimeSize );

    m_GlobalPos = Vector3Di(0, 0, 0);
    _releaseVoxelsData = false;
    
    buildVolumeTransforms( );
    
    std::cout << "*** Fim Construtor DVPVolumeData ***" << std::endl;
}


DVPVolumeData::DVPVolumeData( DVPVolumeData* copy, float* voxelsBuffer, int defaultValue )
{
    this->InlineSize = copy->InlineSize;
    this->CrosslineSize = copy->CrosslineSize;
    this->TimeSize = copy->TimeSize;
    
    this->InLineInf = copy->InLineInf;
    this->CrossLineInf = copy->CrossLineInf;
    this->ZInf = copy->ZInf;
     
    // Copia todos os dados do volume.
    if( voxelsBuffer != 0 )
    {
        m_Data = (void*) voxelsBuffer;
    }
    else
    {
        int bufferSize = InlineSize * CrosslineSize * TimeSize;
        float* voxels = new float[bufferSize];
        //memset( voxels, defaultValue, sizeof(float) * bufferSize );
        std::fill_n( voxels, bufferSize, defaultValue);
        m_Data = (void*) voxels;
        
    }

    this->P1 = copy->P1;
    this->P2 = copy->P2;
    this->P3 = copy->P3;
    
    this->m_type = copy->m_type;
    this->m_dim = copy->m_dim;

    this->m_GlobalPos = copy->m_GlobalPos;
    this->_releaseVoxelsData = copy->_releaseVoxelsData;    
    
    _fileName = copy->_fileName;

    buildVolumeTransforms( );
    getMinMax( this->MinAmpVoxel, this->MaxAmpVoxel );
}


int DVPVolumeData::getVoxel( int Inline, int Crossline, int Time, void*&ReturnedVoxel )
{
	// Casos de retorno:
	if( ( Inline < 0 ) || ( (unsigned int)Inline >= InlineSize ) ||
			( Crossline < 0 ) || ( (unsigned int)Crossline >= CrosslineSize ) ||
			( Time < 0 ) || ( (unsigned int)Time >= TimeSize ) )
		return -1;
        
                
	int returnedIndex;
	returnedIndex = Inline * ( CrosslineSize * TimeSize ) +
			Crossline * TimeSize + ( TimeSize - Time - 1 );

	if( m_type == SoDataSet::UNSIGNED_BYTE )
	{
		unsigned char* dataVector = (unsigned char*) m_Data;
		ReturnedVoxel = &dataVector[returnedIndex];
	}

	if( m_type == SoDataSet::SIGNED_BYTE )
	{
		char* dataVector = (char*) m_Data;
		ReturnedVoxel = &dataVector[returnedIndex];
	}

	if( m_type == SoDataSet::UNSIGNED_SHORT )
	{
		unsigned short* dataVector = (unsigned short*) m_Data;
		ReturnedVoxel = &dataVector[returnedIndex];
	}

	if( m_type == SoDataSet::SIGNED_SHORT )
	{
		short* dataVector = (short*) m_Data;
		ReturnedVoxel = &dataVector[returnedIndex];
	}

	if( m_type == SoDataSet::FLOAT )
	{
		float* dataVector = (float*) m_Data;
		ReturnedVoxel = &dataVector[returnedIndex];
	}
	return 0;
}


/**
* Recebe a coordenada (0<=x<dimX, 0<=y<dimY, 0<=z<dimZ) de um suposto voxel e verifica se essa coordenada pertence ao volume.
* @return Retorna false em caso de erro, ou true caso esteja dentro das dimensoes do volume.
*/
bool DVPVolumeData::isValidCoord( int inLine, int crossLine, int time )
{
	if( (inLine<0) || (inLine>=m_dim.x) || (crossLine<0) || (crossLine>=m_dim.y) || (time<0) || (time>=m_dim.z) )
		return false;

	return true;
}


/**
  * Obtem o valor da coordenada global desse (possivel) sub-volume dentro do volume principal. Caso o valor de posicao global nao
  * seja inserido, a classe simplesmente admite a posicao global como sendo o ponto (0, 0, 0).
  */
void DVPVolumeData::getGlobalPos( Vector3Di& globalPos )
{
  // Retorna a posicao espacial global do volume caso seja recebida:
  globalPos = m_GlobalPos;
}


void DVPVolumeData::getDataChar( SoDataSet::DataType &type, Vector3Ds &dim )
{
	dim = Vector3Ds( (short)m_dim.x, (short)m_dim.y, (short)m_dim.z );
	type = m_type;
}


//SoVolumeReader::ReadError SoPetroVolumeData::getDataChar( SbBox3f &size, SoDataSet::DataType &type, Vector3Di &dim )
void DVPVolumeData::getDataChar( SoDataSet::DataType &type, Vector3Di &dim )
{
    dim = m_dim;
    type = m_type;
}

void DVPVolumeData::getDimensions( int& inlineSize, int& crosslineSize, int& domainSize )
{
    inlineSize = m_dim.x;
    crosslineSize = m_dim.y;
    domainSize = m_dim.z;
}

/**
* Transforma entre as duas formas de acessar um dado ( (x,y) -> traceNumber ).
*/
int DVPVolumeData::xyToTraceNumber( int x, int y, int& traceNumber )
{
  if( x < 0 || y < 0 || ( (unsigned int)x >= InlineSize ) || ( (unsigned int)y >= CrosslineSize ) )
	return -1;

  int traceInit = x * ( CrosslineSize * TimeSize ) + y * TimeSize;
  traceNumber = traceInit / TimeSize;

  return 0;
}


int DVPVolumeData::xyzToInternalPos( int x, int y, int z, int& internalPos )
{
	// Casos de retorno:
	if( x < 0 || y < 0 || z < 0 || ( (unsigned int)x >= InlineSize ) || ( (unsigned int)y >= CrosslineSize ) || ( (unsigned int)z >= TimeSize ) )
	  return -1;

	int voxelPosition;
	voxelPosition = x * ( CrosslineSize * TimeSize ) +
					y * TimeSize + ( TimeSize - z - 1 );

	internalPos = voxelPosition;
	return 0;
}


/**
  * Transforma a posicao no vetor de voxels do dado para coordenadas 3D.
  */
int DVPVolumeData::internalPosToXYZ( int& x, int& y, int& z, int internalPos )
{
	if( internalPos < 0 || (unsigned int)internalPos >= CrosslineSize * InlineSize * TimeSize )
	  return -1;

	int voxelPosition = internalPos;

	int sx = voxelPosition / ( CrosslineSize * TimeSize );
	int resto = voxelPosition % ( CrosslineSize * TimeSize );
	int sy = resto / TimeSize;
	// z = TimeSize - 1 - (resto % TimeSize);
	int sz = TimeSize - 1 - (resto % TimeSize);

	x = sx;
	y = sy;
	z = sz;

	return 0;
}


/**
* Gets the SEGY trace data of the specified trace  traceNumber.  traceNumber  can range from zero
* to the number of traces minus one. If you want to get a subset of the data, specify the  start
* position in the trace and the number of data values ( size ). The return value is the number of
* data values read, or -1, if the operation failed.
* No nosso caso, int traceInit = i * ( CrosslineSize * TimeSize ) + j * TimeSize;
*/
int DVPVolumeData::getSegyTraceData( int traceNumber, void *&traceData, int start, int size )
{
  if( traceNumber < 0 || (unsigned int)traceNumber >= CrosslineSize * InlineSize )
	return -1;

  //  0 < i < dimX;
  //  0 < j < dimY;
  /**
	 * para cada fatia andamos CrosslineSize tracos.
	 * dividindo traceNumber pela dimensio da fatia (que e CrosslineSize) encontramos
	 * em que fatia esta o traco requerido. O resto da divisio da quantos tracos temos
	 * que andar dentro dessa fatia.
	 */
  int i = traceNumber / CrosslineSize;
  int j = ( traceNumber % CrosslineSize );
  int traceInit = i * ( CrosslineSize * TimeSize ) + j * TimeSize;


  // Retornando ponteiro para o vetor:
  if( m_type == SoDataSet::FLOAT )
  {
	float* dataVector = (float*) m_Data;
	traceData = &dataVector[traceInit];
  }

  return 0;
}


/**
  * Calcula estatisticas dentro do dado. Retorna sua media e desvio padrao.
  * Disponivel somente para dados tipo FLOAT.
  */
void DVPVolumeData::statistics( double& mean, double& stdDev )
{
  mean = 0;
  stdDev = 0;

  if( (MinAmpVoxel == 0.0f) && (MaxAmpVoxel == 0.0f) )
  {
	  double min, max;
	  getMinMax( min, max );
	  MinAmpVoxel = min;
	  MaxAmpVoxel = max;
  }


  double ratio = (MaxAmpVoxel/10.0f);


  unsigned long bufferSize = (unsigned long)InlineSize * CrosslineSize * TimeSize;
  if( m_type == SoDataSet::FLOAT )
  {
	  double totalsum = 0;
	  float* Voxels = (float*)m_Data;
	  for( unsigned long pos = 0 ; pos < bufferSize ; pos++ )
		  totalsum += (Voxels[pos]/ratio);
	  mean = totalsum / bufferSize;

	  totalsum = 0;
	  for( unsigned long pos = 0 ; pos < bufferSize ; pos++ )
		  totalsum += (mean - (Voxels[pos]/ratio)) * (mean - (Voxels[pos]/ratio));
	  stdDev = totalsum / bufferSize;
	  stdDev = sqrt( stdDev );

	  stdDev = stdDev * ratio;
	  mean = mean * ratio;
  }
}


/**
  * Calcula estatisticas dentro do dado. Retorna sua media e desvio padrao.
  * Disponivel somente para dados tipo FLOAT.
  */
void DVPVolumeData::statistics( long double& mean, long double& stdDev )
{
	double dmean, dstdDev;
	statistics( dmean, dstdDev );

	mean = dmean;
	stdDev = dstdDev;
}


/**
  * Funcao que constroi a transformacao a ser sofrida pelo volume.
  * @return Retorna false caso o volume nao exista, ou true.
  */
bool DVPVolumeData::buildVolumeTransforms( )
{    
    int inlineSteps = (InLineInf.y - InLineInf.x) / InLineInf.z;

    int crosslineSteps = (CrossLineInf.y - CrossLineInf.x) / CrossLineInf.z;

    gridConverter = new MSA::GridCoordConverter( InLineInf.x, InLineInf.z, inlineSteps,
                                                 CrossLineInf.x, CrossLineInf.z, crosslineSteps,
                                                 P1.x, P1.y, P2.x, P2.y, P3.x, P3.y );

    return true;
}



Vector3Df DVPVolumeData::gridToGeoRef( Vector3Di entryVoxel )
{
	double x, y;

	gridConverter->indexToMap(x, y, (double)entryVoxel.x, (double)entryVoxel.y);

	Vector3Df entryVoxelfTransfGeoRef;
	entryVoxelfTransfGeoRef = Vector3Df(x, y, ZInf.x + entryVoxel.z * ZInf.z);

	return entryVoxelfTransfGeoRef;
}



Vector3Di DVPVolumeData::geoRefToGrid( Vector3Df entryVoxel )
{
	double inlineIndex;
	double crosslineIndex;

	gridConverter->mapToIndex( entryVoxel.x, entryVoxel.y, inlineIndex, crosslineIndex);

	double domainIndex = (entryVoxel.z - (double)ZInf.x) / (double)ZInf.z;

	return Vector3Di( Rounded(inlineIndex), Rounded(crosslineIndex), Rounded(domainIndex) );

}



/**
* Obtem o numero aproximado de picos existentes no dado (obtem o numero de picos do traco central).
*/
int DVPVolumeData::getEstimatedPeaks( )
{
	// Dados do volume sismico:
	SoDataSet::DataType type;
	Vector3Di dim;
	getDataChar( type, dim );

	// Pegando o traco do centro do volume:
	// Caso contrario, obtem o numero traco:
	int traceNumber;
	if( xyToTraceNumber( RandInt(1, (dim.x -2)), RandInt(1, (dim.y -2)), traceNumber ) == -1 )
		return -1;

	void* traceData = NULL;
	if( getSegyTraceData( traceNumber, traceData ) == -1 )
		return  -1;

	float* traceDataf = (float*) traceData;

	// Contando o numero de vezes em que o sinal do traco muda de polo:
	bool NegPosFlag = (traceDataf[0] > 0)? true : false;
	int cont = 0;

	for( int i=1 ; i<dim.z ; i++ )
	{
		bool thisNegPosFlag = (traceDataf[i] > 0)? true : false ;
		if( NegPosFlag != thisNegPosFlag )
		{
			cont++;
			NegPosFlag = thisNegPosFlag;
		}
	}
	return cont;
}


/**
* Analiza um determinado traco sismico e retorna o numero de picos e suas posicoes dentro do traco.
*/
int DVPVolumeData::getTracePeaks( int x, int y, std::vector<int>& negPeaks, std::vector<int>& posPeaks )
{
	// Dados do volume sismico:
	SoDataSet::DataType type;
	Vector3Di dim;
	getDataChar( type, dim );

	if( (x<0) || (x>=dim.x) || (y<0) || (y>=dim.y) )
		return -1;

	if( m_type != SoDataSet::FLOAT )
		return -1;

	// Pegando o traco do centro do volume:
	// Caso contrario, obtem o numero traco:
	int traceNumber;
	if( xyToTraceNumber( x, y, traceNumber ) == -1 )
		return -1;

	void* traceData = NULL;
	if( getSegyTraceData( traceNumber, traceData ) == -1 )
		return -1;

	float* traceDataf = (float*) traceData;

	// Contando o numero de vezes em que o sinal do traco muda de polo:
	bool NegPosFlag = (traceDataf[0] > 0)? true : false;

	int greaterAbsPos = 0;
	float value = fabs(traceDataf[greaterAbsPos]);

	for( int i=1 ; i<dim.z ; i++ )
	{
		bool thisNegPosFlag = (traceDataf[i] > 0)? true : false ;

		float thisValue = fabs(traceDataf[i]);

		// Enquanto as amostras forem do mesmo sinal, armazena a posicao de maior valor absoluto:
		if( NegPosFlag == thisNegPosFlag )
		{
			if( thisValue > value )
			{
				greaterAbsPos = i;
				value = thisValue;
			}
		}
		// Caso a amostra mude de sinal:
		if( NegPosFlag != thisNegPosFlag )
		{
			// Caso o flag indique amostras negativas, ou positivas, insere no vetor correpondente:
			if( NegPosFlag == true )  //< Amostras positivas:
				posPeaks.push_back( greaterAbsPos );
			if( NegPosFlag == false )  //< Amostras negativas:
				negPeaks.push_back( greaterAbsPos );

			// Muda o estado do flag:
			NegPosFlag = thisNegPosFlag;
			// Atualiza o valor e a posicao de value:
			value = thisValue;
			greaterAbsPos = i;
		}
	}
	return 0;
}


/**
* Analiza um determinado traco sismico e retorna as posicoes dentro do traco onde houveram mudancas de sinal
* (positivo => negativo e negativo => positivo).
*/
int DVPVolumeData::getTraceSignChanges( int x, int y, std::vector<int>& negToPos, std::vector<int>& posToNeg )
{
	// Dados do volume sismico:
	SoDataSet::DataType type;
	Vector3Di dim;
	getDataChar( type, dim );

	if( (x<0) || (x>=dim.x) || (y<0) || (y>=dim.y) )
		return -1;

	if( m_type != SoDataSet::FLOAT )
		return -1;

	// Pegando o traco do centro do volume:
	// Caso contrario, obtem o numero traco:
	int traceNumber;
	if( xyToTraceNumber( x, y, traceNumber ) == -1 )
		return -1;

	void* traceData = NULL;
	if( getSegyTraceData( traceNumber, traceData ) == -1 )
		return -1;

	float* traceDataf = (float*) traceData;

	// NegPosFlag armazena o sinal da amplitude do voxel atual:
	bool NegPosFlag = (traceDataf[0] > 0)? true : false;

//        int greaterAbsPos = 0;

	for( int i=1 ; i<dim.z ; i++ )
	{
		bool thisNegPosFlag = (traceDataf[i] > 0)? true : false ;

		// Caso a amostra mude de sinal:
		if( NegPosFlag != thisNegPosFlag )
		{
			// Caso o flag indique amostras negativas, ou positivas, insere no vetor correpondente:
			if( NegPosFlag == true )  //< Pos to Neg:
				posToNeg.push_back( i-1 );
			if( NegPosFlag == false )  //< Neg to Pos:
				negToPos.push_back( i-1 );

			// Muda o estado do flag:
			NegPosFlag = thisNegPosFlag;
		}
	}
	return 0;
}


/**
* Returns min max if stored in file for float data type.
*/
bool DVPVolumeData::getMinMax( float& min, float& max )
{
  int bufferSize = InlineSize * CrosslineSize * TimeSize;

  if( m_type != SoDataSet::FLOAT )
	  return -1;

  if( m_type == SoDataSet::FLOAT )
  {
	  // Valores de mimino e maximo:
	  float MIN = std::numeric_limits<float>::max( );
	  float MAX = std::numeric_limits<float>::min( );
	  float* dataVector = (float*) m_Data;
	  int cont = 0;
	  for(; cont < bufferSize; cont++ )
	  {
		  if( dataVector[cont] < MIN )
			  MIN = dataVector[cont];
		  if( dataVector[cont] > MAX )
			  MAX = dataVector[cont];
	  }
	  min = MIN;
	  max = MAX;
	  MinAmpVoxel = min;
	  MaxAmpVoxel = max;
	  return true;
  }

  min = 0.0;
  max = 0.0;
  return false;
}


/**
  * Returns min max if stored in file for double data type.
  */
bool DVPVolumeData::getMinMax( double& min, double& max )
{
  int bufferSize = InlineSize * CrosslineSize * TimeSize;

  if( m_type == SoDataSet::FLOAT )
  {
	// Valores de minino e maximo:
	float MIN = std::numeric_limits<float>::max( );
	float MAX = std::numeric_limits<float>::min( );
	float* dataVector = (float*) m_Data;
	for(int cont = 0; cont < bufferSize; cont++ )
	{
	  if( dataVector[cont] < MIN )
		MIN = dataVector[cont];
	  if( dataVector[cont] > MAX )
		MAX = dataVector[cont];
	}
	min = (double)MIN;
	max = (double)MAX;
	MinAmpVoxel = MIN;
	MaxAmpVoxel = MAX;
	return true;
  }

  min = 0.0;
  max = 0.0;
  return false;
}


/**
* Constructor.
*/
DVPVolumeData::DVPVolumeData():
        gridConverter( 0 ),
        _releaseVoxelsData(false)
{
  InlineSize = 0;
  CrosslineSize = 0;
  TimeSize = 0;
  m_Data = NULL;
  MinAmpVoxel = 0.0;
  MaxAmpVoxel = 0.0;
  InLineInf = Vector3Di( 0, 0, 0 );
  CrossLineInf = Vector3Di( 0, 0, 0 );
  ZInf = Vector3Di( 0, 0, 0 );
  m_type = SoDataSet::UNDEFINED;

  m_GlobalPos = Vector3Di(0, 0, 0);
}



/**
* Destrutor.
*/
DVPVolumeData::~DVPVolumeData( )
{
    if( m_Data != NULL && _releaseVoxelsData)
    {
            float* voxels = (float*)m_Data;
            delete[] voxels;
    }
}


/**
* Gets the Z range. This method returns the actual values from the file if the corresponding set method has not
* been called previously. Otherwise, the values previously specified with setZRange are returned.
*/
void DVPVolumeData::getZRange (int &ilInitial, int &xlInitial, int &increment)
{
    ilInitial = ZInf.x;
    xlInitial = ZInf.y;
    increment = ZInf.z;
}


void DVPVolumeData::getInlineRange (int &ilInitial, int &xlInitial, int &increment)
{
    ilInitial = InLineInf.x;
    xlInitial = InLineInf.y;
    increment = InLineInf.z;
}


void DVPVolumeData::getCrosslineRange (int &ilInitial, int &xlInitial, int &increment)
{
    ilInitial = CrossLineInf.x;
    xlInitial = CrossLineInf.y;
    increment = CrossLineInf.z;
}

/**
 * Insere as dimensoes do volume.
 */
void DVPVolumeData::setDimensions( int inlineSize, int crosslineSize, int timeSize )
{
	InlineSize = inlineSize;
	CrosslineSize = crosslineSize;
	TimeSize = timeSize;
}


bool DVPVolumeData::validateVoxels( SDKVolume* volume )
{
    for(size_t i = 0; i < 5; i++ )
    {
        for(size_t j = 0; j < 5; j++ )
        {
            for(size_t k = 0; k < 5; k++ )
            {
                void* retVoxel = NULL;
                this->getVoxel( i, j, k, retVoxel );
                float* retVoxelf = (float*) retVoxel;
                float dvpValue = retVoxelf[0];
                
                //std::cout << "(dvpValue : " << " (" << dvpValue << ")" << std::endl;
                
                double il, xl;
                gridConverter->indexToGrid( (double) i, (double) j, il, xl);
                std::cout << "(" << i << "," << j << "," << k << ")"; 
                std::cout << " - (" << il << "," << xl << "," << (ZInf.x + k) << ")" << std::endl;
                //std::cout << " - (" << il << "," << xl << "," << (ZInf.x + k) << ")" << std::endl;
                //float sdkValue = volume->getValor( (int) il, (int) xl, (ZInf.x + (ZInf.z*4)) );

                float sdkValue = (float) volume->getValor( InLineInf.x, CrossLineInf.x, ZInf.x );
                std::cout << " - (" << InLineInf.x << "," << CrossLineInf.x << "," << ZInf.x << ")" << std::endl;
                std::cout << "(dvpValue, sdkValue) : " << " (" << dvpValue << ", " << sdkValue << ")" << std::endl;
            }
        }
    }
    
    return true;
}


int DVPVolumeData::setVoxel( int Inline, int Crossline, int Time, void* Voxel )
{
    // Casos de retorno:
    if( ( Inline > InlineSize ) || ( Inline < 0 ) ||
            ( Crossline > CrosslineSize ) || ( Crossline < 0 ) ||
            ( Time > TimeSize ) || ( Time < 0 ) )
        return -1;

    int voxelPosition;
    voxelPosition = Inline * ( CrosslineSize * TimeSize ) +
            Crossline * TimeSize + ( TimeSize - Time - 1 );

    if( m_type == SoDataSet::FLOAT )
    {
        float* dataVector = (float*) m_Data;
        dataVector[voxelPosition] = ( (float*) Voxel )[0];
    }
    return 0;
}

std::string DVPVolumeData::getFileName()
{
    return _fileName;
}

}
