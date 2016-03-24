#ifndef DVPVOLUMEDATA_H
#define DVPVOLUMEDATA_H

#include "SoDataSet.h"
#include "utl.h"

#include "../../math/Vector.h"

#include <vector>
#include <string>

class SDKVolume;

namespace MSA
{
    
class GridCoordConverter;

class DVPVolumeData
{

public:

    /**
    * Constructor.
    */
    DVPVolumeData( );

    
    /**
    * Constructor recebendo um volume do v3o2.
    */
    DVPVolumeData( SDKVolume* volume );
    
    /**
     * Constructor de cópia.
     */
    DVPVolumeData( DVPVolumeData* copy, float* voxelsBuffer = 0, int defaultValue = -1 );

    /**
    * Destrutor.
    */
    ~DVPVolumeData( );
    
    /**
    * Recebe as tres coordenadas de um determinado voxel e retorna seu valor.
    * @param Inline Coordenada X do voxel.
    * @param Crossline Coordenada Y do voxel.
    * @param Time Coordenada Z do voxel.
    * @param ReturnedVoxel Valor retornado do voxel.
    * @return Retorna -1 em caso de erro (parametros, volume inexistente, etc) ou 0 caso ok.
    */
    int getVoxel( int Inline, int Crossline, int Time, void*&ReturnedVoxel );

    
    /**
    * Recebe as tres coordenadas de um determinado voxel e insere seu valor no vetor do volume.
    * @param Inline Coordenada X do voxel.
    * @param Crossline Coordenada Y do voxel.
    * @param Time Coordenada Z do voxel.
    * @param GNG_Voxel Valor inserido do voxel.
    * @return Retorna -1 em caso de erro (pariï¿œmetros, volume inexistente, etc) ou 0 caso ok.
    */
    int setVoxel( int Inline, int Crossline, int Time, void* Voxel );
    
    
    /**
    * Recebe a coordenada (0<=x<dimX, 0<=y<dimY, 0<=z<dimZ) de um suposto voxel e verifica se essa coordenada pertence ao volume.
    * @return Retorna false em caso de erro, ou true caso esteja dentro das dimensoes do volume.
    */
    bool isValidCoord( int inLine, int crossLine, int time );


    /**
    * Specifies the path of the file.
    * @param filename Filaname.
    * @return Returns 0 for success. Any other return value indicates failure.
    */
    //int setFilename( VolumeFloatV3o2* inputVolume );
        

    /**
      * Obtem o valor da coordenada global desse (possivel) sub-volume dentro do volume principal. Caso o valor de posicao global nao
      * seja inserido, a classe simplesmente admite a posicao global como sendo o ponto (0, 0, 0).
      */
    void getGlobalPos( Vector3Di& globalPos );


    /**
    * Gets the characteristics (file header) of the data volume.
    * @param type Type of the data.
    * @param dim Dimension of the data.
    * @return Return an error message, or ok.
    */
    void getDataChar( SoDataSet::DataType &type, Vector3Ds &dim );


    /**
    * Gets the characteristics (file header) of the data volume. See SoVolumeData.
    * @param type Type of the data.
    * @param dim Dimension of the data.
    * @return Return an error message, or ok.
    */
    void getDataChar( SoDataSet::DataType &type, Vector3Di &dim );

    /**
     * Retorna as dimensoes do dado
     */
    void getDimensions( int& inlineSize, int& crosslineSize, int& timeSize );


    /**
    * Transforma entre as duas formas de acessar um dado ( (x,y) -> traceNumber ).
    */
    int xyToTraceNumber( int x, int y, int& traceNumber );


    /**
      * Transforma de coordenadas 3D para a posicao no vetor de voxels do dado.
      */
    int xyzToInternalPos( int x, int y, int z, int& internalPos );


    /**
      * Transforma a posicao no vetor de voxels do dado para coordenadas 3D.
      */
    int internalPosToXYZ( int& x, int& y, int& z, int internalPos );


    /**
    * Gets the SEGY trace data of the specified trace  traceNumber.  traceNumber  can range from zero
    * to the number of traces minus one. If you want to get a subset of the data, specify the  start
    * position in the trace and the number of data values ( size ). The return value is the number of
    * data values read, or -1, if the operation failed.
    * No nosso caso, int traceInit = i * ( CrosslineSize * TimeSize ) + j * TimeSize;
    */
    int getSegyTraceData( int traceNumber, void *&traceData, int start = 0, int size = -1 );


    /**
      * Calcula estatisticas dentro do dado. Retorna sua media e desvio padrao.
      * Disponivel somente para dados tipo FLOAT.
      */
    void statistics( double& mean, double& stdDev );


    /**
      * Calcula estatisticas dentro do dado. Retorna sua media e desvio padrao.
      * Disponivel somente para dados tipo FLOAT.
      */
    void statistics( long double& mean, long double& stdDev );


    /**
      * Funcao que constroi a transformacao a ser sofrida pelo volume.
      * @return Retorna false caso o volume nao exista, ou true.
      */
    bool buildVolumeTransforms( );


    /**
     * Transforma de coordenadas de indíce para coordenadas de mapa.
     * @return
     */
    Vector3Df gridToGeoRef( Vector3Di entryVoxel );


    /**
     * Transforma de coordenadas de mapa para coordenadas de índice.
     * @return
     */
    Vector3Di geoRefToGrid( Vector3Df entryVoxel );


    /**
    * Obtem o numero aproximado de picos existentes no dado (obtem o numero de picos do traco central).
    */
    int getEstimatedPeaks( );


    /**
    * Analiza um determinado traco sismico e retorna o numero de picos e suas posicoes dentro do traco.
    */
    int getTracePeaks( int x, int y, std::vector<int>& negPeaks, std::vector<int>& posPeaks );


    /**
    * Analiza um determinado traco sismico e retorna as posicoes dentro do traco onde houveram mudancas de sinal
    * (positivo => negativo e negativo => positivo).
    */
    int getTraceSignChanges( int x, int y, std::vector<int>& negToPos, std::vector<int>& posToNeg );


    /**
    * Returns min max if stored in file for float data type.
    */
    bool getMinMax( float& min, float& max );


    /**
      * Returns min max if stored in file for double data type.
      */
    bool getMinMax( double& min, double& max );
    
    /**
    * Gets the Z range. This method returns the actual values from the file if the corresponding set method has not
    * been called previously. Otherwise, the values previously specified with setZRange are returned.
    */
    void getZRange (int &ilInitial, int &xlInitial, int &increment);

    void getInlineRange (int &ilInitial, int &xlInitial, int &increment);
    
    void getCrosslineRange (int &ilInitial, int &xlInitial, int &increment);
    
    /**
     * Insere as dimensoes do volume.
     */
    void setDimensions( int inlineSize, int crosslineSize, int timeSize );
        
    // Coordenadas georeferenciadas:
    Vector2Dd P1, P2, P3;
        
    /**
     * Recebe um volume SDK de entrada e compara voxel a voxel com o volume atual.
     * @return Retorna true caso todos os voxels sejam iguais.
     */
    bool validateVoxels( SDKVolume* volume );
    
    /**
     * Retorna o nome do arquivo.
     */
    std::string getFileName();
    
private:
    
    void* m_Data;
    double MaxAmpVoxel;    
    double MinAmpVoxel;

    unsigned int InlineSize;    
    unsigned int CrosslineSize;    
    unsigned int TimeSize;

    // InLine inicial, final, incremento.
    Vector3Di InLineInf;
    
    // CrossLine inicial, final, incremento.
    Vector3Di CrossLineInf;
    
    // Domínio inicial, final, incremento.
    Vector3Di ZInf;

    // String armazenando o nome do volume (incluindo o path):
    std::string _fileName;

    MSA::GridCoordConverter* gridConverter;

    Vector3Di m_dim;
    SoDataSet::DataType m_type;

    // Posicao global desse volume dentro do volume pai. Tera o valor (0, 0, 0) caso nao seja um filho:
    Vector3Di m_GlobalPos;

    /** Indica */
    bool _releaseVoxelsData;
};

}

#endif // DVPVOLUMEDATA_H
