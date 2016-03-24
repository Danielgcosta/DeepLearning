#ifndef SEISMICPROCESSINGCONTROLER_H
#define SEISMICPROCESSINGCONTROLER_H

#include <omp.h>
#include <cstdio>
#include <math.h>
#include <limits>
#include <list>
#include <vector>
#include <deque>
#include <map>
#include <time.h>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>


#include "../data/dipVolume.h"

#include "../data/utl.h"
#include "../data/stl_utils.h"
#include "../data/DVPVolumeData.h"
#include "../data/DataGrid3.h"
#include "../data/Volume_Smp.h"
#include "GNG_CPP.h"

#include "../../math/Vector.h"

namespace MSA
{

static int entrou = 0;
static int entrouEsaiu = 0;


/**
  * Classe de mapeamento de dados sismicos. A partir de uma sismica de entrada, divide o
  * volume de entrada em sub-volumes contendo partes que se sobrepoem. A classe entao cria
  * uma instancia de SeismicProcessinUnit para cada sub-volume, treinando instancias do algoritmo
  * de clusterizacao que sao treinados somente para essas porcoes menores do dado sismico original.
  */
class SeismicProcessingController
{
private:

  /**
    * Transforma de coordenadas 3D para a posicao no vetor de voxels do dado.
    */
  int xyzToInternalPos( int x, int y, int z, int& internalPos )
  {
    // Casos de retorno:
    if( ( x >= _spuEntryVolumeDim[0] ) || ( y >= _spuEntryVolumeDim[1] ) || ( z >= _spuEntryVolumeDim[2] ) )
      return -1;
    int voxelPosition;
    voxelPosition = x * ( _spuEntryVolumeDim[1] * _spuEntryVolumeDim[2] ) +
        y * _spuEntryVolumeDim[2] + ( _spuEntryVolumeDim[2] - z - 1 );
    internalPos = voxelPosition;
    return 0;
  }


  /**
    * Transforma a posicao no vetor de voxels do dado para coordenadas 3D.
    */
  int internalPosToXYZ( int& x, int& y, int& z, int internalPos )
  {
    if( internalPos >= _spuEntryVolumeDim[0]*_spuEntryVolumeDim[1]*_spuEntryVolumeDim[2] )
      return -1;
    int voxelPosition = internalPos;
    int sx = voxelPosition / ( _spuEntryVolumeDim[1] * _spuEntryVolumeDim[2] );
    int resto = voxelPosition % ( _spuEntryVolumeDim[1] * _spuEntryVolumeDim[2] );
    int sy = resto / _spuEntryVolumeDim[2];
    // z = _spuEntryVolumeDim[2] - 1 - (resto % _spuEntryVolumeDim[2]);
    int sz = _spuEntryVolumeDim[2] - 1 - (resto % _spuEntryVolumeDim[2]);
    x = sx;
    y = sy;
    z = sz;
    return 0;
  }


  /**
    * Funcao auxiliar a ser chamada depois do treinamento. Classifica todas as amostras
    * de treinamento para encontrar estatisticas importantes como medias e variancias.
    * Feito isso, insere esses valores em cada um dos clusters.
    * @param gngObject Instancia de GNG a ser utilizada (positiva ou negativa).
    */
  int classifySamples( std::vector<std::vector<float*> >& eachClusterSamplesVector, GNG_CPP* gngObject,
                       Volume_Smp* spuVolume_Smp )
  {
    // O ultimo passo consiste em calcular medias e desvios-padrao dos clusters em relacao Ã s suas amostras:
    int gCenterssize = gngObject->getNumClusters();
    eachClusterSamplesVector.reserve( gCenterssize );
    eachClusterSamplesVector.resize( gCenterssize );

    // Dados do volume sismico:
    SoDataSet::DataType type;
    Vector3Di dim;
    _spuEntryVolume->getDataChar( type, dim );


    for( int x=0 ; x<dim[0] ; x++ )
    {
      for( int y=0 ; y<dim[1] ; y++ )
      {
        for( int z=0 ; z<dim[2] ; z++ )
        {
          float centralValue;
          float* sample = NULL;
          sample = spuVolume_Smp->getSample( x, y, z, centralValue, _spuVolume_SmpType );
          if( sample != NULL )
          {
            // Encontra os dois clusters mais proximos da amostra:
            Cluster* s1 = NULL;
            Cluster* s2 = NULL;
            Cluster* s3 = NULL;
            float s1error, s2error, s3error;
            gngObject->getBMU( sample, s1, s2, s3, s1error, s2error, s3error );
            int clusterIndex;
            s1->getClusterId( clusterIndex );
            clusterIndex = gngObject->getClusterPosFromId( clusterIndex );
            eachClusterSamplesVector[clusterIndex].push_back( sample );
          }
        }
      }
    }

    // Otimiza a utilizacao de memoria dos vetores:
    for( unsigned int i2=0 ; i2< eachClusterSamplesVector.size() ; i2++ )
      vectorTrim( eachClusterSamplesVector[i2] );

    return 0;
  }



public:

  // Define o tipo de eixo de procura.
  enum AxisType
  {
    AXIS_X=0,   //< Procura ao longo do eixo x.
    AXIS_Y      //< Procura ao longo do eixo y.
  };

  // Tamanho das amostras de entrada:
  int _spuSamplesSize;

  // Parametros a serem utilizados para configuracao das instancias de GNG:
  int _spuNneurons;
  int _spuLambda;
  float _spuEb;
  float _spuEn;
  float _spuAlpha;
  float _spuD;
  int _spuAmax;

  float _spuNneuronsMulti;

  DVPVolumeData* _spuEntryVolume;     //< Volume de entrada.
  char* _spuEntryVolumeName;
  float _spuMinStdDevForTrain;
  Volume_Smp* _spuVolume_Smp;
  Volume_Smp::SamplesType _spuVolume_SmpType;
  GNG_CPP* _spuGng_Cpp;

  // Dados do volume sismico:
  SoDataSet::DataType _spuEntryVolumeType;
  Vector3Di _spuEntryVolumeDim;

  // Depois do processamento, podemos encontrar tudas as sub-superficies que possuem voxels de um mesmo vizinho.
  // Esse processamento, depois de executado, vai produzir um volume de saida que possui "pedacos" corretos de horizontes.
  // Esses pedacos serao salvos num volume especifico, o _spuOutRawProcessedVolume
  DVPVolumeData* _spuOutRawProcessedVolume;
  char* _spuOutRawProcessedVolumeName;
  Volume_Smp* _spuOutRawProcessedVolume_Smp;        //< Instancia de Volume_Smp que permite encontrar horizontes.

  // Matrizes de distancias e de distancias relativas:
  std::vector< std::vector<float> > _spuDistancesMatrix;
  std::vector< std::vector<float> > _spuRelDistancesMatrix;

  // Criando dois vetores temporarios para armazenar cada um dos erros medios dos clusters:
  std::vector< int > _spuGng_Cpp_ClustersNumSamples;
  std::vector< long double > _spuGng_Cpp_ClustersTotalError;

  std::vector<std::vector<std::vector<short> > > _spuCvCVolume;



    /**
     * Inicializa o volume utilizado para verificacao de continuidade x conflitos.
     */
    int initCvCVolume()
    {
    	// Alocando a memoria para CvCVolume:
    	_spuCvCVolume.resize(_spuEntryVolumeDim.x);
    	for (int i = 0; i < _spuEntryVolumeDim.x; ++i)
    	{
    		_spuCvCVolume[i].resize(_spuEntryVolumeDim.y);
    		for (int j = 0; j < _spuEntryVolumeDim.y; ++j)
    		{
    			_spuCvCVolume[i][j].resize(_spuEntryVolumeDim.z);
    			for (int k = 0; k < _spuEntryVolumeDim.z; k++)
    			{
    				_spuCvCVolume[i][j][k] = -1;
    			}
    		}
    	}

	    int x, y, z;
		for (x = 0; x < _spuEntryVolumeDim.x; x++)
		{
			for (y = 0; y < _spuEntryVolumeDim.y; y++)
			{
				for (z = 0; z < _spuEntryVolumeDim.z; z++)
				{
					float centralAmpVoxel = 0;
					float* thisSample = _spuVolume_Smp->getSample_( x, y, z, centralAmpVoxel, Volume_Smp::NEGATIVE_PEAKS );
					if (thisSample != NULL)
					{
						int internalPos = -1;
						_spuEntryVolume->xyzToInternalPos( x, y, z, internalPos );
					}
				}
			}
		}

    }


  /**
    * Construtor.
    */
  SeismicProcessingController( )
  {
    _spuEntryVolume = NULL;
    _spuEntryVolumeName = NULL;
    _spuOutRawProcessedVolume = NULL;
    _spuOutRawProcessedVolumeName = NULL;
    _spuOutRawProcessedVolume_Smp = NULL;

    _spuMinStdDevForTrain = 0.0f;
    _spuVolume_Smp = NULL;
    _spuVolume_SmpType = Volume_Smp::UNDEFINED;
    _spuGng_Cpp = NULL;

    // Para o caso geral, todas as amostras serao utilizadas no treinamento:
    _spuVolume_SmpType = Volume_Smp::UNDEFINED;
  }


  /**
   * Configura parametros de configuracao.
   */
  void setParameters( int spuLambda, float spuEb, float spuEn, float spuAlpha, float spuD, int spuAmax,
		  int spuSamplesSize, float spuNneuronsMulti, float spuMinStdDevForTrain )
  {
	    // Demais parametros de treinamento:
	    _spuLambda      	= spuLambda;
	    _spuEb          	= spuEb;
	    _spuEn          	= spuEn;
	    _spuAlpha       	= spuAlpha;
	    _spuD           	= spuD;
	    _spuAmax        	= spuAmax;
	    _spuSamplesSize		= spuSamplesSize;
	    _spuNneuronsMulti	= spuNneuronsMulti;
	    _spuMinStdDevForTrain = spuMinStdDevForTrain;
  }


  /**
    * Cria as matrizes de distancias e distancias relativas para todos os clusters.
    * Sao retornadas distancia euclideana e distancia relativa.
    */
  int createDistancesMatrices( bool isTrainnig = false )
  {
    if( _spuGng_Cpp == NULL )
      return -1;
    int numClusters = _spuGng_Cpp->getNumClusters();
    if( numClusters == 0 )
      return -1;

    _spuDistancesMatrix.resize   ( numClusters );
    _spuRelDistancesMatrix.resize( numClusters );
    for (int i = 0; i < numClusters; i++)
    {
      _spuDistancesMatrix   [i].resize( numClusters );
      _spuRelDistancesMatrix[i].resize( numClusters );
      for (int j = 0; j < numClusters; j++)
      {
        _spuDistancesMatrix   [i][j] = std::numeric_limits<float>::max();
        _spuRelDistancesMatrix[i][j] = std::numeric_limits<float>::max();
      }
    }


    // Inserindo os valores de erro medio nos clusters:
    if( isTrainnig == true )
    {
      // Verificando mudancas na distancia relativa:
      std::cout << "Verificando mudancas na distancia relativa: " << std::endl;

      std::vector<Cluster*> clustersVetor = _spuGng_Cpp->getClustersVector();
      for (int i = 0; i < numClusters; i++)
      {
        int clusterIndex;
        Cluster* s1 = clustersVetor[i];
        s1->getClusterId( clusterIndex );
        clusterIndex = _spuGng_Cpp->getClusterPosFromId( clusterIndex );

        float thisRelDistance = ((float)(_spuGng_Cpp_ClustersTotalError[clusterIndex]) / (float)(_spuGng_Cpp_ClustersNumSamples[clusterIndex]));

//        float oldRelativeDistance;
//        s1->getRelativeDistance( oldRelativeDistance );
//        std::cout << "Cluster: " << i << " ." << "antiga: " << oldRelativeDistance << ", nova: " << thisRelDistance << std::endl;
        s1->setRelativeDistance( thisRelDistance );
      }
    }


    // Obtendo o vetor de clusters da instancia:
    for (int i = 0; i < numClusters; i++)
    {
      Cluster* p1s1 = NULL;
      int p1ClusterIndex = _spuGng_Cpp->getClusterPosFromId( i );
      p1s1 = _spuGng_Cpp->getClusterFromClusterId( p1ClusterIndex );

      for (int j = 0; j < numClusters; j++)
      {
        Cluster* p2s1 = NULL;
        int p2ClusterIndex = _spuGng_Cpp->getClusterPosFromId( j );
        p2s1 = _spuGng_Cpp->getClusterFromClusterId( p2ClusterIndex );

        float* p1Pos = p1s1->_clCenter;
        float p1RelDist;
        p1s1->getRelativeDistance( p1RelDist );
        float* p2Pos = p2s1->_clCenter;

        float distRet = _spuGng_Cpp->fEuclideanDist( p1Pos, p2Pos, _spuGng_Cpp->_gDim );
        float relDistRet = distRet / p1RelDist;
        _spuDistancesMatrix   [i][j] = distRet;
        _spuRelDistancesMatrix[i][j] = relDistRet;
      }
    }
    return 0;
  }


  /**
    * Encontra um volume de falhas utilizando para isso o desvio padrao das estimativas das
    * amostras ao longo do proprio traco.
    */
  int createMeansVolume()
  {
    // Criando o volume de saida:
    DVPVolumeData* _spuMeanFaultVolume = new DVPVolumeData( _spuEntryVolume, 0, 0 );

    // CASO 1: Utiliza o volume de labels para estimar as falhas a partir do desvio padrao das estimativas das amostras:
    std::cout << "Iniciando createStdDevFaultVolume:" << std::endl;
    int samplesSize = _spuGng_Cpp->getSamplesSize();

    for (int x = 0; x < _spuEntryVolumeDim[0] ; x++)
    {
      std::cout << "Step: " << x << " de " << _spuEntryVolumeDim[0] << std::endl;
      for (int y = 0; y < _spuEntryVolumeDim[1]; y++)
      {
        for (int z = samplesSize+1; z < _spuEntryVolumeDim[2]-samplesSize-1; z++)
        {
          // Obtem o voxel de entrada no volume de labels:
          void* retVoxel = NULL;
          _spuOutRawProcessedVolume->getVoxel( x, y, z, retVoxel );
          float* retVoxelf = (float*)retVoxel;
          float clusterIndexf = retVoxelf[0];
          int clusterIndex = Rounded( clusterIndexf );

          if( clusterIndex == -1 )
            continue;

          clusterIndex = _spuGng_Cpp->getClusterPosFromId(clusterIndex);
          Cluster* cluster = _spuGng_Cpp->getClusterFromClusterId( clusterIndex );
          float* clusterCenter = NULL;
          int clusterSize = -1;
          cluster->getPos( clusterSize, clusterCenter );
          int k, pos;
          for( k = z-(clusterSize/2), pos=0 ; k<(z+(clusterSize/2)) ; k++, pos++ )
          {
            retVoxel = NULL;
            _spuEntryVolume->getVoxel(x, y, _spuEntryVolumeDim[2] - k - 1, retVoxel);
            retVoxelf = (float*) retVoxel;
            float voxel = retVoxelf[0];

            // Obtem o valor atual do voxel de sa�da:
            retVoxel = NULL;
            _spuMeanFaultVolume->getVoxel(x, y, _spuEntryVolumeDim[2] - k - 1, retVoxel);
            retVoxelf = (float*) retVoxel;
            voxel = retVoxelf[0];
            // Soma o valor atual ao novo valor:
            voxel += clusterCenter[pos];

            // Insere o valor desse voxel no volume de saida:
            _spuMeanFaultVolume->setVoxel(x, y, _spuEntryVolumeDim[2] - k - 1, (void*) (&(voxel)) );
          }
        }
      }
    }
    // Depois do processamento, temos que dividir pelo numero de amostras e tirar raiz quadrada:
    for (int x = 0; x < _spuEntryVolumeDim[0]; x++)
    {
      for (int y = 0; y < _spuEntryVolumeDim[1]; y++)
      {
        for (int z = 0; z < _spuEntryVolumeDim[2]; z++)
        {
          // Obtem o voxel:
          void* retVoxel = NULL;
          _spuMeanFaultVolume->getVoxel( x, y, z, retVoxel );
          float* retVoxelf = (float*)retVoxel;
          float clusterIndexf = retVoxelf[0];
          clusterIndexf = clusterIndexf / samplesSize;

          // Insere o valor desse voxel no volume de saida:
          _spuMeanFaultVolume->setVoxel(x, y, z, (void*) (&(clusterIndexf )) );
        }
      }
    }
    
    return 0;
  }

  /**
   * Completa o treinamento
   */
  int Train()
  {
	    // Dados especificos vindos do volume de entrada (dimensoes, tipo de voxel, posicao espacial 3D):
	    _spuEntryVolume->getDataChar( _spuEntryVolumeType, _spuEntryVolumeDim );

	    // Obtem o numero aproximado de neuronios para o treinamento da instancia referente ao sub-volume:
	    int nPeaks = _spuEntryVolume->getEstimatedPeaks(); //< 32 para F3_Hale
	    std::cout << "nPeaks: " << nPeaks << std::endl;

	    // O numero de neuronios de treinamento tera um componente aleatorio:
	    _spuNneurons = nPeaks * 2;
	    _spuNneurons = Rounded(_spuNneurons * _spuNneuronsMulti);

	    _spuNneurons = 45;
                        
	    std::cout << "_spuNneurons: " << _spuNneurons << std::endl;

	    // Cria as amostras que serao treinadas:
	    if( _spuEntryVolumeType != SoDataSet::FLOAT )
	      return -1;

	    // Cria a instancia da classe que permite a criacao de quaisquer amostras:
	    _spuVolume_Smp = new Volume_Smp( _spuEntryVolume, _spuSamplesSize, Vector3Di(0, 0, 0), _spuMinStdDevForTrain );

	    // Cria a instancia de GNG:
	    _spuGng_Cpp = new GNG_CPP( _spuNneurons, _spuSamplesSize );

	    // Numero de treinamentos executados (a ser retornado pela funcao TrainGNG):
	    int numTrainsRet = 0 ;

	    // Imprimindo o instante de inicio de treinamento:
	    std::cout << "Treinando novo GNG: " << _spuSamplesSize << " valores, e " << _spuNneurons << " neuronios. " << std::endl;

	    // Treinando a instancia ate atingir o numero de neuronios desejado:
	    _spuGng_Cpp->allSamplesTrain( _spuVolume_Smp, _spuLambda, _spuEb, _spuEn, _spuAlpha, _spuAmax, _spuD, numTrainsRet, 5, _spuVolume_SmpType );

	    // Executa mais 20*lambda treinamentos na instancia (com taxas de aprendizado menores), sem a inclusao de novos neuronios (minimizacao dos erros):
	    _spuGng_Cpp->allSamplesTrain( _spuVolume_Smp, 5*_spuLambda, _spuEb/5.0f, _spuEn/5.0f, _spuAlpha, _spuAmax, _spuD, numTrainsRet, -1, _spuVolume_SmpType );

	    // Chama a funcao que finaliza o  treinamento das instancias:
	    // Cria a matriz numerica de arestas, ordena o vetor de vertices (), classifica o vetor de amostras recebido nos clusters
	    // formados, e com isso calcula estata�sticas importantes, tais como media, variancia, matriz de covariancias entre os
	    // componentes do cluster, entre outras informacoes.  Caso o vetor de amostras seja vazio, somente ordena os nos e
	    // cria as matrizes de arestas e de distancia por arestas (sem calcular as estata�sticas).
	    std::vector<float*>samplesEMPTY;                                    //< NAO ESTA SENDO USADO.
	    std::vector<std::vector<float*> > classifiedSamplesVector;          //< NAO ESTA SENDO USADO.
	    std::vector<std::vector<int> > edgesMatrix;                         //< NAO ESTA SENDO USADO.
	    _spuGng_Cpp->finalizeTrain( _spuVolume_Smp, 1000000, edgesMatrix, classifiedSamplesVector );


	    // Limpa o vetor de arestas (esse vetor e apenas uma copia do vetor de arestas interno da classe):
	    vectorFreeMemory( edgesMatrix );

	    // Classificando as amostras da instancia. "eachClusterSamplesVector" possuira em cada posicao as amostras classificadas no seu cluster.
	    // Ex: eachClusterSamplesVector[0] contera todas as amostras classificadas no cluster de id=0 da instancia de GNG:
	    // Temporariamente nao encontratemos medias e variancias:
	    //        classifySamples( _spuGNG_InfNode->eachClusterSamplesVector, _spuGng_Cpp, _spuVolume_Smp );
	    //        _spuGng_Cpp->getStatisticsFromVector( _spuGNG_InfNode->eachClusterSamplesVector );


	    // Criando o volume de saida:
	    _spuOutRawProcessedVolume = new DVPVolumeData( _spuEntryVolume, 0, -1 );
	    
	    // Classifica todas as amostras do volume:
	    createOutRawProcessedVolume( true );

	    // Cria as matrizes de distancias e de distancias relativas, a partir dos clusters da instancia de GNG:
	    if( createDistancesMatrices( true ) == -1 )
	      return -1;

	    return 0;
  }


  /**
    *
    */
   int Train( DVPVolumeData* spuEntryVolume, const char* spuOutRawProcessedVolumeName )
   {
 	    // Nome do volume de entrada:
 	    if( _spuEntryVolumeName != NULL )
 	    {
 	      int numCharEntryVolume = strlen(spuEntryVolume->getFileName().c_str());
 	      _spuEntryVolumeName = new char[numCharEntryVolume];
 	      strcpy(_spuEntryVolumeName, spuEntryVolume->getFileName().c_str());
 	    }

 	    // Nome do volume processado:
 	    if( spuOutRawProcessedVolumeName != NULL )
 	    {
 	      int numCharEntryVolume = strlen(spuOutRawProcessedVolumeName);
 	      _spuOutRawProcessedVolumeName = new char[numCharEntryVolume];
 	      strcpy(_spuOutRawProcessedVolumeName, spuOutRawProcessedVolumeName);
 	    }

 	    // Carregando o volume sismico de entrada:
 	    _spuEntryVolume = spuEntryVolume;

 	    // Executa o treinamento:
 	    return Train();
   }


  /**
    * Funcao que permite criar o volume de saa�da, classificando as amostras nos clusters.
    */
  int createOutRawProcessedVolumeThreadStep( int minX, int maxX, int& perc )
  {
    for( int x=minX ; x<maxX ; x++ )
    {
      // Incrementa o contador que permite prever quando o processo se encerra:
      perc++;
      if( (perc % 20) == 0 )
        std::cout << "Perc: " << perc << std::endl;

      for( int y=0 ; y<_spuEntryVolumeDim[1] ; y++ )
      {
        // Amostras com a�ndices onde amplitudes mudam de sinal (positivo => negativo):
        for( int z=0 ; z<_spuEntryVolumeDim[2] ; z++ )
        {
          float centralValue;
          float* sample = NULL;
          sample = _spuVolume_Smp->getSample( x, y, z, centralValue, _spuVolume_SmpType );
          if( sample != NULL )
          {
            // Encontra os dois clusters mais proximos da amostra:
            Cluster* s1 = NULL;
            Cluster* s2 = NULL;
            Cluster* s3 = NULL;
            float s1error, s2error, s3error;
            _spuGng_Cpp->getBMU( sample, s1, s2, s3, s1error, s2error, s3error );
            int clusterIndex;
            s1->getClusterId( clusterIndex );
            clusterIndex = _spuGng_Cpp->getClusterPosFromId( clusterIndex );

            // Soma o numero de amostras e o erro:
            _spuGng_Cpp_ClustersNumSamples[clusterIndex] = _spuGng_Cpp_ClustersNumSamples[clusterIndex] + 1;
            _spuGng_Cpp_ClustersTotalError[clusterIndex] = _spuGng_Cpp_ClustersTotalError[clusterIndex] + s1error;


            // Insere o valor desse voxel no volume de saa�da:
            float clusterIndexf = (float)clusterIndex;
            //                        if( _spuOutRawProcessedVolume->setVoxel( x, y, _spuEntryVolumeDim[2]-z-1, (void*)(&(clusterIndexf)) ) == -1 )
            if( _spuOutRawProcessedVolume->setVoxel( x, y, z, (void*)(&(clusterIndexf)) ) == -1 )
              return -1;
          }
        }
      }
    }
    return 0;
  }


  /**
    * Funcao que permite criar o volume de saa�da, classificando as amostras nos clusters.
    */
  int createOutRawProcessedVolume( bool usingThreads = false )
  {
    // Medindo o tempo de processamento:
    time_t initTime;
    initTime = time (NULL);

    int numNeurons = _spuGng_Cpp->getNumClusters();
    _spuGng_Cpp_ClustersNumSamples.clear();
    _spuGng_Cpp_ClustersNumSamples.reserve( numNeurons );
    _spuGng_Cpp_ClustersNumSamples.resize( numNeurons );
    _spuGng_Cpp_ClustersTotalError.clear();
    _spuGng_Cpp_ClustersTotalError.reserve( numNeurons );
    _spuGng_Cpp_ClustersTotalError.resize( numNeurons );


    // Caso em que a classificacao nao sera feita em paralelo:
    if( usingThreads == false )
    {
      for( int x=0 ; x<_spuEntryVolumeDim[0] ; x++ )
      {
        std::cout << x << " de " << _spuEntryVolumeDim[0] << std::endl;
        for( int y=0 ; y<_spuEntryVolumeDim[1] ; y++ )
        {
          // Amostras com a�ndices onde amplitudes mudam de sinal (positivo => negativo):
          for( int z=0 ; z<_spuEntryVolumeDim[2] ; z++ )
          {
            float centralValue;
            float* sample = NULL;
            sample = _spuVolume_Smp->getSample( x, y, z, centralValue, _spuVolume_SmpType );
            if( sample != NULL )
            {
              // Encontra os dois clusters mais proximos da amostra:
              Cluster* s1 = NULL;
              Cluster* s2 = NULL;
              Cluster* s3 = NULL;
              float s1error, s2error, s3error;
              _spuGng_Cpp->getBMU( sample, s1, s2, s3, s1error, s2error, s3error );
              int clusterIndex;
              s1->getClusterId( clusterIndex );
              clusterIndex = _spuGng_Cpp->getClusterPosFromId( clusterIndex );

              // Soma o numero de amostras e o erro:
              _spuGng_Cpp_ClustersNumSamples[clusterIndex] = _spuGng_Cpp_ClustersNumSamples[clusterIndex] + 1;
              _spuGng_Cpp_ClustersTotalError[clusterIndex] = _spuGng_Cpp_ClustersTotalError[clusterIndex] + s1error;

              // Insere o valor desse voxel no volume de saa�da:
              float clusterIndexf = (float)clusterIndex;
              //                        if( _spuOutRawProcessedVolume->setVoxel( x, y, _spuEntryVolumeDim[2]-z-1, (void*)(&(clusterIndexf)) ) == -1 )
              if( _spuOutRawProcessedVolume->setVoxel( x, y, z, (void*)(&(clusterIndexf)) ) == -1 )
                return -1;
            }
          }
        }
      }
    }

    // Caso de classificacao em multi-tarefa:
    if( usingThreads == true )
    {
      int numThreads = 8;
      int ret = 0;
      int numPUnits = numThreads;
      float inc = _spuEntryVolumeDim[0] / numPUnits;
      std::vector< std::pair<int, int> > threadDiv;
      threadDiv.reserve( numPUnits );
      threadDiv.reserve( numPUnits );

      std::cout << 0 << "_:_" << _spuEntryVolumeDim[0] << std::endl;
      for( int i=0 ; i<numPUnits ; i++ )
      {
        threadDiv.push_back( std::pair<int, int> (Rounded( (double)(i*inc)), Rounded( (double)((i+1)*inc)) ) );
        std::cout << (threadDiv[i]).first << " : " << (threadDiv[i]).second << std::endl;
      }
      // So pra garantir que nao hajam erros de arredondamento:
      threadDiv[ threadDiv.size()-1 ].second = _spuEntryVolumeDim[0];

      int perc=0;
//      omp_set_nested(1);
//      omp_set_dynamic( numThreads );
      omp_set_dynamic( 0 );
      omp_set_num_threads( numThreads );
#pragma omp parallel for
      for( int i=0 ; i<numPUnits ; i++ )
      {
        ret+= createOutRawProcessedVolumeThreadStep( (threadDiv[i]).first ,(threadDiv[i]).second, perc );
      }
    }

    // Imprimindo o tempo de processamento:
    time_t endTime;
    endTime = time (NULL);
    std::cout << "Levou: " << (endTime-initTime) << " seg." << std::endl;


    return 0;
  }

  
  /**
    * Verifica a distancia entre duas coordenadas recebidas.
    * Sao retornadas distancia euclideana e distancia relativa.
    */
  int getDistance( Vector3Di p1, Vector3Di p2, float& distRet, float& relDistRet )
  {
    void* retVoxel = NULL;

    int p1x, p1y, p1z;
    p1.getValue(p1x, p1y, p1z);
    _spuOutRawProcessedVolume->getVoxel( p1x, p1y, p1z, retVoxel );
    float* p1RetVoxelf = (float*)retVoxel;
    float p1ClusterIndexf = p1RetVoxelf[0];
    int p1ClusterIndex = Rounded( p1ClusterIndexf );

    int p2x, p2y, p2z;
    p2.getValue(p2x, p2y, p2z);
    _spuOutRawProcessedVolume->getVoxel( p2x, p2y, p2z, retVoxel );
    float* p2RetVoxelf = (float*)retVoxel;
    float p2ClusterIndexf = p2RetVoxelf[0];
    int p2ClusterIndex = Rounded( p2ClusterIndexf );


    // Encontra os cluster da primeira e segunda amostras:
    p1ClusterIndex = _spuGng_Cpp->getClusterPosFromId( p1ClusterIndex );
    p2ClusterIndex = _spuGng_Cpp->getClusterPosFromId( p2ClusterIndex );

    // Testa se os clusterIndexes estao dentro da faixa de indices dos clusters:
    int numClusters = _spuGng_Cpp->getNumClusters();
    // Caso alguma das coordenadas seja invalida, retorna -1:
    if( (p1ClusterIndex < 0) || (p1ClusterIndex >= numClusters) )
      return -1;
    if( (p2ClusterIndex < 0) || (p2ClusterIndex >= numClusters) )
      return -1;

    // Utiliza as matrizes de distancias e distancias relativas, evitando calculo:
    distRet = _spuDistancesMatrix[p1ClusterIndex][p2ClusterIndex];
    relDistRet = _spuRelDistancesMatrix[p1ClusterIndex][p2ClusterIndex];

    return 0;
  }


  /**
    * Funcao que encontra o melhor vizinho de p1 a partir da coordenada de entrada p2.
    * A coordenada 2D do melhor vizinho sera a mesma de p2, com a terceira dimensao podendo
    * variar:  (p2.z-1) <= z <= p2.z+1).
    * A funcao ira retornar as distancias euclideana e relativa.
    * @return retorna -1 em caso de erro, ou 0 caso ok.
    */
  int getBestColumnNeighbor( Vector3Di p1, Vector3Di& p2, float& distRet, float& relDistRet, int maxDesloc=1 )
  {
    // Obtendo as coordenadas de p2:
    int x2, y2, z2;
    p2.getValue( x2, y2, z2 );


    float distMin    = std::numeric_limits<float>::max();
    float relDistMin = std::numeric_limits<float>::max();
    Vector3Di neighborMin;
    for( z2=p2[2]-maxDesloc ; z2 <= p2[2]+maxDesloc ; z2++ )
    {
      float dist, relDist;
      if( getDistance( p1, Vector3Di(x2, y2, z2), dist, relDist ) == 0 )
      {

        if( relDist <= relDistMin )
        {
          distMin = dist;
          relDistMin = relDist;
          neighborMin.setValue( x2, y2, z2 );
        }
      }
    }
    // Caso tenha sido encontrado pelo menos um vizinho:
    if( distMin < std::numeric_limits<float>::max() )
    {
      p2 = neighborMin;
      distRet = distMin;
      relDistRet = relDistMin;
      return 0;
    }

    return -1;
  }


  /**
    * Verifica a distancia entre duas coordenadas recebidas, considerando-as como o centro
    * de uma amostra janelada. O valor do janelamento e recebido como parametro.
    * Sao retornadas distancia euclideana e distancia relativa do somatorio do janelamento.
    */
  int getWindowDistance( Vector3Di p1, Vector3Di p2, int window, float& distRet, float& relDistRet, int maxDesloc=1 )
  {
    // Obtendo as coordenadas de p1:
    int x, y, z;
    p1.getValue( x, y, z );
    // Obtendo as coordenadas de p2:
    int x2, y2, z2;
    p2.getValue( x2, y2, z2 );

    // Define a borda que garante nao existirem coordenadas invalidas:
    int border = (_spuSamplesSize / 2) + 1 + window + 3;

    // Coordenadas invalidas:
    if( (z-window-1) < border )                             return -1;
    if( (z+window+1) + border >= _spuEntryVolumeDim[2] )    return -1;
    if( (z2-window-1) < border )                            return -1;
    if( (z2+window+1) + border >= _spuEntryVolumeDim[2] )   return -1;

    distRet = 0;
    relDistRet = 0;
    for( int i=-window ; i<=window ; i++ )
    {
      Vector3Di p1Tmp( x, y, z+i );
      Vector3Di p2Tmp( x2, y2, z2+i );
      float distRetTmp, relDistRetTmp;
      if( getBestColumnNeighbor( p1Tmp, p2Tmp, distRetTmp, relDistRetTmp, maxDesloc ) == -1 )
        return -1;
      distRet += distRetTmp;
      relDistRet += relDistRetTmp;
    }
    return 0;
  }


  /**
    * Funcao que verifica se um voxel pode ser considerado voxel de falha.
    * Recebe como parametro a coordenada do voxel em questao, alem de um
    * parametro definindo o tamanho da janela.
    */
  int isFaultVoxelAxisX( Vector3Di candidateVoxel, int window, int hDist, float& faultValue, float& faultRelValue )
  {
    if( (_spuOutRawProcessedVolume_Smp != NULL) && (_spuOutRawProcessedVolume_Smp->_vSmp_Size != 2*window+1) )
    {
      delete _spuOutRawProcessedVolume_Smp;
      _spuOutRawProcessedVolume_Smp = NULL;
    }

    if( _spuOutRawProcessedVolume_Smp == NULL )
      _spuOutRawProcessedVolume_Smp = new Volume_Smp( _spuOutRawProcessedVolume, 2*window+1, Vector3Di(0, 0, 0) );

    // A distancia horizontal maxima e 2 nessa implementacao:
    if( (hDist <= 0) || (hDist > 3) )
      return -1;

    // Inicializa os valores a serem retornados:
    faultValue = 0;
    faultRelValue = 0;

    // Obtendo as coordenadas do voxel a ser testado:
    int x, y, z;
    candidateVoxel.getValue( x, y, z );

    // Define a borda que garante nao existirem coordenadas invalidas:
    int border = (_spuSamplesSize / 2) + 1 + window;

    // Coordenadas invalidas:
    if( (z-window-1) < border )                             return -1;
    if( (z+window+1) + border >= _spuEntryVolumeDim[2] )    return -1;


    // Coordenadas mi�nimas da vizinhanca:
    int xm = ((x-hDist)>= 0)? (x-hDist) : 0;
    int xM = ((x+hDist)>= _spuEntryVolumeDim[0])? _spuEntryVolumeDim[0]-1 : (x+hDist);

    int p1X = candidateVoxel[0]-hDist;
    if( (p1X<xm) || (p1X>xM) )
      return -1;
    int p2X = candidateVoxel[0]+hDist;
    if( (p2X<xm) || (p2X>xM) )
      return -1;

    Vector3Di p1( p1X, candidateVoxel[1], candidateVoxel[2] );
    Vector3Di p2( p2X, candidateVoxel[1], candidateVoxel[2] );

    float distRet, relDistRet;
    if( getWindowDistance( p1, p2, window, distRet, relDistRet ) == -1 )
      return -1;

    faultValue = distRet;
    faultRelValue = relDistRet;

    return 0;
  }


  /**
    * Funcao que verifica se um voxel pode ser considerado voxel de falha.
    * Recebe como parametro a coordenada do voxel em questao, alem de um
    * parametro definindo o tamanho da janela.
    */
  int isFaultVoxelAxisY( Vector3Di candidateVoxel, int window, int hDist, float& faultValue, float& faultRelValue )
  {
    if( (_spuOutRawProcessedVolume_Smp != NULL) && (_spuOutRawProcessedVolume_Smp->_vSmp_Size != 2*window+1) )
    {
      delete _spuOutRawProcessedVolume_Smp;
      _spuOutRawProcessedVolume_Smp = NULL;
    }

    if( _spuOutRawProcessedVolume_Smp == NULL )
      _spuOutRawProcessedVolume_Smp = new Volume_Smp( _spuOutRawProcessedVolume, 2*window+1, Vector3Di(0, 0, 0) );

    // A distancia horizontal maxima e 2 nessa implementacao:
    if( (hDist <= 0) || (hDist > 3) )
      return -1;

    // Inicializa os valores a serem retornados:
    faultValue = 0;
    faultRelValue = 0;

    // Obtendo as coordenadas do voxel a ser testado:
    int x, y, z;
    candidateVoxel.getValue( x, y, z );

    // Define a borda que garante nao existirem coordenadas invalidas:
    int border = (_spuSamplesSize / 2) + 1 + window;

    // Coordenadas invalidas:
    if( (z-window-1) < border )                             return -1;
    if( (z+window+1) + border >= _spuEntryVolumeDim[2] )    return -1;


    // Coordenadas mi�nimas da vizinhanca:
    int ym = ((y-hDist)>= 0)? (y-hDist) : 0;
    int yM = ((y+hDist)>= _spuEntryVolumeDim[1])? _spuEntryVolumeDim[1]-1 : (y+hDist);

    int p1Y = candidateVoxel[1]-hDist;
    if( (p1Y<ym) || (p1Y>yM) )
      return -1;
    int p2Y = candidateVoxel[1]+hDist;
    if( (p2Y<ym) || (p2Y>yM) )
      return -1;

    Vector3Di p1( candidateVoxel[0], p1Y, candidateVoxel[2] );
    Vector3Di p2( candidateVoxel[0], p2Y, candidateVoxel[2] );

    float distRet, relDistRet;
    if( getWindowDistance( p1, p2, window, distRet, relDistRet ) == -1 )
      return -1;

    faultValue = distRet;
    faultRelValue = relDistRet;

    return 0;
  }


  /**
    * Funcao auxiliar. Recebe um a�ndice e converte para a coordenada do vizinho correspondente.
    */
  Vector3Di indexToCoord( int hDist, int index )
  {
    Vector3Di vec;
    vec.setValue( 0, 0, 0 );
    if( index == 0 )
    {
      vec[0]=+hDist; vec[1]=-hDist;
    }
    if( index == 1 )
    {
      vec[0]=+hDist; vec[1]= 0;
    }
    if( index == 2 )
    {
      vec[0]=+hDist; vec[1]=+hDist;
    }
    if( index == 3 )
    {
      vec[0]= 0; vec[1]=+hDist;
    }
    if( index == 4 )
    {
      vec[0]=-hDist; vec[1]=+hDist;
    }
    if( index == 5 )
    {
      vec[0]=-hDist; vec[1]= 0;
    }
    if( index == 6 )
    {
      vec[0]=-hDist; vec[1]=-hDist;
    }
    if( index == 7 )
    {
      vec[0]= 0; vec[1]=-hDist;
    }
    return vec;
  }

  /**
    * Procura por voxels de falha levando em conta todos os vizinhos.
    */
  int isFaultVoxel360( Vector3Di candidateVoxel, int window, int hDist, float& faultValue, float& faultRelValue )
  {
    entrou++;
    // A distancia horizontal maxima e 2 nessa implementacao:
    if( (hDist <= 0) || (hDist > 4) )
    {
        std::cout << "A distancia horizontal esta maior que o maximo permitido. Falha nao mapeada: " << std::endl;
        return -1;
    }

    // Inicializa os valores a serem retornados:
    faultValue = 0;
    faultRelValue = 0;

    // Obtendo as coordenadas do voxel a ser testado:
    int x, y, z;
    candidateVoxel.getValue( x, y, z );

    // Define a borda que garante nao existirem coordenadas invalidas:
    int border = (_spuSamplesSize / 2) + 1 + window;

    // Coordenadas invalidas:
    if( (z-window-1) < border )                             return -1;
    if( (z+window+1) + border >= _spuEntryVolumeDim[2] )    return -1;


    // Coordenadas mi�nimas da vizinhanca:
    int xm = ((x-hDist)>= 0)? (x-hDist) : 0;
    int xM = ((x+hDist)>= _spuEntryVolumeDim[0])? _spuEntryVolumeDim[0]-1 : (x+hDist);

    int ym = ((y-hDist)>= 0)? (y-hDist) : 0;
    int yM = ((y+hDist)>= _spuEntryVolumeDim[1])? _spuEntryVolumeDim[1]-1 : (y+hDist);


    int numDists = 0;
    float distRetSum = 0;
    float relDistRetSum = 0;

    // As distancias serao a menor das maiores:
    float distRetFinal    = std::numeric_limits<float>::max();
    float relDistRetFinal = std::numeric_limits<float>::max();

    for( int i=0 ; i<=7 ; i++ )
    {
      float distRetMax    = std::numeric_limits<float>::min();
      float relDistRetMax = std::numeric_limits<float>::min();
      Vector3Di p1 = candidateVoxel + indexToCoord( hDist, i );
      if( (p1[0]<xm) || (p1[0]>xM) ) return -1;
      if( (p1[1]<ym) || (p1[1]>yM) ) return -1;

      for( int j=0 ; j<=7 ; j++ )
      {
        if( i == j )
          continue;
        Vector3Di p2 = candidateVoxel + indexToCoord( hDist, j );
        if( (p2[0]<xm) || (p2[0]>xM) ) return -1;
        if( (p2[1]<ym) || (p2[1]>yM) ) return -1;

        float distRet, relDistRet;
        if( getWindowDistance( p1, p2, window, distRet, relDistRet ) == -1 )
          return -1;
        if( relDistRet > relDistRetMax )
        {
          relDistRetMax = relDistRet;
          distRetMax = distRet;
        }
      }

      // Pega a menor das maiores:
      if( relDistRetFinal > relDistRetMax )
      {
        relDistRetFinal = relDistRetMax;
        distRetFinal = distRetMax;
      }

      if( relDistRetMax > std::numeric_limits<float>::min() )
      {
        numDists++;
        distRetSum += distRetMax;
        relDistRetSum += relDistRetMax;

      }
    }

    // Retorna a menor das maiores:
    faultValue = distRetFinal;
    faultRelValue = relDistRetFinal;

    // Retorna a media das maiores diferencas:
    faultValue = distRetSum / numDists;
    faultRelValue = relDistRetSum / numDists;

    entrouEsaiu++;
    return 0;
  }


  /**
    * Procura por voxels de falha levando em conta todos os vizinhos.
    */
  int isFaultVoxel( Vector3Di candidateVoxel, int window, int hDist, float& faultValue, float& faultRelValue )
  {
    entrou++;
    // A distancia horizontal maxima e 4 nessa implementacao:
    if( (hDist <= 0) || (hDist > 4) )
    {
        std::cout << "A distancia horizontal esta maior que o maximo permitido. Falha nao mapeada: " << std::endl;
        return -1;
    }

    // Inicializa os valores a serem retornados:
    faultValue = 0;
    faultRelValue = 0;

    // Obtendo as coordenadas do voxel a ser testado:
    int x, y, z;
    candidateVoxel.getValue( x, y, z );

    // Define a borda que garante nao existirem coordenadas invalidas:
    int border = (_spuSamplesSize / 2) + 1 + window + 1;

    // Coordenadas invalidas:
    if( (z-window-1) < border )                             return -1;
    if( (z+window+1) + border >= _spuEntryVolumeDim[2] )    return -1;


    // Coordenadas mi�nimas da vizinhanca:
    int xm = ((x-hDist)>= 0)? (x-hDist) : 0;
    int xM = ((x+hDist)>= _spuEntryVolumeDim[0])? _spuEntryVolumeDim[0]-1 : (x+hDist);

    int ym = ((y-hDist)>= 0)? (y-hDist) : 0;
    int yM = ((y+hDist)>= _spuEntryVolumeDim[1])? _spuEntryVolumeDim[1]-1 : (y+hDist);

    int zm = ((z-border)>= 0)? (z-border) : 0;
    int zM = ((z+border)>= _spuEntryVolumeDim[0])? _spuEntryVolumeDim[0]-border : (z+border);

    // As distancias serao a menor das maiores:
    int numDistsFinal = 0;
    float distRetFinal    = 0;
    float relDistRetFinal = 0;

	float distRetGlobalMin    = std::numeric_limits<float>::max();
	float relDistRetGlobalMin = std::numeric_limits<float>::max();

    Vector3Di p1 = candidateVoxel;
    for( int i=0 ; i<=hDist ; i++ )
    {
    	for( int j0=xm ; j0<=xM ; j0++ )
    	{
    		for( int j1=ym ; j1<=yM ; j1++ )
    		{
    			if( j0 == j1 )
    				continue;

    			float distRet, relDistRet;
    			// A cada passo pega a menor distancia:
    			float distRetMin    = std::numeric_limits<float>::max();
    			float relDistRetMin    = std::numeric_limits<float>::max();

    			int zmThis = ((z-i)>= zm)? ((z-i)) : zm;
    			int zMThis = ((z+i)>= zM)? zM : (z+i);

    			for( int k=zmThis ; k<=zMThis ; k++ )
    			{
    				Vector3Di p2( j0, j1, k );

    				if( getWindowDistance( p1, p2, window, distRet, relDistRet, 2 ) == -1 )
    					return -1;
    				if( relDistRet < relDistRetMin )
    				{
    					relDistRetMin = relDistRet;
    					distRetMin = distRet;
    				}

    				if( relDistRet < relDistRetGlobalMin )
    				{
    					relDistRetGlobalMin = relDistRet;
    					distRetGlobalMin = distRet;
    				}
    			}
    			if( relDistRetMin != std::numeric_limits<float>::max() )
    			{
    				numDistsFinal++;
    				distRetFinal += distRetMin;
    				relDistRetFinal += relDistRetMin;
    			}
    		}
    	}
    }

    // Retorna a media das diferencas:
    faultValue = distRetFinal / numDistsFinal;
    faultRelValue = relDistRetFinal / numDistsFinal;

    // Retorna a menor das diferencas:
//    faultValue = distRetGlobalMin;
//    faultRelValue = relDistRetGlobalMin;

    entrouEsaiu++;
    return 0;
  }


  /**
    *
    */
  int createFaultVolumeThreadStep( int xInit, int xEnd, int window, int hDist, DVPVolumeData* faultVolume )
  {
      if(xInit<hDist)
          xInit=hDist;
      if(xEnd>(_spuEntryVolumeDim[0]-hDist))
          xEnd=(_spuEntryVolumeDim[0]-hDist);
      for( int x=xInit ; x<xEnd ; x++ )
      {
        for( int y=hDist ; y<_spuEntryVolumeDim[1]-hDist ; y++ )
        {
          // Amostras com a�ndices onde amplitudes mudam de sinal (positivo => negativo):
          for( int z=0 ; z<_spuEntryVolumeDim[2] ; z++ )
          {
            float faultValue360=0;
            float faultRelValue360=0;

            if( isFaultVoxel( Vector3Di(x,y,z), window, hDist, faultValue360, faultRelValue360 ) == -1 )
              continue;
            if( (entrou % 30000) == 0)
              std::cout << faultRelValue360 << std::endl;
            if( faultRelValue360 < 0 )
              faultRelValue360 = 0;
            if( faultValue360 < 0 )
          	  faultValue360 = 0;

            // Insere o valor desse voxel no volume de saa�da:
            if( faultVolume->setVoxel( x, y, _spuEntryVolumeDim[2]-z-1, (void*)(&(faultRelValue360)) ) == -1 )
              return -1;
          }
        }
//        std::cout << "Coordenada: " << x << " executada!" << std::endl;
//        std::cout << "Entrou: " << entrou << " entrou e saiu: " << entrouEsaiu << std::endl;
      }
  }


  /**
    * Processa um volume de falhas.
    */
  int createFaultVolume( int window, int hDist, DVPVolumeData*& faultVolume, bool usingThreads = false )
  {
      if( (_spuOutRawProcessedVolume_Smp != NULL) && (_spuOutRawProcessedVolume_Smp->_vSmp_Size != 2*window+1) )
      {
          delete _spuOutRawProcessedVolume_Smp;
          _spuOutRawProcessedVolume_Smp = NULL;
      }

      if( _spuOutRawProcessedVolume_Smp == NULL )
          _spuOutRawProcessedVolume_Smp = new Volume_Smp( _spuOutRawProcessedVolume, 2*window+1, Vector3Di(0, 0, 0) );

    // Criando o volume de saida:
    faultVolume = new DVPVolumeData( _spuEntryVolume, 0, 0 );
    
    std::cout << "Iniciando o processo de mapeamento de falhas: " << std::endl;

    if(usingThreads==false)
    {
//        for( int x=50 ; x<55 ; x++ )
        for( int x=hDist ; x<(_spuEntryVolumeDim[0]-hDist) ; x++ )
        {
          for( int y=hDist ; y<_spuEntryVolumeDim[1]-hDist ; y++ )
          {
            // Amostras com a�ndices onde amplitudes mudam de sinal (positivo => negativo):
            for( int z=_spuSamplesSize+1 ; z<_spuEntryVolumeDim[2]-(_spuSamplesSize-1) ; z++ )
            {
              float faultValue360=0;
              float faultRelValue360=0;

              if( isFaultVoxel( Vector3Di(x,y,z), window, hDist, faultValue360, faultRelValue360 ) == -1 )
                continue;
              if( (entrou % 3000) == 0)
                std::cout << faultRelValue360 << std::endl;
              if( faultRelValue360 < 0 )
                faultRelValue360 = 0;
              if( faultValue360 < 0 )
            	  faultValue360 = 0;

              // Insere o valor desse voxel no volume de saa�da:
              if( faultVolume->setVoxel( x, y, _spuEntryVolumeDim[2]-z-1, (void*)(&(faultRelValue360)) ) == -1 )
                //                        if( faultVolume->setVoxel( x, y, z, (void*)(&(faultRelValue)) ) == -1 )
                return -1;
            }
          }
//          if( (x % 10)==0 )
//            faultVolume->saveVolume( s.c_str() );
          std::cout << "Coordenada: " << x << " executada!" << std::endl;
          std::cout << "Entrou: " << entrou << " entrou e saiu: " << entrouEsaiu << std::endl;
        }
    }

    if(usingThreads==true)
    {
      int numThreads = 8;
      int ret = 0;
      int numPUnits = numThreads;

      float ini = hDist;
      float end = _spuEntryVolumeDim[0]-hDist;

//      float ini = (_spuEntryVolumeDim[0]/2)-20;
//      float end = (_spuEntryVolumeDim[0]/2)+20;
      float inc = (end-ini) / numPUnits;
      std::vector< std::pair<int, int> > threadDiv;
      threadDiv.reserve( numPUnits );
      threadDiv.reserve( numPUnits );

      std::cout << 0 << "_:_" << _spuEntryVolumeDim[0] << std::endl;
      for( int i=0 ; i<numPUnits ; i++ )
      {
        threadDiv.push_back( std::pair<int, int> (Rounded( (double)(ini+(i*inc))), Rounded( (double)(ini+((i+1)*inc))) ) );
        std::cout << (threadDiv[i]).first << " : " << (threadDiv[i]).second << std::endl;
      }
      // So pra garantir que nao hajam erros de arredondamento:
      threadDiv[ threadDiv.size()-1 ].second = (int)end;

//      omp_set_nested(1);
//      omp_set_dynamic( numThreads );
      omp_set_dynamic( 0 );
      omp_set_num_threads( numThreads );
#pragma omp parallel for
      for( int i=0 ; i<numPUnits ; i++ )
      {
        ret+= createFaultVolumeThreadStep( (threadDiv[i]).first, (threadDiv[i]).second, window, hDist, faultVolume );
      }
    }

//    if( faultVolume->saveVolume( s.c_str() ) == -1 )
//      return -1;

    return 0;
  }


  /**
    * Verifica se um determinado voxel e voxel de falha.
    */
  int findFaultValue( Vector3Di seed, int window, Vector3Di reach, std::vector<std::vector<int> >& usedPositions, float& faultValue )
  {
    /* Gerando uma semente nova, para os nmeros aleatrios: */
    srand( (unsigned)time( NULL ) );

    // Obtendo as coordenadas de seed:
    int x, y, z;
    seed.getValue( x, y, z );

    // Define a borda que garante nao existirem coordenadas invalidas:
    int border = (_spuSamplesSize / 2) + 1 + window + 3;

    // Coordenadas invalidas:
    if( (z-window-1) < border )                             return -1;
    if( (z+window+1) + border >= _spuEntryVolumeDim[2] )    return -1;

    // Coordenadas minimas da vizinhanca:
    int xm = ((x-reach[0])> 0)? (x-reach[0]) : 1;
    int ym = ((y-reach[1])> 0)? (y-reach[1]) : 1;
    int zm = ((z-1)> 0) ? (z-1) : 1;
    // Coordenadas maximas da vizinhanca:
    int xM = ((x+reach[0])>= (_spuEntryVolumeDim[0]-1))? _spuEntryVolumeDim[0]-2 : (x+reach[0]);
    int yM = ((y+reach[1])>= (_spuEntryVolumeDim[1]-1))? _spuEntryVolumeDim[1]-2 : (y+reach[1]);
    int zM = ((z+1) >= (_spuEntryVolumeDim[2]-1))? _spuEntryVolumeDim[2]-2 : (z+1);
    
    // Utiliza as coordenadas para construir a BBox de procura:
    Vector3Di min( xm, ym, zm );
    Vector3Di max( xM, yM, zM );

    // OBS: PARA REGIONLAYERS NAO E NECESSARIO SABER O SINAL DA SEMENTE!!!

#define USING_LIST
#ifdef USING_LIST
    // Lista que ira conter todos os nos:
    std::list<Vector3Di> hzList;
    hzList.push_back( seed );
    usedPositions[seed[0]][seed[1]] = seed[2];
    std::list<Vector3Di>::iterator hzListIt = hzList.begin();

    float totalRelDist = 0;
    while( hzListIt != hzList.end() )
    {
      Vector3Di thisSample = (*hzListIt);

      if( this->intersects( min, max, thisSample ) )
      {
          std::vector<Vector3Di> neihborsRet;
          std::vector<float> distsRet;
          std::vector<float> relDistsRet;
          if( getImmediateWindowNeighbors( seed, thisSample, window, neihborsRet, distsRet, relDistsRet, false ) == 0 )
          {
            for( unsigned int i=0 ; i<neihborsRet.size() ; i++ )
            {
              Vector3Di thisNeighbor = neihborsRet[i];
              // Caso esse vizinho ja tenha sido incluso, apenas continua:
              if( usedPositions[thisNeighbor[0]][thisNeighbor[1]] != -1 )
                continue;
              // Caso contrario, inclui esse vizinho na lista e marca como descoberto:
              hzList.push_back( thisNeighbor );
              usedPositions[thisNeighbor[0]][thisNeighbor[1]] = thisNeighbor[2];
              totalRelDist += relDistsRet[i];
            }
            neihborsRet.clear();
          }
      }

      hzListIt++;
    }
#endif

//#define USING_MULTIMAP
#ifdef USING_MULTIMAP

    // Multimap que ira conter todos os nos:
    std::multimap< float, Vector3Di > hzMultiMap;
    hzMultiMap.insert( std::pair<float, Vector3Di>( 0.0f, seed ) );

    usedPositions[seed[0]][seed[1]] = seed[2];
    std::multimap< float, Vector3Di >::iterator hzMultiMapIt = hzMultiMap.begin();

    float totalRelDist = 0;
    while( hzMultiMapIt != hzMultiMap.end() )
    {
      Vector3Di thisSample = (*hzMultiMapIt).second;

      if( neighborsBBox.intersect( (const Vector3Di)thisSample ) )
      {
          std::vector<Vector3Di> neihborsRet;
          std::vector<float> distsRet;
          std::vector<float> relDistsRet;
          if( getImmediateWindowNeighbors( seed, thisSample, window, neihborsRet, distsRet, relDistsRet, false ) == 0 )
          {
            for( unsigned int i=0 ; i<neihborsRet.size() ; i++ )
            {
              Vector3Di thisNeighbor = neihborsRet[i];
              // Caso esse vizinho ja tenha sido incluso, apenas continua:
              if( usedPositions[thisNeighbor[0]][thisNeighbor[1]] != -1 )
                continue;
              // Caso contrario, inclui esse vizinho na lista e marca como descoberto:
              hzMultiMap.insert( std::pair<float, Vector3Di>( relDistsRet[i], thisNeighbor ) );
              usedPositions[thisNeighbor[0]][thisNeighbor[1]] = thisNeighbor[2];
              totalRelDist += relDistsRet[i];
            }
            neihborsRet.clear();
          }
      }

      hzMultiMapIt = hzMultiMap.begin();
    }
#endif

    // Zera os valores da BBox que foi utilizada:
    for( int i=xm-1 ; i<=xM+1 ; i++)
        for( int j=ym-1 ; j<=yM+1 ; j++)
            usedPositions[i][j] = -1;

    // Retorna o valor de falha encontrado:
    faultValue = totalRelDist;

    return 0;
  }


  /**
    * Processa uma porcao de um volume de falhas.
    */
  int createFaultVolume2ThreadStep( int xInit, int xEnd, int reach, int window, DVPVolumeData*& faultVolume )
  {
      // Em usedPositions teremos toda essa porcao do horizonte da falha mapeado ao final do processo:
      std::vector<std::vector<int> >  usedPositions;
      usedPositions.resize(_spuEntryVolumeDim[0]);
      for (int i = 0; i < _spuEntryVolumeDim[0]; ++i)
      {
        usedPositions[i].resize(_spuEntryVolumeDim[1]);
        for (int j = 0; j < _spuEntryVolumeDim[1]; ++j)
          usedPositions[i][j] = -1;
      }

      for( int x=xInit ; x<xEnd ; x++ )
      {
        for( int y=reach ; y<_spuEntryVolumeDim[1]-reach-1 ; y++ )
        {
          // Amostras com a�ndices onde amplitudes mudam de sinal (positivo => negativo):
          for( int z=10 ; z<_spuEntryVolumeDim[2]-10 ; z++ )
          {
            float faultValue=0;

            findFaultValue( Vector3Di( x, y, z ), window, Vector3Di(reach, reach, reach), usedPositions, faultValue );
            // Insere o valor desse voxel no volume de saida:
            if( faultVolume->setVoxel( x, y, _spuEntryVolumeDim[2]-z-1, (void*)(&(faultValue)) ) == -1 )
              return -1;
          }
        }
        std::cout << "createFaultVolume2ThreadStep. Coordenada: " << x << " executada!" << std::endl;
      }

      return 0;
  }


  /**
    * Processa um volume de falhas.
    */
  int createFaultVolume2( int window, int reach, DVPVolumeData*& faultVolume, bool usingThreads = false )
  {
    if( (_spuOutRawProcessedVolume_Smp != NULL) && (_spuOutRawProcessedVolume_Smp->_vSmp_Size != 2*window+1) )
    {
      delete _spuOutRawProcessedVolume_Smp;
      _spuOutRawProcessedVolume_Smp = NULL;
    }

    if( _spuOutRawProcessedVolume_Smp == NULL )
      _spuOutRawProcessedVolume_Smp = new Volume_Smp( _spuOutRawProcessedVolume, 2*window+1, Vector3Di(0, 0, 0) );

	  // Criando o volume de saida:
	  faultVolume = new DVPVolumeData( _spuEntryVolume, 0 , 0 );	  

    std::cout << "Iniciando o processo de mapeamento de falhas: " << std::endl;

    if( usingThreads == false )
    {
        // Em usedPositions teremos toda essa porcao do horizonte da falha mapeado ao final do processo:
        std::vector<std::vector<int> >  usedPositions;
        usedPositions.resize(_spuEntryVolumeDim[0]);
        for (int i = 0; i < _spuEntryVolumeDim[0]; ++i)
        {
          usedPositions[i].resize(_spuEntryVolumeDim[1]);
          for (int j = 0; j < _spuEntryVolumeDim[1]; ++j)
            usedPositions[i][j] = -1;
        }

        for( int x=reach ; x<_spuEntryVolumeDim[0]-reach-1 ; x++ )
        {
          for( int y=reach ; y<_spuEntryVolumeDim[1]-reach-1 ; y++ )
          {
            // Amostras com a�ndices onde amplitudes mudam de sinal (positivo => negativo):
            for( int z=20 ; z<_spuEntryVolumeDim[2]-20 ; z++ )
            {
              float faultValue=0;

              findFaultValue( Vector3Di( x, y, z ), window, Vector3Di(reach, reach, reach), usedPositions, faultValue );
              // Insere o valor desse voxel no volume de saida:
              if( faultVolume->setVoxel( x, y, _spuEntryVolumeDim[2]-z-1, (void*)(&(faultValue)) ) == -1 )
                return -1;
            }
          }
          std::cout << "createFaultVolume2. Coordenada: " << x << " executada!" << std::endl;
        }
    }

    if( usingThreads == true )
    {
      int numThreads = 10;
      int ret = 0;
      int numPUnits = numThreads;
      float ini = reach;
      float end = _spuEntryVolumeDim[0]-reach-1;
//      float ini = (_spuEntryVolumeDim[0]/2)-20;
//      float end = (_spuEntryVolumeDim[0]/2)+20;
      float inc = (end-ini) / numPUnits;
      std::vector< std::pair<int, int> > threadDiv;
      threadDiv.reserve( numPUnits );
      threadDiv.reserve( numPUnits );

      std::cout << 0 << "_:_" << _spuEntryVolumeDim[0] << std::endl;
      for( int i=0 ; i<numPUnits ; i++ )
      {
        threadDiv.push_back( std::pair<int, int> (Rounded( (double)(ini+(i*inc))), Rounded( (double)(ini+((i+1)*inc))) ) );
        std::cout << (threadDiv[i]).first << " : " << (threadDiv[i]).second << std::endl;
      }
      // So pra garantir que nao hajam erros de arredondamento:
      threadDiv[ threadDiv.size()-1 ].second = (int) end;

      omp_set_dynamic( 0 );
      omp_set_num_threads( numThreads );
#pragma omp parallel for
      for( int i=0 ; i<numPUnits ; i++ )
      {
        ret+= createFaultVolume2ThreadStep( (threadDiv[i]).first, (threadDiv[i]).second, reach, window, faultVolume );
      }
    }

    return 0;
  }


  /**
   * CODIGO CRIADO PARA GERAR UMA PRIMEIRA VERSAO DE WEIGHTEDDEEPVOLUME.
   * CODIGO CRIADO PARA GERAR UMA PRIMEIRA VERSAO DE WEIGHTEDDEEPVOLUME.
   */
  int getWindowNeighbor( Vector3Di seed, Vector3Di p1, int window, Vector3Di& neihborRet,
		  float& distRet, float& relDistRet )
  {
	  // Obtendo as coordenadas de p1:
	  int p1x, p1y, p1z;
	  p1.getValue( p1x, p1y, p1z );

	  // Caso seed nao seja um pico de amplitude, retorna erro:
      float peakVoxel = 0.0;
      if( isPeak( seed,  Volume_Smp::UNDEFINED , peakVoxel ) == false )
    	  return -1;

	  // Define a borda que garante nao existirem coordenadas invalidas:
	  int border = (_spuSamplesSize / 2) + 1 + window + 3;

	  // Coordenadas invalidas:
	  if( (p1z-window-1) < border )                             return -1;
	  if( (p1z+window+1) + border >= _spuEntryVolumeDim[2] )    return -1;

	  // Coordenadas minimas e maximas da vizinhanca:
	  int zm = ((p1z-2)>= 0) ? (p1z-2) : 0;
	  int zM = ((p1z+2) < _spuEntryVolumeDim[2])? (p1z+2) : _spuEntryVolumeDim[2]-1;

	  distRet = std::numeric_limits<float>::max();
	  relDistRet = std::numeric_limits<float>::max();

	  // Percorre os vizinhos imediatos do voxel, encontrando possiveis vizinhos:
	  float distMin    = std::numeric_limits<float>::max();
	  float relDistMin = std::numeric_limits<float>::max();
	  Vector3Di neighborMin;
	  for( int z2=zm ; z2<=zM ; z2++ )
	  {
	      if( isPeak( Vector3Di(p1x, p1y, z2),  Volume_Smp::UNDEFINED , peakVoxel ) == false )
	    	  continue;

		  float dist, relDist;
		  // Caso nao estejamos procurando por picos, basta proceder com a busca:
		  if( getWindowDistance( seed, Vector3Di(p1x, p1y, z2), window, dist, relDist ) == 0 )
		  {
			  if( relDist < relDistMin )
			  {
				  distMin = dist;
				  relDistMin = relDist;
				  neighborMin.setValue( p1x, p1y, z2 );
			  }
		  }
		  else    //< Para que a distancia de colunas exista, todos os candidatos da coluna tem que existir:
		  {
			  distMin    = std::numeric_limits<float>::max();
			  relDistMin = std::numeric_limits<float>::max();
			  break;
		  }
	  }
	  // Caso tenha sido encontrado pelo menos um vizinho:
	  if( distMin < std::numeric_limits<float>::max() )
	  {
		  neihborRet = neighborMin;
		  distRet = distMin;
		  relDistRet = relDistMin;
	  }
	  return 0;
  }


  /**
   * Recebe a coordenada de um vertice e uma direcao. Encontra o valor e um Calcula o valor do deep entre duas coordenadas recebidas.
   */
  int getDipNeighbor( Vector3Di p1, Vector2Di p2_2d, int window, Vector3Di& neighborRet,
		  float& difRet )
  {
	  // Inicializando neighborRet com valores invalidos:
	  neighborRet.setValue( -1,-1,-1 );

	  // Obtendo as coordenadas de p1:
	  int p1x, p1y, p1z;
	  p1.getValue( p1x, p1y, p1z );

	  Vector3Di p2( p2_2d[0], p2_2d[1], p1z );

	  float distRet, relDistRet;
	  int ret1 = getWindowNeighbor( p1, p2, window, neighborRet, distRet, relDistRet );
	  if( ret1 == -1 )
		  return -1;

	  Vector3Di neighbor2Ret(-2,-2,-2);
	  int ret2 = getWindowNeighbor( neighborRet, p1, window, neighbor2Ret, distRet, relDistRet );
	  if( ret2 == -1 )
		  return -1;

	  difRet = (float)neighborRet[2] - (float)p1[2];
	  if( (difRet < -2.0) || (difRet > 2.0f) )
		{
			std::cout << "SeismicProcessingControler: getDipNeighbor difRet invalido - Erro!!" << std::endl;
//			return -1;
		}


	  if( neighbor2Ret[2] ==  p1[2] )
		  return 0;

	  return -1;
  }


  /**
   * Funcao que cria um weightedDeepVolume.
   */
  int createweightedDipVolume( int window,  DVPVolumeData*& faultVolume, weightedDipVolume*& wDV )
  {
	  Vector3Di dim( _spuEntryVolumeDim[0], _spuEntryVolumeDim[1], _spuEntryVolumeDim[2] );

	  wDV = new weightedDipVolume( dim );


	  for( int x = 2; x < dim.x-2; x++ )
	  {
		  float perc = 100.0f*((float)x / (float)(dim.x-1) );
	        std::cout << "createweightedDipVolume. Executados :" << perc << " por cento. " << std::endl;

		  for( int y = 2; y < dim.y-2; y++ )
		  {
			  for( int z = 20; z < dim.z-20; z++ )
			  {
				  dipNode* thisNode = wDV->getDipNode( Vector3Di( x, y, z ) );
				  Vector4Df thisNodeDir( DIP_DUMMY_VALUE, DIP_DUMMY_VALUE, DIP_DUMMY_VALUE, DIP_DUMMY_VALUE );

				  // Percorre os vizinhos imediatos do voxel, encontrando possiveis vizinhos:
				  int x2, y2;

				  x2=x+1;		y2=y;
				  Vector3Di neighborRet;
				  float difRet;
				  int ret;
				  ret =  getDipNeighbor( Vector3Di(x, y, z), Vector2Di(x2, y2), window, neighborRet, difRet );
				  if( ret == 0 )
					  thisNodeDir.x = (float)difRet;

				  x2=x+1;		y2=y+1;
				  ret =  getDipNeighbor( Vector3Di(x, y, z), Vector2Di(x2, y2), window, neighborRet, difRet );
				  if( ret == 0 )
					  thisNodeDir.y = (float)difRet;

				  x2=x-1;		y2=y;
				  ret =  getDipNeighbor( Vector3Di(x, y, z), Vector2Di(x2, y2), window, neighborRet, difRet );
				  if( ret == 0 )
					  thisNodeDir.z = (float)difRet;

				  x2=x;		y2=y-1;
				  ret =  getDipNeighbor( Vector3Di(x, y, z), Vector2Di(x2, y2), window, neighborRet, difRet );
				  if( ret == 0 )
					  thisNodeDir.w = (float)difRet;

				  // Inserindo o valor do vetor direcao:
				  thisNode->setDir( thisNodeDir );

				  // Obtendo e inserindo o valor do atributo de discontinuidade:
	              float valueAct = 0.0;
	              int zTmp = _spuEntryVolumeDim[2]-z-1;
	              void* retVoxel = NULL;
	              faultVolume->getVoxel( x, y, zTmp, retVoxel );
	              float* retVoxelf = (float*)retVoxel;
	              valueAct = retVoxelf[0];
	              // Inserindo esse valor:
				  thisNode->setConf( valueAct );
			  }
		  }
	  }
	  return 0;
  }

  /**
   * FIM DO CODIGO CRIADO PARA GERAR UMA PRIMEIRA VERSAO DE WEIGHTEDDEEPVOLUME.
   * FIM DO CODIGO CRIADO PARA GERAR UMA PRIMEIRA VERSAO DE WEIGHTEDDEEPVOLUME.
   */


  /**
    * Funcao que encontra os vizinhos imediatos de um determinado voxel, cuja a posicao e recebida.
    * A funcao ira retornar um vetor de tres vetores, um contendo todos os voxels vizinhos mais proximos,
    * o segundo retornando suas respectivas distancias euclideanas e o terceiro suas respectivas distancias
    * relativas.
    */
  int getImmediateWindowNeighbors( Vector3Di seed, Vector3Di p1, int window, std::vector<Vector3Di>& neihborsRet,
                                   std::vector<float>& distsRet, std::vector<float>& relDistsRet, bool lookingForSameSignal = true,
                                   Volume_Smp::SamplesType type = Volume_Smp::NEGATIVE_PEAKS, bool lookingForPeaks = false )
  {
    // Obtendo as coordenadas de p1:
    int x, y, z;
    p1.getValue( x, y, z );

    // Define a borda que garante nao existirem coordenadas invalidas:
    int border = (_spuSamplesSize / 2) + 1 + window + 3;

    // Coordenadas invalidas:
    if( (z-window-1) < border )                             return -1;
    if( (z+window+1) + border >= _spuEntryVolumeDim[2] )    return -1;


    // Coordenadas mi�nimas da vizinhanca:
    int xm = ((x-1)>= 0)? (x-1) : 0;
    int ym = ((y-1)>= 0)? (y-1) : 0;
    int zm = ((z-1)>= 0) ? (z-1) : 0;
    // Coordenadas maximas da vizinhanca:
    int xM = ((x+1)>= _spuEntryVolumeDim[0])? _spuEntryVolumeDim[0]-1 : (x+1);
    int yM = ((y+1)>= _spuEntryVolumeDim[1])? _spuEntryVolumeDim[1]-1 : (y+1);
    int zM = ((z+1) >= _spuEntryVolumeDim[2])? _spuEntryVolumeDim[2]-1 : (z+1);


    // Limpa os vetores de entrada:
    neihborsRet.clear();
    distsRet.clear();
    relDistsRet.clear();


    // Percorre os vizinhos imediatos do voxel, encontrando possa�veis vizinhos:
    for( int x2=xm ; x2<=xM ; x2++ )
    {
      for( int y2=ym ; y2<=yM ; y2++ )
      {
        float distMin    = std::numeric_limits<float>::max();
        float relDistMin = std::numeric_limits<float>::max();
        Vector3Di neighborMin;
        for( int z2=zm ; z2<=zM ; z2++ )
        {
          if( (x==x2) && (y==y2) )
            continue;
          float dist, relDist;
          // Caso nao estejamos procurando por picos, basta proceder com a busca:
          if( lookingForSameSignal == false )
          {
            if( getWindowDistance( seed, Vector3Di(x2, y2, z2), window, dist, relDist ) == 0 )
            {
              if( relDist < relDistMin )
              {
                distMin = dist;
                relDistMin = relDist;
                neighborMin.setValue( x2, y2, z2 );
              }
            }
            else    //< Para que a distancia de colunas exista, todos os candidatos da coluna tem que existir:
            {
              distMin    = std::numeric_limits<float>::max();
              relDistMin = std::numeric_limits<float>::max();
              break;
            }
          }


          // Casos em que estamos procurando por amostras do mesmo sinal:
          if( (lookingForSameSignal == true) && (lookingForPeaks==false) )
          {
            if( isSameSignal( Vector3Di(x2, y2, z2), type ) == true )
            {
              // Modificando Z:
              float valueAct = 0.0;
              int zTmp = _spuEntryVolumeDim[2]-z2-1;
              void* retVoxel = NULL;
              _spuEntryVolume->getVoxel( x2, y2, zTmp, retVoxel );
              float* retVoxelf = (float*)retVoxel;
              valueAct = retVoxelf[0];

              // Os picos somente serao aceitos caso sua amplitude seja maior que o valor estabelecido:
              float vSmp_mean   = _spuVolume_Smp->_vSmp_mean;
              float vSmp_stdDev = _spuVolume_Smp->_vSmp_stdDev;
              if( fabs(valueAct) >= (vSmp_mean + ( _spuMinStdDevForTrain * vSmp_stdDev)) )
              {
                if( getWindowDistance( seed, Vector3Di(x2, y2, z2), window, dist, relDist ) == 0 )
                {
                  if( relDist < relDistMin )
                  {
                    distMin = dist;
                    relDistMin = relDist;
                    neighborMin.setValue( x2, y2, z2 );
                  }
                }
              }
            }
          }
          // Caso estejamos procurando por picos:
          if( (lookingForSameSignal == true) && (lookingForPeaks==true) )
          {
            float peakVoxel = 0.0;
            if( isPeak( Vector3Di(x2, y2, z2), type , peakVoxel ) == true )
            {
              // Os picos somente serao aceitos caso sua amplitude seja maior que o valor estabelecido:
              float vSmp_mean   = _spuVolume_Smp->_vSmp_mean;
              float vSmp_stdDev = _spuVolume_Smp->_vSmp_stdDev;
              if( fabs(peakVoxel) >= (vSmp_mean + ( _spuMinStdDevForTrain * vSmp_stdDev)) )
              {
                if( getWindowDistance( seed, Vector3Di(x2, y2, z2), window, dist, relDist ) == 0 )
                {
                  if( relDist < relDistMin )
                  {
                    distMin = dist;
                    relDistMin = relDist;
                    neighborMin.setValue( x2, y2, z2 );
                  }
                }
              }
            }
          }


        }
        // Caso tenha sido encontrado pelo menos um vizinho:
        if( distMin < std::numeric_limits<float>::max() )
        {
          neihborsRet.push_back( neighborMin );
          distsRet.push_back( distMin );
          relDistsRet.push_back( relDistMin );
        }
      }
    }
    return 0;
  }


  /**
    * Funcao auxiliar. Retorna se uma coordenada e de pico do tipo definido.
    */
  bool isPeak( Vector3Di seed, Volume_Smp::SamplesType type, float& peakVoxel )
  {
    // Caso em que o tipo de pido nao e recebido:
    if( type == Volume_Smp::UNDEFINED )
    {
      if( isPeak( seed, Volume_Smp::POSITIVE_PEAKS, peakVoxel) == true )
        return true;
      if( isPeak( seed, Volume_Smp::NEGATIVE_PEAKS, peakVoxel) == true )
        return true;

      return false;
    }

    // Verificando se e realmente um voxel de pico:
    // Obtendo os tres valores, e verificando se a amostra e realmente do tipo correta:
    int x, y, z;
    seed.getValue( x, y, z );

    float valueAnt;
    float valueAct;
    float valuePos;

    // Modificando Z:
    z = _spuEntryVolumeDim[2]-z-1;

    // Caso as coordenadas sejam invalidas, retorna erro:
    if( _spuEntryVolume->isValidCoord( x, y, z ) == false )
      return false;
    if( _spuEntryVolume->isValidCoord( x, y, z-1 ) == false )
      return false;
    if( _spuEntryVolume->isValidCoord( x, y, z+1 ) == false )
      return false;

    void* retVoxel = NULL;
    _spuEntryVolume->getVoxel( x, y, z-1, retVoxel );
    float* retVoxelf = (float*)retVoxel;
    valueAnt = retVoxelf[0];

    retVoxel = NULL;
    _spuEntryVolume->getVoxel( x, y, z, retVoxel );
    retVoxelf = (float*)retVoxel;
    valueAct = retVoxelf[0];

    // Retorna o valor do voxel central:
    peakVoxel = valueAct;

    retVoxel = NULL;
    _spuEntryVolume->getVoxel( x, y, z+1, retVoxel );
    retVoxelf = (float*)retVoxel;
    valuePos = retVoxelf[0];

    bool ok = false;
    if( type == Volume_Smp::NEGATIVE_PEAKS )
      if( (valueAct<0) && (valueAct<valueAnt) && (valueAct<valuePos) )
        ok = true;
    if( type == Volume_Smp::POSITIVE_PEAKS )
      if( (valueAct>0) && (valueAct>valueAnt) && (valueAct>valuePos) )
        ok = true;

    return ok;
  }


  /******************************************************************************************************************
                Apres Terca-feira: INICIO
                Apres Terca-feira: INICIO
                Apres Terca-feira: INICIO
   ******************************************************************************************************************/

  /**
    * Obt�m um vetor contendo todos os picos de amplitude que estao dentro da minima faixa de amplitude exigida.
    */
  int getSamplePositions( std::vector<std::vector<std::vector<bool> > >& spuSamplesPositions3D )
  {
      /* Gerando uma semente nova, para os nmeros aleatrios: */
      srand( (unsigned)time( NULL ) );

      // Cria uma matriz 3D de booleans, indicando onde existem amostras:
      spuSamplesPositions3D.resize(_spuEntryVolumeDim[0]);
      for (int i = 0; i < _spuEntryVolumeDim[0]; i++)
      {
          spuSamplesPositions3D[i].resize(_spuEntryVolumeDim[1]);
          for (int j = 0; j < _spuEntryVolumeDim[1]; j++)
          {
              spuSamplesPositions3D[i][j].resize(_spuEntryVolumeDim[2]);
              for (int k = 0; k < _spuEntryVolumeDim[2]; k++)
                  spuSamplesPositions3D[i][j][k] = false;
          }
      }


      // Os picos somente serao aceitos caso sua amplitude seja maior que o valor estabelecido:
      float vSmp_mean   = _spuVolume_Smp->_vSmp_mean;
      float vSmp_stdDev = _spuVolume_Smp->_vSmp_stdDev;


      std::vector<std::vector<std::vector<std::vector<bool> > > > samplesPositions3D;
      samplesPositions3D = _spuVolume_Smp->getSamplesPositions3D( );

      for( int x=0 ; x<_spuEntryVolumeDim[0] ; x++ )
      {
          for( int y=0 ; y<_spuEntryVolumeDim[1] ; y++ )
          {
              for (int z = 2; z < _spuEntryVolumeDim[2]-2; z++)
              {
                  float valueAct;
                  if( samplesPositions3D[ (int)(Volume_Smp::POSITIVE_PEAKS)][x][y][z] == true )
                  {
                      if( isPeak( Vector3Di( x, y, (_spuEntryVolumeDim[2]-z-1)), Volume_Smp::POSITIVE_PEAKS, valueAct ) == true )
                      {
                          if( fabs(valueAct) >= (vSmp_mean + ( _spuMinStdDevForTrain * vSmp_stdDev)) )
                              spuSamplesPositions3D[x][y][(_spuEntryVolumeDim[2]-z-1)] = true;
                      }
                  }

                  if( samplesPositions3D[(int)(Volume_Smp::NEGATIVE_PEAKS)][x][y][z] )
                  {
                      if( isPeak( Vector3Di( x, y, (_spuEntryVolumeDim[2]-z-1)), Volume_Smp::NEGATIVE_PEAKS, valueAct ) == true )
                      {
                          if( fabs(valueAct) >= (vSmp_mean + ( _spuMinStdDevForTrain * vSmp_stdDev)) )
                              spuSamplesPositions3D[x][y][(_spuEntryVolumeDim[2]-z-1)] = true;
                      }
                  }
              }
          }
      }
      return 0;
  }


  /**
    * Sorteia uma nova semente (pico) com base no valor de falha.
    */
  Vector3Di getGoodSeed( float maxFaultRelValue, int window, int hdist, std::vector<std::vector<std::vector<bool> > > spuSamplesPositions3D,
                         float& retFaultValue )
  {
      float tmpfaultValue = std::numeric_limits<float>::max();
      Vector3Di tmpBestSeed;

      int x = -1;
      int y = -1;
      int z = -1;
      int cont = 0;
      while( (tmpfaultValue > maxFaultRelValue) || (cont<100000) )
      {
          x = RandInt( (hdist+2), (_spuEntryVolumeDim[0]-hdist-3) );
          y = RandInt( (hdist+2), (_spuEntryVolumeDim[1]-hdist-3) );
          z = RandInt( (_spuSamplesSize+window), (_spuEntryVolumeDim[1]-(_spuSamplesSize+window)) );

          // Em usedPositions teremos toda a porcao do horizonte da falha mapeado ao final do processo:
          std::vector<std::vector<int> >  usedPositions;
          usedPositions.resize(_spuEntryVolumeDim[0]);
          for (int i = 0; i < _spuEntryVolumeDim[0]; ++i)
          {
            usedPositions[i].resize(_spuEntryVolumeDim[1]);
            for (int j = 0; j < _spuEntryVolumeDim[1]; ++j)
              usedPositions[i][j] = -1;
          }

          float thisFaultValue = std::numeric_limits<float>::max();
          if( spuSamplesPositions3D[x][y][z] == true )
              findFaultValue( Vector3Di( x, y, z ), window, Vector3Di(hdist, hdist, hdist), usedPositions, thisFaultValue );

          if( thisFaultValue < tmpfaultValue )
          {
              tmpBestSeed.setValue( x, y, z );
              tmpfaultValue = thisFaultValue;
          }

          cont++;
      }

      retFaultValue = tmpfaultValue;
      return tmpBestSeed;
  }

   
  /**
      * Recebe as tres coordenadas de um determinado voxel e retorna seu valor.
      * @param Inline Coordenada X do voxel.
      * @param Crossline Coordenada Y do voxel.
      * @param Time Coordenada Z do voxel.
      * @param ReturnedVoxel Valor retornado do voxel.
      * @return Retorna -1 em caso de erro (parametros, volume inexistente, etc) ou 0 caso ok.
      */
  float getFloatVoxel( DVPVolumeData* spuVolume, int x, int y, int z )
  {
      void* retVoxel = NULL;
      spuVolume->getVoxel( x, y, z, retVoxel );
      float* retVoxelf = (float*)retVoxel;
      float valueAct = retVoxelf[0];
      return valueAct;
  }


  /**
      * Recebe as tres coordenadas de um determinado voxel e insere seu valor no vetor do volume.
      * @param Inline Coordenada X do voxel.
      * @param Crossline Coordenada Y do voxel.
      * @param Time Coordenada Z do voxel.
      * @param GNG_Voxel Valor inserido do voxel.
      * @return Retorna -1 em caso de erro (parametros, volume inexistente, etc) ou 0 caso ok.
      */
  void setFloatVoxel( DVPVolumeData* spuVolume, int x, int y, int z, float voxel )
  {
      float voxelf = voxel;
      spuVolume->setVoxel(x, y, z, (void*) (&(voxelf)) );
  }


  


  /******************************************************************************************************************
                Apres Terca-feira: FIM
                Apres Terca-feira: FIM
                Apres Terca-feira: FIM
   ******************************************************************************************************************/



  /**
    * Encontra uma porcao de um horizonte a partir de UMA UNICA semente.
    */
  int findHorizonPortion( Vector3Di& seed, int window, float maxRelDist, std::vector< Vector3Di >& Horizon )
  {
    /* Gerando uma semente nova, para os nmeros aleatrios: */
    srand( (unsigned)time( NULL ) );

    if( (_spuOutRawProcessedVolume_Smp != NULL) && (_spuOutRawProcessedVolume_Smp->_vSmp_Size != 2*window+1) )
    {
      delete _spuOutRawProcessedVolume_Smp;
      _spuOutRawProcessedVolume_Smp = NULL;
    }

    if( _spuOutRawProcessedVolume_Smp == NULL )
      _spuOutRawProcessedVolume_Smp = new Volume_Smp( _spuOutRawProcessedVolume, 2*window+1, Vector3Di(0, 0, 0) );

    // Em usedPositions teremos toda essa porcao do horizonte mapeado ao final do processo:
    std::vector<std::vector<int> >  usedPositions;
    usedPositions.resize(_spuEntryVolumeDim[0]);
    for (int i = 0; i < _spuEntryVolumeDim[0]; ++i)
    {
      usedPositions[i].resize(_spuEntryVolumeDim[1]);
      for (int j = 0; j < _spuEntryVolumeDim[1]; ++j)
        usedPositions[i][j] = -1;
    }

    // Descobrindo o sinal do voxel semente, e verificando se e realmente um voxel de pico.
    // Caso nao seja de pico mapeia, mas emite uma mensagem:
    // Obtendo os tres valores, e verificando se a amostra e realmente do tipo correta:
    Volume_Smp::SamplesType type;
    {
      int x, y, z;
      seed.getValue( x, y, z );

      float valueAct;

      // Invertendo o valor de z:
      z = _spuEntryVolumeDim[2] - z - 1;

      void* retVoxel = NULL;
      _spuEntryVolume->getVoxel( x, y, z, retVoxel );
      float* retVoxelf = (float*)retVoxel;
      valueAct = retVoxelf[0];

      if( valueAct <= 0 )
        type = Volume_Smp::NEGATIVE_PEAKS;
      if( valueAct > 0 )
        type = Volume_Smp::POSITIVE_PEAKS;

      // Testa cada um dos vizinhos retornados:
      //      if( isPeak( seed, type) == false )
      //        std::cout << "SPC()::findHorizonPortion(). Warning! Semente nao e de pico" << std::endl;
    }


    // MultiMap que ira conter todos os nos ordenados por relevancia:
    std::list<Vector3Di> hzList;
    hzList.push_back( seed );
    usedPositions[seed[0]][seed[1]] = seed[2];
    std::list<Vector3Di>::iterator hzListIt = hzList.begin();

    while( hzListIt != hzList.end() )
    {
      Vector3Di thisSample = (*hzListIt);

      std::vector<Vector3Di> neihborsRet;
      std::vector<float> distsRet;
      std::vector<float> relDistsRet;
      if( getImmediateWindowNeighbors( seed, thisSample, window, neihborsRet, distsRet, relDistsRet, true, type ) == 0 )
      {
        for( unsigned int i=0 ; i<neihborsRet.size() ; i++ )
        {
          if( relDistsRet[i] > maxRelDist )
            continue;
          Vector3Di thisNeighbor = neihborsRet[i];
          // Caso esse vizinho ja tenha sido incluso, apenas continua:
          if( usedPositions[thisNeighbor[0]][thisNeighbor[1]] != -1 )
            continue;
          // Caso contrario, inclui esse vizinho na lista e marca como descoberto:
          hzList.push_back( thisNeighbor );
          usedPositions[thisNeighbor[0]][thisNeighbor[1]] = thisNeighbor[2];
        }
        neihborsRet.clear();
      }
      hzListIt++;
    }

    // Caso j� exista algum horizonte mapeado do passo anterior, seus voxels devem ser marcados:
    std::vector<std::vector<int> >  globalUsedPositions;
    globalUsedPositions.resize(_spuEntryVolumeDim[0]);
    for (int i = 0; i < _spuEntryVolumeDim[0]; ++i)
    {
      globalUsedPositions[i].resize(_spuEntryVolumeDim[1]);
      for (int j = 0; j < _spuEntryVolumeDim[1]; ++j)
        globalUsedPositions[i][j] = -1;
    }
    // Inserindo em globalUsedPositions todos os horizontes marcados do passo anterior:
    if( Horizon.size() > 1 )
    {
      for( unsigned int i=0 ; i<Horizon.size() ; i++ )
      {
        Vector3Di thisVoxel = Horizon[i];
        globalUsedPositions[thisVoxel[0]][thisVoxel[1]] = thisVoxel[2];
      }
    }

    int addedVoxels = 0;
    // Ao final do processo, inclui da nova parte mapeada somente os novos voxels desse horizonte:
    if( hzList.size() > 1 )
    {
      for( hzListIt = hzList.begin(); hzListIt != hzList.end(); ++hzListIt )
      {
        Vector3Di thisValue = *hzListIt;
        thisValue[2] = _spuEntryVolumeDim[2] - thisValue[2] - 1;
        if( globalUsedPositions[thisValue[0]][thisValue[1]] == -1 )
        {
          Horizon.push_back( thisValue );
          addedVoxels++;
        }
      }
    }

    std::cout << "addedVoxels: " << addedVoxels << std::endl;

    return 0;
  }


  /**
    * Encontra uma camada proporcional regional qualquer a partir de UMA UNICA semente.
    */
  int findRegionLayerPortion( Vector3Di& seed, int window, float maxRelDist, std::vector< Vector3Di >& Horizon )
  {
    /* Gerando uma semente nova, para os nmeros aleatrios: */
    srand( (unsigned)time( NULL ) );

    if( (_spuOutRawProcessedVolume_Smp != NULL) && (_spuOutRawProcessedVolume_Smp->_vSmp_Size != 2*window+1) )
    {
      delete _spuOutRawProcessedVolume_Smp;
      _spuOutRawProcessedVolume_Smp = NULL;
    }

    if( _spuOutRawProcessedVolume_Smp == NULL )
      _spuOutRawProcessedVolume_Smp = new Volume_Smp( _spuOutRawProcessedVolume, 2*window+1, Vector3Di(0, 0, 0) );

    // Em usedPositions teremos toda essa porcao do horizonte mapeado ao final do processo:
    std::vector<std::vector<int> >  usedPositions;
    usedPositions.resize(_spuEntryVolumeDim[0]);
    for (int i = 0; i < _spuEntryVolumeDim[0]; ++i)
    {
      usedPositions[i].resize(_spuEntryVolumeDim[1]);
      for (int j = 0; j < _spuEntryVolumeDim[1]; ++j)
        usedPositions[i][j] = -1;
    }

    // OBS: PARA REGIONLAYERS NAO E NECESSARIO SABER O SINAL DA SEMENTE!!!

    // MultiMap que ira conter todos os nos ordenados por relevancia:
    std::list<Vector3Di> hzList;
    hzList.push_back( seed );
    usedPositions[seed[0]][seed[1]] = seed[2];
    std::list<Vector3Di>::iterator hzListIt = hzList.begin();

    while( hzListIt != hzList.end() )
    {
      Vector3Di thisSample = (*hzListIt);

      std::vector<Vector3Di> neihborsRet;
      std::vector<float> distsRet;
      std::vector<float> relDistsRet;
      if( getImmediateWindowNeighbors( seed, thisSample, window, neihborsRet, distsRet, relDistsRet, false ) == 0 )
      {
        for( unsigned int i=0 ; i<neihborsRet.size() ; i++ )
        {
          if( relDistsRet[i] > maxRelDist )
            continue;
          Vector3Di thisNeighbor = neihborsRet[i];
          // Caso esse vizinho ja tenha sido incluso, apenas continua:
          if( usedPositions[thisNeighbor[0]][thisNeighbor[1]] != -1 )
            continue;
          // Caso contrario, inclui esse vizinho na lista e marca como descoberto:
          hzList.push_back( thisNeighbor );
          usedPositions[thisNeighbor[0]][thisNeighbor[1]] = thisNeighbor[2];
        }
        neihborsRet.clear();
      }
      hzListIt++;
    }

    // Caso j� exista algum horizonte mapeado do passo anterior, seus voxels devem ser marcados:
    std::vector<std::vector<int> >  globalUsedPositions;
    globalUsedPositions.resize(_spuEntryVolumeDim[0]);
    for (int i = 0; i < _spuEntryVolumeDim[0]; ++i)
    {
      globalUsedPositions[i].resize(_spuEntryVolumeDim[1]);
      for (int j = 0; j < _spuEntryVolumeDim[1]; ++j)
        globalUsedPositions[i][j] = -1;
    }
    // Inserindo em globalUsedPositions todos os horizontes marcados do passo anterior:
    if( Horizon.size() > 1 )
    {
      for( unsigned int i=0 ; i<Horizon.size() ; i++ )
      {
        Vector3Di thisVoxel = Horizon[i];
        globalUsedPositions[thisVoxel[0]][thisVoxel[1]] = thisVoxel[2];
      }
    }

    int addedVoxels = 0;
    // Ao final do processo, inclui da nova parte mapeada somente os novos voxels desse horizonte:
    if( hzList.size() > 1 )
    {
      for( hzListIt = hzList.begin(); hzListIt != hzList.end(); ++hzListIt )
      {
        Vector3Di thisValue = *hzListIt;
        thisValue[2] = _spuEntryVolumeDim[2] - thisValue[2] - 1;
        if( globalUsedPositions[thisValue[0]][thisValue[1]] == -1 )
        {
          Horizon.push_back( thisValue );
          addedVoxels++;
        }
      }
    }

    std::cout << "addedVoxels: " << addedVoxels << std::endl;

    return 0;
  }


  /**
    * Funcao auxiliar. Retorna se uma coordenada e de pico do tipo definido.
    */
  bool isSameSignal( Vector3Di seed, Volume_Smp::SamplesType type )
  {
    // Verificando se e realmente um voxel de pico:
    // Obtendo os tres valores, e verificando se a amostra e realmente do tipo correta:
    int x, y, z;
    seed.getValue( x, y, z );

    // Caso nao seja uma amostra valida, nao e preciso testar se e pico:
    float centralAmpVoxel;
    float* sample = _spuVolume_Smp->getSample( x, y, z, centralAmpVoxel );
    if( sample == NULL )
      return false;

    //        for( int i=0 ; i<_spuSamplesSize ; i++ )
    //        {
    //            std::cout << i << ". " << sample[i] << std::endl;
    //        }

    float valueAct;

    // Modificando Z:
    z = _spuEntryVolumeDim[2]-z-1;

    void* retVoxel = NULL;
    _spuEntryVolume->getVoxel( x, y, z, retVoxel );
    float* retVoxelf = (float*)retVoxel;
    valueAct = retVoxelf[0];

    bool ok = false;
    if( type == Volume_Smp::NEGATIVE_PEAKS )
      if( valueAct<0 )
        ok = true;
    if( type == Volume_Smp::POSITIVE_PEAKS )
      if( valueAct>0 )
        ok = true;

    return ok;
  }


  /**
    * Recebe dois números e retorna se possuem o mesmo sinal (0 e considerado negativo).
    */
  bool isSameValueSignal( float a, float b )
  {
    if( (a>0) && (b>0) )
      return true;
    if( (a<=0) && (b<=0) )
      return true;
    return false;
  }


  /**
    * Funcao auxiliar. Recebe uma coordenada de tipo definido (amplitude positiva ou negativa).
    * Encontra e retorna seu pico de amplitude correspondente.
    * @param seed Coordenada a ser analizada. Caso seu pico seja encontrado, seu valor sera atualizado.
    */
  bool toPeak( Vector3Di& seed, Volume_Smp::SamplesType type )
  {
    // Caso a amostra nao seja do mesmo sinal do tipo recebido, retorna erro:
    if( isSameSignal( seed, type) == false )
      return false;

    // Verificando se e realmente um voxel de pico:
    // Obtendo os tres valores, e verificando se a amostra e realmente do tipo correta:
    int x, y, z;
    seed.getValue( x, y, z );

    float centralAmpVoxel;
    float* sample = _spuVolume_Smp->getSample( x, y, z, centralAmpVoxel );
    if( sample == NULL )
      return false;
    //      for( int i=0 ; i<_spuSamplesSize ; i++ )
    //      {
    //        std::cout << i << ". " << sample[i] << std::endl;
    //      }

    // Modificando Z:
    int seedZ = _spuEntryVolumeDim[2]-z-1;

    // Pega o valor e sinal do voxel recebido:
    void* retVoxel = NULL;
    _spuEntryVolume->getVoxel( x, y, seedZ, retVoxel );
    float* retVoxelf = (float*)retVoxel;
    float valueAct = retVoxelf[0];

    if( (type==Volume_Smp::POSITIVE_PEAKS) && (valueAct < 0) )
      return false;
    if( (type==Volume_Smp::NEGATIVE_PEAKS) && (valueAct > 0) )
      return false;

    // Encontrando o comeco da faixa de valores positivos:
    int begin, end;

    // inicializando o z de procura:
    z = seedZ;
    z--;
    retVoxel = NULL;
    _spuEntryVolume->getVoxel( x, y, z, retVoxel );
    retVoxelf = (float*)retVoxel;
    float valueAnt = retVoxelf[0];

    while( (isSameValueSignal(valueAct, valueAnt)) && _spuEntryVolume->isValidCoord( x, y, z ) )
    {
      z--;
      retVoxel = NULL;
      _spuEntryVolume->getVoxel( x, y, z, retVoxel );
      retVoxelf = (float*)retVoxel;
      valueAnt = retVoxelf[0];
    }
    begin = z;

    // Reinicializando Z:
    z = seedZ;
    z++;
    retVoxel = NULL;
    _spuEntryVolume->getVoxel( x, y, z, retVoxel );
    retVoxelf = (float*)retVoxel;
    float valuePos = retVoxelf[0];

    while( (isSameValueSignal(valueAct, valuePos)) && _spuEntryVolume->isValidCoord( x, y, z ) )
    {
      z++;
      retVoxel = NULL;
      _spuEntryVolume->getVoxel( x, y, z, retVoxel );
      retVoxelf = (float*)retVoxel;
      valuePos = retVoxelf[0];
    }
    end = z;

    // Caso estejamos arredondando um pico positivo:
    if( type==Volume_Smp::POSITIVE_PEAKS )
    {
      // Percorre todos os voxels do intervalo, retornando o de maior amplitude:
      float maxValue = std::numeric_limits<float>::min();
      int maxValueIndex = -1;
      for( int i=begin ; i<=end ; i++ )
      {
        retVoxel = NULL;
        _spuEntryVolume->getVoxel( x, y, i, retVoxel );
        retVoxelf = (float*)retVoxel;
        valueAct = retVoxelf[0];
        // Armazena o maior valor j� encontrado:
        if( maxValue < valueAct )
        {
          maxValue = valueAct;
          maxValueIndex = i;
        }
      }
      seed[2] = _spuEntryVolumeDim[2]-maxValueIndex-1;;
    }

    // Caso estejamos arredondando um pico negativo:
    if( type==Volume_Smp::NEGATIVE_PEAKS )
    {
      // Percorre todos os voxels do intervalo, retornando o de maior amplitude:
      float minValue = std::numeric_limits<float>::max();
      int minValueIndex = -1;
      for( int i=begin ; i<=end ; i++ )
      {
        retVoxel = NULL;
        _spuEntryVolume->getVoxel( x, y, i, retVoxel );
        retVoxelf = (float*)retVoxel;
        valueAct = retVoxelf[0];
        // Armazena o maior valor j� encontrado:
        if( minValue > valueAct )
        {
          minValue = valueAct;
          minValueIndex = i;
        }
      }
      seed[2] = _spuEntryVolumeDim[2]-minValueIndex-1;
    }


    std::cout << "Chega ao fim do toPeak." << std::endl;

    return true;
  }


  /**
    * Encontra um horizonte composto somente de picos de amplitude, a partir da semente.
    */
  int findPeaksHorizonPortion( Vector3Di seed, int window, float maxDist, float maxRelDist,
                               std::vector< Vector3Di >& Horizon, Volume_Smp::SamplesType type )
  {
    if( (_spuOutRawProcessedVolume_Smp != NULL) && (_spuOutRawProcessedVolume_Smp->_vSmp_Size != 2*window+1) )
    {
      delete _spuOutRawProcessedVolume_Smp;
      _spuOutRawProcessedVolume_Smp = NULL;
    }

    if( _spuOutRawProcessedVolume_Smp == NULL )
      _spuOutRawProcessedVolume_Smp = new Volume_Smp( _spuOutRawProcessedVolume, 2*window+1, Vector3Di(0, 0, 0) );

    // Em usedPositions teremos o horizonte mapeado ao final do processo:
    std::vector<std::vector<Vector3Di> >  usedPositions;
    usedPositions.resize(_spuEntryVolumeDim[0]);
    for (int i = 0; i < _spuEntryVolumeDim[0]; ++i)
    {
      usedPositions[i].resize(_spuEntryVolumeDim[1]);
      for (int j = 0; j < _spuEntryVolumeDim[1]; ++j)
        usedPositions[i][j] = Vector3Di(i, j, -1);
    }


    // Verificando se e realmente um voxel de pico:
    // Obtendo os tres valores, e verificando se a amostra e realmente do tipo correta:
    {
      int x, y, z;
      seed.getValue( x, y, z );
      float valueAnt;
      float valueAct;
      float valuePos;

      // Invertendo o valor de z:
      z = _spuEntryVolumeDim[2] - z - 1;

      void* retVoxel = NULL;
      _spuEntryVolume->getVoxel( x, y, z-1, retVoxel );
      float* retVoxelf = (float*)retVoxel;
      valueAnt = retVoxelf[0];

      retVoxel = NULL;
      _spuEntryVolume->getVoxel( x, y, z, retVoxel );
      retVoxelf = (float*)retVoxel;
      valueAct = retVoxelf[0];

      retVoxel = NULL;
      _spuEntryVolume->getVoxel( x, y, z+1, retVoxel );
      retVoxelf = (float*)retVoxel;
      valuePos = retVoxelf[0];

      int numErrors = 0;
      if( (valueAct<0) || (valueAct<valueAnt) || (valueAct<valuePos) )
        numErrors++;
    }

    // Verificando se e realmente um voxel de pico:
    // Obtendo os tres valores, e verificando se a amostra e realmente do tipo correta:
    float peakVoxel;
    if( isPeak( seed, type, peakVoxel ) == false )
      return -1;

    // MultiMap que ira conter todos os nos ordenados por relevancia:
    std::list<Vector3Di> hzList;
    hzList.push_back( seed );
    usedPositions[seed[0]][seed[1]] = seed;
    std::list<Vector3Di>::iterator hzListIt = hzList.begin();

    while( hzListIt != hzList.end() )
    {
      Vector3Di thisSample = (*hzListIt);

      std::vector<Vector3Di> neihborsRet;
      std::vector<float> distsRet;
      std::vector<float> relDistsRet;
      if( getImmediateWindowNeighbors( seed, thisSample, window, neihborsRet, distsRet, relDistsRet, true, type ) == 0 )
      {
        for( unsigned int i=0 ; i<neihborsRet.size() ; i++ )
        {
          if( relDistsRet[i] > maxRelDist )
            continue;
          Vector3Di thisNeighbor = neihborsRet[i];

          // Testa cada um dos vizinhos retornados:
          float peakVoxel;
          if( isPeak( thisNeighbor, type, peakVoxel ) == false )
          {
            std::cout << "SPC()::findPeaksHorizonPortion(). Erro! Vizinho nao e de pico" << std::endl;
            continue;
          }

          Vector3Di testPt = (usedPositions[thisNeighbor[0]][thisNeighbor[1]]);

          // Caso seja -1, pode ser adicionado sem problemas:
          int z = testPt[2];
          if( z == -1 )
          {
            // Caso contrario, inclui esse vizinho na lista e marca como descoberto:
            usedPositions[thisNeighbor[0]][thisNeighbor[1]] = thisNeighbor;
            hzList.push_back( thisNeighbor );
            continue;
          }
          if( z == thisNeighbor[2] )
            continue;
          if( z != thisNeighbor[2] )
            usedPositions[thisNeighbor[0]][thisNeighbor[1]]  = Vector3Di(thisNeighbor[0], thisNeighbor[1], -2);
        }
        neihborsRet.clear();
      }
      hzListIt++;

    }

    // Ao final do processo, retorna esse horizonte:
    Horizon.clear();
    for( hzListIt = hzList.begin(); hzListIt != hzList.end(); ++hzListIt )
    {
      Vector3Di thisValue = *hzListIt;
      thisValue[2] = (_spuEntryVolumeDim[2]-thisValue[2]-1);
      Horizon.push_back( thisValue );
    }

    return 0;
  }
  
  
  bool intersects( Vector3Di minpt, Vector3Di maxpt, Vector3Di pt )
  {
    return !(pt[0] < minpt[0] || pt[0] > maxpt[0] ||
             pt[1] < minpt[1] || pt[1] > maxpt[1] ||
             pt[2] < minpt[2] || pt[2] > maxpt[2]);
  }
 
};

}

#endif // SEISMICPROCESSINGCONTROLER_H


