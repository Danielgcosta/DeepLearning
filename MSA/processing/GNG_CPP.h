#ifndef GNG_CPP_H
#define GNG_CPP_H

#include <math.h>
#include <limits>
#include <iostream>
#include <stdio.h>
#include <sstream>
#include <map>

//#include "Vol_Samples.h"
#include "../data/Volume_Smp.h"
#include "Cluster.h"
#include "Graph_Algorithms.h"

namespace MSA
{

/**
  * Classe auxiliar para representacÃ£o das arestas entre clusters de GNG.
  */
class GNG_Edge
{
public:

  Cluster* cl1;
  Cluster* cl2;
  int age;


  /**
    * Construtor.
    */
  GNG_Edge( Cluster* c1, Cluster* c2, int a );


  /**
    * Retorna um ponteiro contendo uma cÃ³pia da instÃ¢ncia.
    */
  GNG_Edge* copy();


  /**
    * Caso receba um cluster como parÃ¢metro, retorna um ponteiro para o outro.
    */
  Cluster* getOtherNode( Cluster* c );


  /**
    * Retorna ponteiros para os nÃ³s nas extreminades da aresta.
    */
  void getClusters( Cluster*& c1, Cluster*& c2 );


  /**
    * Reset the edge.
    */
  void Reset();


  /**
    * Reset the edge.
    */
  void ResetAge();


  /**
    * Incrementa a idade da aresta de 1 caso um dos clusters recebidos seja sua extremidade.
    */
  int incAge( Cluster* c );


  /**
    * Retorna a idade da aresta.
    */
  int getAge();


  /**
    * Retorna true se o cluster recebido Ã© extremidade da aresta.
    */
  bool isNode( Cluster* c );


  /**
    * Destrutor.
    */
  ~GNG_Edge();

};



/**
  * Classe auxiliar de manutencÃ£o dos dois clusters mais prÃ³ximos de uma determinada amostra.
  */
class BMU_SegBMU
{
public:
  Cluster* cl1;
  float dist_cl1;
  Cluster* cl2;
  float dist_cl2;
  Cluster* cl3;
  float dist_cl3;
  std::multimap<float, Cluster*> clustersMultiMap;


  /**
    * Construtor.
    */
  BMU_SegBMU( Cluster* c1, float dist_c1, Cluster* c2=NULL, float dist_c2=-1 );


  /**
    * Destrutor.
    */
  ~BMU_SegBMU();


  /**
    * Atualiza os dois de melhores distÃ¢ncias.
    */
  void UpDate( Cluster* c, float dist_c );


  /**
    * Atualiza os dois de melhores distÃ¢ncias.
    */
  void UpDate2Neighbors( Cluster* c, float dist_c );

};


/**
  * Classe de Growing Neural Gas.
  */
class GNG_CPP
{
protected:

  // O mesmo vetor de arestas Ã© recriado apÃ³s a execucÃ£o do treinamento (permitindo salvar a instÃ¢ncia em memÃ³ria).
  // Nesse novo vetor, apenas existe a indicacÃ£o da existÃªncia ou nÃ£o da aresta:
  std::vector< std::vector<int> > _gEdgesMatrix;

public:

  int _gClustersIdInit;               //< Valor inicial dos id's dos clusters.
  int _gNumClusters;                  //< NÃºmero de clusters depois do treinamento.
  int _gDim;                          //< DimensÃ£o das amostras.
  std::vector<Cluster*>  _gCenters;   //< Vector de Clusters.
  std::vector<GNG_Edge*> _gEdges;     //< Vetor de arestas.

  std::vector< std::vector<int> > _gDistanceByEdges;  //< Matriz de distÃ¢ncias pelo nÃºmero de arestas.

  // ParÃ¢metros de aprendizado do GNG:
  int   _gLambda;
  float _gEb;
  float _gEn;
  float _gAlpha;
  int   _gAmax;
  float _gD;


  /**
    * Retorna uma cÃ³pia da instÃ¢ncia atual.
    */
  GNG_CPP* copy()
  {
    GNG_CPP* retCopy = new GNG_CPP( _gNumClusters, _gDim );
    // Copiando variÃ¡veis:
    retCopy->_gClustersIdInit = _gClustersIdInit;
    retCopy->_gLambda = _gLambda;
    retCopy->_gEb = _gEb;
    retCopy->_gEn = _gEn;
    retCopy->_gAlpha = _gAlpha;
    retCopy->_gAmax = _gAmax;
    retCopy->_gD = _gD;

    // Copiando as arestras (matriz de arestas numÃ©ricas Ã© opcional), caso jÃ¡ tenha sido calculada:
    if( _gEdgesMatrix.size() > 0 )
    {
      int dimMatrix = _gCenters.size();
      retCopy->_gEdgesMatrix = std::vector<std::vector<int> > (dimMatrix,std::vector<int>(dimMatrix,0));
      for( int i=0 ; i<dimMatrix ; i++ )
        for( int j=0 ; j<dimMatrix ; j++ )
          retCopy->_gEdgesMatrix[i][j] = _gEdgesMatrix[i][j];
    }

    // Armazena o nÃºmero de clusters:
    retCopy->_gNumClusters = _gNumClusters;
    int num_clusters = _gNumClusters;

    // Alocando o espaco necessÃ¡rio ao vetor de clusters (retCopy->_gCenters):
    retCopy->_gCenters.reserve( num_clusters );
    retCopy->_gCenters.resize( num_clusters );

    // Recriando o mapa contendo os clusters ordenados por erro:
    int i=0;
    for( ; i<num_clusters ; i++ )
    {
      Cluster* cluster = (_gCenters[i])->copy();
      retCopy->_gCenters[i] = cluster;
    }

    // Recriando o vetor de arestas (retCopy->_gEdges) a partir da matrix booleana:
    if( retCopy->createEdgesVectorFromMatrix() == -1 )
    {
      delete retCopy;
      return NULL;
    }

    // Constroi a matriz de distÃ¢ncias por arestas:
    retCopy->findDistancesByEdges();

    return retCopy;
  }


  /**
    * Retorna a norna de um vetor qualquer.
    */
  float lenght( float* vec1 );


  /**
    * Normaliza o vetor recebido.
    */
  float normalize( float* vec1 );


  /**
    * Encontra o produto interno entre dois vetores.
    */
  float inner( float* vec1, float* vec2 );


  /**
  * Retorna a distÃ¢ncia Euclideana.
  */
  float fEuclideanDist( float* vec1, float* vec2, int size = -1 );


  /**
    * Retorna a distancia Euclideana de correlacao cruzada.
    */
  float fCrossCorrEuclideanDist( float* vec1, float* vec2, int size );


  /**
    * Retorna a correlacao cruzada entre dois vetores.
    */
  float fCrossCorrelationDist( float* vec1, float* vec2, int size );


  /**
  * Retorna o quadrado da distancia de dois vetores.
  */
  float fQuadDist( float* vec1, float* vec2 );


  /**
    * Calcula a distÃ¢ncia das amostras ponderada pelo desvio-padrÃ£o.
    * Essa funcÃ£o permite encontrar amostras que, apesar de classificadas como pertencentes ao cluster,
    * podem ser consideradas outliers.
    * @return Retorna a distÃ¢ncia ponderada, ou -1 em caso de erro.
    */
  float fWeightedDist( float* vec1, float* vec2, float* stdDev );



  /**
    * Retorna o nÃºmero de nÃ³s.
    */
  int getNumClusters();


  /**
    * Construtor.
    */
  GNG_CPP( int gNumClusters, int gDim, int gClustersIdInit=0 );


  /**
    * Construtor.
    */
  GNG_CPP( const char* filename );


  /**
    * Destrutor.
    */
  ~GNG_CPP();


  /**
    * ObtÃ©m um ponteiro para o vetor de clusters.
    */
  std::vector<Cluster*>& getClustersVector();


  /**
    * Inicia o treinamento (gera duas unidades em posicÃµes aleatÃ³rias).
    */
  int createFirstTwoClusters( float* sp1, float* sp2 );


  /**
    * Adiciona um novo cluster.
    */
  void addNode();


  /**
    * Recebe uma amostra e encontra o cluster mais prÃ³ximo.
    */
  int getBMU( float* smp, Cluster* &retCluster, Cluster* &retClusterNeighbor,  Cluster* &retClusterSecNeighbor,
              float& error, float& errorNeighbor, float& errorSecNeighbor );


  /**
      * Recebe uma amostra e encontra o cluster mais prÃ³ximo segundo o critÃ©rio de distÃ¢ncia euclideana.
      * Permite modificar a dimensÃ£o das amostras durante a procura.
      */
  int getBMU( float* smp, Cluster* &retCluster, float& error, int size );


  /**
    * ObtÃ©m a distÃ¢ncia entre dois clusters como o nÃºmero de distÃ¢ncias mÃ©dias entre o cluster referÃªncia (o primeiro) e o segundo cluster.
    */
  float fNumMeansDistance( int cluster1Pos, int cluster2Pos );


  /**
    * ObtÃ©m a distÃ¢ncia entre dois clusters como o nÃºmero de distÃ¢ncias mÃ©dias entre o cluster referÃªncia (o primeiro) e o segundo cluster.
    */
  float fNumMeansDistance( Cluster* c1, Cluster* c2 );


  /**
    * Ordena o vetor de nÃ³s pela distÃ¢ncia mÃ©dia do cluster aos seus vizinhos topolÃ³gicos. Insere o Id do cluster a partir da sua nova posicÃ£o no vetor.
    */
  void orderingByNeighborsDist();


  /**
    * Calcula a distÃ¢ncia entre os clusters atravÃ©s do nÃºmero de arestas percorridas entre um cluster e outro.
    */
  int findDistancesByEdges();


  /**
    * Cria um vetor descrevendo as arestas de ligacÃ£o entre os clusters, baseado no id numÃ©rico dos clusters.
    */
  void createNumericEdgesMatrix( std::vector<std::vector<int> >& edgesMatrix );


  /**
    * Recria o vetor de arestas a partir da matriz booleana.
    */
  int createEdgesVectorFromMatrix();

  /**
    * FuncÃ£o auxiliar no processo de treinamento.
    * Utilizada tanto pela funcÃ£o de treinamento que recebe um vetor de amostras, quanto pela que recebe
    * uma instÃ¢ncia de VolumeSamples.
    */
  void auxiliaryTrainStep( float* smp, bool& trained, unsigned int& numTotalTrains );


  /**
    * O treinamento pode ser feito passo a passo caso o vetor inteiro, contendo todas as amostras de treino, nÃ£o esteja disponÃ­vel.
    * Isso poderÃ¡ ser feito atravÃ©s da utilizacÃ£o de duas funcÃµes: a primeira (initTrain) inicializa as variÃ¡veis de treinamento, devendo
    * receber duas amostras iniciais. A segunda serÃ¡ a funcÃ£o "auxiliaryTrainStep", que deverÃ¡ ser chamada a cada amostra a ser treinada,
    * iterando atÃ© que retorne seu parÃ¢metro "bool& trained" como true. Isso indica que o treinamento terÃ¡ sido finalizado.
    */
  int initTrain( int gLambda, float gEb, float gEn, float gAlpha, int gAmax, float gD,
                           int& numExecutedTrains, float* sp1,  float* sp2, int nneurons = -1 );


  /**
    * Executa o treinamento recebendo um vetor de treinamento que contÃ©m todas as amostras que deverÃ£o ser treinadas.
    */
  int allSamplesTrain( std::vector<float*> samples, int gLambda, float gEb, float gEn, float gAlpha, int gAmax, float gD,
             int& numExecutedTrains, int nneurons = -1 );


  /**
    * Termina a etapa de treinamento, calculando informacÃµes importante mas que nÃ£o fazem parte do algoritmo
    * padrÃ£o de treinamento do GNG. Cria a matriz numÃ©rica de arestas, ordena o vetor de vÃ©rtices (), classifica
    * o vetor de amostras recebido nos clusters formados, e com isso calcula estatÃ­sticas importantes, tais como
    * mÃ©dia, variÃ¢ncia, matriz de covariÃ¢ncias entre os componentes do cluster, entre outras informacÃµes.
    * @param samples Vetor contendo todas as amostras que serÃ£o utilizadas nos cÃ¡lculos.
    * @param edgesMatrix Matriz de arestas numÃ©ricas entre os clusters, a ser retornada.
    * @return Retorna -1 em caso de erro, ou 0 caso ok.
    */
  int finalizeTrain( std::vector<float*> samples, std::vector<std::vector<int> >& edgesMatrix,
                     std::vector<std::vector<float*> >& classifiedSamplesVector,
                     bool retClassifiedSamples=false );


  /**
    * Executa o treinamento.
    */
  int allSamplesTrain( Volume_Smp* samples, int gLambda, float gEb, float gEn, float gAlpha, int gAmax, float gD,
                      int& numExecutedTrains, int nneurons = -1, Volume_Smp::SamplesType smpType = Volume_Smp::UNDEFINED );


  /**
      * Termina a etapa de treinamento, calculando informacÃµes importantes mas que nÃ£o fazem parte do algoritmo
      * padrÃ£o de treinamento do GNG. Cria a matriz numÃ©rica de arestas, ordena o vetor de vÃ©rtices (), classifica
      * o vetor de amostras recebido nos clusters formados, e com isso calcula estatÃ­sticas importantes, tais como
      * mÃ©dia, variÃ¢ncia, matriz de covariÃ¢ncias entre os componentes do cluster, entre outras informacÃµes.
      * Caso o vetor de amostras seja vazio, somente ordena os nÃ³s e cria as matrizes de arestas e
      * de distÃ¢ncia por arestas (sem calcular as estatÃ­sticas).
      * @param samples Vetor contendo todas as amostras que serÃ£o utilizadas nos cÃ¡lculos.
      * @param edgesMatrix Matriz de arestas numÃ©ricas entre os clusters, a ser retornada.
      * @return Retorna -1 em caso de erro, ou 0 tenha somente ordenado os nÃ³s e criado as matrizes de arestas e
      * de distÃ¢ncia por arestas (sem calcular as estatÃ­sticas), ou 1 caso tenha tambÃ©m calculado mÃ©dias, variÃ¢nciaes e
      * matrizes de covariÃ¢ncia dos clusters utilizando as amostras.
      */
  int finalizeTrain( Volume_Smp* samples, int numSamplesUsed, std::vector<std::vector<int> >& edgesMatrix,
                              std::vector<std::vector<float*> >& classifiedSamplesVector,
                              bool retClassifiedSamples=false );


  /**
    * FuncÃ£o auxiliar. Converte de clusterId para o Ã­ndice do cluster num vetor.
    */
  int getClusterPosFromId( int clusterId );


  /**
    * FuncÃ£o auxiliar. Converte de clusterId para o Ã­ndice do cluster num vetor.
    */
  Cluster* getClusterFromClusterId( int clusterId );


  /**
    * FuncÃ£o auxiliar. Recebe um vetor contendo um conjunto de amostras classificadas em cada cluster (pelo clusterId)
    * e encontra, com base nesse vetor, os valores de mÃ©dias e variÃ¢ncias.
    */
  int getStatisticsFromVector( std::vector<std::vector<float*> >& eachClusterSamplesVector );



#ifdef USING_OPENCV
  /**
    * Estimated distortion associated to fitting K centers to the dataset.
    */
  double JumpMethod( std::vector<std::vector<float*> > eachClusterSamplesVector, int samplesSize, double** covarMatrixInv = NULL )
  {
    // Caso o vetor nÃ£o possua o nÃºmero de clusters:
    if( eachClusterSamplesVector.size() != _gCenters.size() )
      return -1;
    // Caso vetor de amostras ou de clusters seja vazio:
    if( (eachClusterSamplesVector.size() == 0) || (_gCenters.size() == 0) )
      return -1;

    // Encontra a distorcÃ£o total desse cluster em relacÃ£o Ã s suas amostras:
    double totalSamples = 0;
    for( unsigned int i = 0; i < _gCenters.size() ; i++ )
      totalSamples += (eachClusterSamplesVector[i]).size();

    double distortion = 0;
    for( unsigned int i = 0; i < _gCenters.size() ; i++ )
    {
      Cluster* thisCluster = _gCenters[i];
      double thisDistortion = thisCluster->getClusterEstimatedDistortion( eachClusterSamplesVector[i], samplesSize, covarMatrixInv );
      distortion += thisDistortion * (eachClusterSamplesVector[i]).size();
    }
    distortion = distortion/totalSamples;
    distortion = distortion/samplesSize;

    return distortion;
  }
#endif

  /**
    * Salva uma instÃ¢ncia de GNG depois de treinada.
    */
  int save( const char *filename );


  /**
    * Carrega uma instÃ¢ncia previamente treinada a partir de um arquivo.
    */
  int load( const char *filename );


  /**
    * Retorna a dimensÃ£o espacial dos clusters.
    */
  int getSamplesSize();



  /**
    * Essas trÃªs Ãºltimas funcÃµes oferecem, na verdade, o processo de treinamento em bloco tradicional
    * do algoritmo k-Means. Podem ser utilizadas para executar um ajuste fino da posicÃ£o dos clusters
    * obtida inicialmente atravÃ©s do treinamento feito pelo algoritmo de Growing Neural Gas.
    * Devem ser utilizadas opcionalmente
    */

  /**
   * k-Means - One step trainning.
   */
  inline int oneStepTrainning( float* sample, std::vector<double*>& nextCenters,
                                   std::vector<int>& nextCentersNumSamples, std::vector<double>& errors )
  {
    double bestDist = std::numeric_limits<double>::max();
    int bestIndex=-1;

    // Iterador para o vetor de clusters:
    // Uma vez que o GNG pode modificar a ordem do vetor de clusters, nÃ£o podemos utilizar
    // o clusterId como referÃªncia enquanto o treinamento nÃ£o for finalizado:
    int cont=0;
    std::vector<Cluster*>::iterator ClustersItr = _gCenters.begin();
    for( ; ClustersItr != _gCenters.end() ; ++ ClustersItr, cont++ )
    {
      Cluster* cluster = *ClustersItr;
      int ctrSize;
      float* ctrValues;
      cluster->getPos( ctrSize, ctrValues );

      double dist = fEuclideanDist( ctrValues, sample );
      if( dist < bestDist )
      {
        bestIndex = cont;
        bestDist = dist;
      }
    }

    // Soma a posicÃ£o dessa amostra Ã  nextCenters do cluster:
    for( int i=0 ; i<_gDim ; i++ )
      nextCenters[bestIndex][i] += (double)sample[i];
    nextCentersNumSamples[bestIndex]++;
    errors[bestIndex]+= bestDist;
    return bestIndex;
  }


  /**
   * k-Means - It trains one epoch.
   */
  inline int oneEpochTrainning( std::vector<float*>& samples, std::vector<double*>& nextCenters,
                                    std::vector<int>& nextCentersNumSamples, std::vector<double>& errors )
  {
    // Initialization of nextCenters:
    for( unsigned int i=0 ; i<_gCenters.size() ; i++ )
    {
      errors[i] = 0.0;
      nextCentersNumSamples[i] = 0;
      for( int j=0 ; j<_gDim ; j++ )
      {
        nextCenters[i][j] = 0.0;
      }
    }
    // It trains all the samples, obtaining the new clusters positions:
    for( unsigned int i=0 ; i<samples.size() ; i++ )
    {
      oneStepTrainning( samples[i], nextCenters, nextCentersNumSamples, errors );
    }
    return 0;
  }


  /**
 * Runs k-means on the given data points with the given initial centers.
 * It iterates up to numIterations times, and for each iteration it does, it decreases numIterations by 1.
 */
  int runKMeans( std::vector<float*>& samples, int &numIterations)
  {
    // Errors vetor:
    std::vector<double> errors;
    // NextCenters clusters vector:
    std::vector<double*>nextCenters;
    // Number of centers classified into each cluster:
    std::vector<int>nextCentersNumSamples;

    for( unsigned int i=0 ; i<_gCenters.size() ; i++ )
    {
      nextCentersNumSamples.push_back(0);
      double* thisClusterPosition = new double[_gDim];
      for( int j=0 ; j<_gDim ; j++ )
      {
        thisClusterPosition[j] = 0.0;
      }
      nextCenters.push_back(thisClusterPosition);
      errors.push_back(0.0);
    }

    bool pBest = false;
    double totalError = std::numeric_limits<double>::max();
    double eRate;
    double errMedio;
    // Train one epoch:
    for( int i=0 ; i<numIterations ; i++ )
    {
      // It trains one epoch:
      if( -1 == oneEpochTrainning( samples, nextCenters, nextCentersNumSamples, errors ) )
        return -1;

      int numtotalSamples = 0;
      for( int conta=0 ; conta<(int)(nextCentersNumSamples.size()) ; conta++ )
        numtotalSamples += nextCentersNumSamples[conta];

      // Assigning the new positions of clusters:
      std::vector<Cluster*>::iterator ClustersItr = _gCenters.begin();
      int nextCentersPos=0;
      for( ; ClustersItr != _gCenters.end() ; ++ ClustersItr, nextCentersPos++ )
      {
        Cluster* cluster = *ClustersItr;
        int ctrSize;
        float* ctrValues;
        cluster->getPos( ctrSize, ctrValues );
        for( int cont=0 ; cont<_gDim ; cont++ )
        {
          ctrValues[cont] = (float)(nextCenters[nextCentersPos][cont]) /
                            (float)(nextCentersNumSamples[nextCentersPos]);
        }
      }

      // Total error:
      double sumError = 0;
      for( int cont=0 ; cont<_gDim ; cont++ )
        sumError += errors[cont];

      // Primeira iteracÃ£o:
      if( i == 0 )
        totalError = sumError;
      else
      {
        eRate = sumError / totalError;
        // Atualiza o erro total:
        totalError = sumError;
        errMedio =  sqrt(totalError/numtotalSamples);
//        std::cout << "Iteracao. " << i << " err Total. " << totalError << " err Medio. " << errMedio << std::endl;

        // This condition finishes the training procedure:
        if( (eRate >= 0.995) && (pBest == true) )
          pBest = false;//break;

        if( (eRate >= 0.995) && (pBest == false) )
          pBest = true;

        if( eRate < 0.995)
          pBest = false;
      }

      // Initializing the nextCenters vector:
      for( unsigned int i = 0; i < _gCenters.size(); i++ )
      {
        errors[i] = 0.0;
        nextCentersNumSamples[i] = 0;
        for( int j = 0; j < _gDim; j++ )
        {
          nextCenters[i][j] = 0.0;
        }
      }
    }

//    std::cout << " err Total. " << totalError << " err Medio. " << errMedio ;//<< std::endl;
    return 0;
  }


};

}

#endif // GNG_CPP_H
