#include "GNG_CPP.h"

namespace MSA
{

/**
  * Construtor.
  */
GNG_Edge::GNG_Edge( Cluster* c1, Cluster* c2, int a )
{
    cl1 = c1;
    cl2 = c2;
    age = a;
}


/**
  * Retorna um ponteiro contendo uma copia da instancia.
  */
GNG_Edge* GNG_Edge::copy()
{
    GNG_Edge* retCopy = new GNG_Edge( cl1, cl2, age );
    return retCopy;
}


/**
  * Caso receba um cluster como parametro, retorna um ponteiro para o outro.
  */
Cluster* GNG_Edge::getOtherNode( Cluster* c )
{
    if( c == NULL )
        return NULL;
    if( c == cl1 )
        return cl2;
    if( c == cl2 )
        return cl1;
    return NULL;
}


/**
  * Retorna ponteiros para os nos nas extreminades da aresta.
  */
void GNG_Edge::getClusters( Cluster*& c1, Cluster*& c2 )
{
    c1 = cl1;
    c2 = cl2;
}



/**
  * Reset the edge.
  */
void GNG_Edge::Reset()
{
    cl1 = NULL;
    cl2 = NULL;
    age = -1;
}


/**
  * Reset the edge.
  */
void GNG_Edge::ResetAge()
{
    age = 0;
}


/**
  * Incrementa a idade da aresta de 1 caso um dos clusters recebidos seja sua extremidade.
  */
int GNG_Edge::incAge( Cluster* c )
{
    if( c == NULL )
        return -1;
    if( (c == cl1) || (c == cl2) )
        age++;
    return 0;
}


/**
  * Retorna a idade da aresta.
  */
int GNG_Edge::getAge()
{
    return age;
}


/**
  * Retorna true se o cluster recebido e extremidade da aresta.
  */
bool GNG_Edge::isNode( Cluster* c )
{
    if( (c == cl1) || (c == cl2) )
        return true;
    return false;
}


/**
  * Destrutor.
  */
GNG_Edge::~GNG_Edge()
{
}


/**
  * Construtor.
  */
BMU_SegBMU::BMU_SegBMU( Cluster* c1, float dist_c1, Cluster* c2, float dist_c2 )
{
    cl1 = NULL;
    cl2 = NULL;
    cl3 = NULL;
    dist_cl1 = std::numeric_limits<float>::max();
    dist_cl2 = std::numeric_limits<float>::max();
    dist_cl3 = std::numeric_limits<float>::max();

    if( c2 != NULL )
    {
        if( dist_c1 < dist_c2 )
        {
            cl1 = c1;
            dist_cl1 = dist_c1;
            cl2 = c2;
            dist_cl2 = dist_c2;
        }
        else
        {
            cl1 = c2;
            dist_cl1 = dist_c2;
            cl2 = c1;
            dist_cl2 = dist_c1;
        }
    }
    else
    {
        cl1 = c1;
        dist_cl1 = dist_c1;
        cl2 = NULL;
        dist_cl2 = std::numeric_limits<float>::max();
    }
}


/**
  * Destrutor.
  */
BMU_SegBMU::~BMU_SegBMU()
{
    // Limpa o Multimap:
    clustersMultiMap.clear();
}


/**
  * Atualiza os dois de melhores distancias.
  */
void BMU_SegBMU::UpDate( Cluster* c, float dist_c )
{
  if( (dist_c < dist_cl1) && (dist_c < dist_cl2) )
  {
    cl2 = cl1;
    dist_cl2 = dist_cl1;
    cl1 = c;
    dist_cl1 = dist_c;
    return;
  }

  if( dist_c < dist_cl2 )
  {
    cl2 = c;
    dist_cl2 = dist_c;
  }
  return;
}


/**
  * Atualiza os dois de melhores distancias.
  */
void BMU_SegBMU::UpDate2Neighbors( Cluster* c, float dist_c )
{
    clustersMultiMap.clear();
    // Insere o cluster que nao for nulo:
    if( c != NULL )
        clustersMultiMap.insert( std::pair<float, Cluster*>( dist_c, c ) );
    if( cl1 != NULL )
        clustersMultiMap.insert( std::pair<float, Cluster*>( dist_cl1, cl1 ) );
    if( cl2 != NULL )
        clustersMultiMap.insert( std::pair<float, Cluster*>( dist_cl2, cl2 ) );
    if( cl3 != NULL )
        clustersMultiMap.insert( std::pair<float, Cluster*>( dist_cl3, cl3 ) );

    if( (clustersMultiMap.size() != 2) && (clustersMultiMap.size() != 3) && (clustersMultiMap.size() != 4) )
        std::cout << "BMU_SegBMU::UpDate: num. de nos invalido." << std::endl;

    if( clustersMultiMap.size() == 2 )
    {
        // Insere os clusters ordenados no vetor:
        std::multimap<float, Cluster*>::iterator clustersMultiMapItr;
        clustersMultiMapItr = clustersMultiMap.begin();
        Cluster* cluster = (*clustersMultiMapItr).second;
        float ClusterDist = (*clustersMultiMapItr).first;
        cl1 = cluster;
        dist_cl1 = ClusterDist;

        ++clustersMultiMapItr;
        cluster = (*clustersMultiMapItr).second;
        ClusterDist = (*clustersMultiMapItr).first;
        cl2 = cluster;
        dist_cl2 = ClusterDist;

        cl3 = NULL;
        dist_cl3 = std::numeric_limits<float>::max();
    }

    if( clustersMultiMap.size() > 2 )
    {
        // Insere os clusters ordenados no vetor:
        std::multimap<float, Cluster*>::iterator clustersMultiMapItr;
        clustersMultiMapItr = clustersMultiMap.begin();
        Cluster* cluster = (*clustersMultiMapItr).second;
        float ClusterDist = (*clustersMultiMapItr).first;
        cl1 = cluster;
        dist_cl1 = ClusterDist;

        ++clustersMultiMapItr;
        cluster = (*clustersMultiMapItr).second;
        ClusterDist = (*clustersMultiMapItr).first;
        cl2 = cluster;
        dist_cl2 = ClusterDist;

        ++clustersMultiMapItr;
        cluster = (*clustersMultiMapItr).second;
        ClusterDist = (*clustersMultiMapItr).first;
        cl3 = cluster;
        dist_cl3 = ClusterDist;
    }

    return;
}


/**
  * Retorna a norna de um vetor qualquer.
  */
float GNG_CPP::lenght( float* vec1 )
{
    int i;
    float temp = 0;
    // Casos de retorno:
    if( vec1 == NULL)
        return -1;
    for( i=0 ; i<_gDim ; i++ )
        temp+= (float)( vec1[i]*vec1[i] );
    temp = (float)sqrt(temp);
    return temp;
}


/**
  * Normaliza o vetor recebido.
  */
float GNG_CPP::normalize( float* vec1 )
{
    int i;
    // Casos de retorno:
    if( vec1 == NULL)
        return -1;

    float l = lenght( vec1 );
    for( i=0 ; i<_gDim ; i++ )
        vec1[i] = vec1[i]/l;
    return l;
}


/**
  * Encontra o produto interno entre dois vetores.
  */
float GNG_CPP::inner( float* vec1, float* vec2 )
{
    int i;
    float temp = 0;
    // Casos de retorno:
    if( (vec1 == NULL) || (vec2 == NULL) )
        return -1;
    for( i=0 ; i<_gDim ; i++ )
        temp+= (float)( vec1[i]*vec2[i] );
    return temp;
}


/**
  * Retorna a distancia Euclideana.
  */
float GNG_CPP::fEuclideanDist( float* vec1, float* vec2, int size )
{
    int i;
    float temp = 0;
    int usedDimension;
    if( (size != -1) && (size <= _gDim) )
        usedDimension = size;
    else
        usedDimension = _gDim;
    // Casos de retorno:
    if( (vec1 == NULL) || (vec2 == NULL) )
        return -1;
    for( i=0 ; i<usedDimension ; i++ )
        temp+= (float)( (vec1[i]-vec2[i])*(vec1[i]-vec2[i]) );
    temp = (float)sqrt(temp);
    return temp;
}


/**
  * Retorna a distancia Euclideana de correlacao cruzada.
  */
//float GNG_CPP::fCrossCorrEuclideanDist( float* vec1, float* vec2, int size )
//float GNG_CPP::fEuclideanDist( float* vec1, float* vec2, int size )
//{
//    int i;
//    float temp = 0;
//    int usedDimension;
//    if( (size != -1) && (size <= _gDim) )
//        usedDimension = size;
//    else
//        usedDimension = _gDim;
//    // Casos de retorno:
//    if( (vec1 == NULL) || (vec2 == NULL) )
//        return -1;
//
//    float r1;
//    float r0;
//    int upDownFlag = 0;
//
//    temp = 0;
//    upDownFlag = -1;
//    if( upDownFlag == -1 )
//    {
//        temp+= (float)( (vec1[0]-vec2[0])*(vec1[0]-vec2[0]) );
//        float v0 = 0;
//        for( i=1 ; i<usedDimension ; i++ )
//        {
//            float v1 = (float)( (vec1[i]-vec2[i-1])*(vec1[i]-vec2[i-1]) );
//            float v0 = (float)( (vec1[i]-vec2[i])*(vec1[i]-vec2[i]) );
//            float result = MIN( v0, v1 );
//            temp += result;
//        }
//    }
//    r1 = temp;
//
//    temp = 0;
//    upDownFlag = 1;
//    if( upDownFlag == 1 )
//    {
//        for( i=0 ; i<usedDimension-1 ; i++ )
//        {
//            float v1 = (float)( (vec1[i]-vec2[i+1])*(vec1[i]-vec2[i+1]) );
//            float v0 = (float)( (vec1[i]-vec2[i])*(vec1[i]-vec2[i]) );
//            float result = MIN( v0, v1 );
//            temp += result;
//        }
//        temp+= (float)( (vec1[usedDimension-1]-vec2[usedDimension-1])*(vec1[usedDimension-1]-vec2[usedDimension-1]) );
//    }
//    r0 = temp;
//
//    temp = MIN( r0, r1 );
//
//    temp = (float)sqrt(temp);
//    return temp;
//}


/**
  * Retorna a correlacao cruzada entre dois vetores.
  */
//fCrossCorrelationDist
float GNG_CPP::fCrossCorrelationDist( float* vec1, float* vec2, int size )
{
    float temp = 0;
    // Casos de retorno:
    if( (vec1 == NULL) || (vec2 == NULL) )
        return -1;

    // Encontrando a norma:
    float n1 = inner( vec1, vec1 );
    float n2 = inner( vec2, vec2 );

    float norm = sqrt( n1 * n2 );

    temp = inner( vec1, vec2 ) / norm;

    // Multiplicando por -1 e somando 1 para garantir que o resultado sera maior que 0:
    temp = (-1.0f * temp) + 1.1f;
    return temp;
}


/**
  * Retorna o quadrado da distancia de dois vetores.
  */
float GNG_CPP::fQuadDist( float* vec1, float* vec2 )
{
    int i;
    float temp = 0;
    // Casos de retorno:
    if( (vec1  == NULL) || (vec2 == NULL) )
        return -1;
    for( i=0 ; i<_gDim ; i++ )
        temp+= (float)( (vec1[i]-vec2[i])*(vec1[i]-vec2[i]) );
    return temp;
}


/**
  * Calcula a distancia das amostras ponderada pelo desvio-padrao.
  * Essa funcao permite encontrar amostras que, apesar de classificadas como pertencentes ao cluster,
  * podem ser consideradas outliers.
  * @return Retorna a distancia ponderada, ou -1 em caso de erro.
  */
float GNG_CPP::fWeightedDist( float* vec1, float* vec2, float* stdDev )
{
    int i;
    float temp = 0;
    float sumStdDev = 0;
    // Casos de retorno:
    if( (vec1  == NULL) || (vec2 == NULL) || (stdDev == NULL) )
        return -1;
    for( i=0 ; i<_gDim ; i++ )
    {
        temp+= (float)( (vec1[i]-vec2[i])*(vec1[i]-vec2[i])*(1/stdDev[i]) );
        sumStdDev += stdDev[i];
    }
    temp = (float)sqrt(temp/sumStdDev);
    return temp;
}



/**
    * Retorna o numero de nos.
    */
int GNG_CPP::getNumClusters()
{
    return _gCenters.size();
}


/**
    * Construtor.
    */
GNG_CPP::GNG_CPP( int gNumClusters, int gDim, int gClustersIdInit )
{
    _gNumClusters = gNumClusters;
    _gDim = gDim;
    _gClustersIdInit = gClustersIdInit;
}


/**
    * Construtor.
    */
GNG_CPP::GNG_CPP( const char* filename )
{
    load( filename );
}


/**
    * Destrutor.
    */
GNG_CPP::~GNG_CPP()
{
    // Reinicializa os vetores:
    for( unsigned int i=0 ; i<_gCenters.size() ; i++ )
        delete _gCenters[i];

    _gCenters.clear();

    for( unsigned int i=0 ; i<_gEdges.size() ; i++ )
        delete _gEdges[i];

    _gEdges.clear();

    for( unsigned int i=0 ; i<_gDistanceByEdges.size() ; i++ )
        vectorFreeMemory( _gDistanceByEdges[i] );

    _gDistanceByEdges.clear();
}


/**
    * Obtem um ponteiro para o vetor de clusters.
    */
std::vector<Cluster*>& GNG_CPP::getClustersVector()
{
    return _gCenters;
}


/**
    * Inicia o treinamento (gera duas unidades em posicoes aleatorias).
    */
int GNG_CPP::createFirstTwoClusters( float* sp1, float* sp2 )
{
    if( (sp1 == NULL ) || (sp2 == NULL ) )
        return -1;

    // Reinicializa os vetores:
    _gCenters.clear();
    _gEdges.clear();

    // Cria dois clusters:
    Cluster* tC1 = new Cluster( _gDim, sp1, 1, 1, -1, NULL, NULL );
    Cluster* tC2 = new Cluster( _gDim, sp2, 1, 1, -1, NULL, NULL );

    // Insere esses dois clusters no vetor:
    _gCenters.push_back( tC1 );
    _gCenters.push_back( tC2 );

    return 0;
}


/**
    * Adiciona um novo cluster.
    */
void GNG_CPP::addNode()
{
    // Pega o no de maior erro acumulado:
    float maxError = std::numeric_limits<float>::min();
    Cluster* greaterErrorCluster = NULL;
    float thisError;
    int numSamples;
    for( size_t i=0 ; i<_gCenters.size() ; i++ )
    {
        (_gCenters[i])->getErrorInf( thisError, numSamples );
        if( maxError < thisError )
        {
            maxError = thisError;
            greaterErrorCluster = _gCenters[i];
        }
    }
    // Pega o vizinho de maior erro acumulado. Tambem armazena a aresta que os liga:
    float maxNeighborError = std::numeric_limits<float>::min();
    std::vector<GNG_Edge*>::iterator maxErrorsClustersEdgeItr;
    Cluster* greaterErrorNeighborCluster = NULL;
    for( size_t i=0 ; i<_gEdges.size() ; i++ )
    {
        if( (_gEdges[i])->isNode( greaterErrorCluster ) )
        {
            Cluster* cluster = (_gEdges[i])->getOtherNode( greaterErrorCluster );
            cluster->getErrorInf( thisError, numSamples );
            if( maxNeighborError < thisError )
            {
                maxNeighborError = thisError;
                greaterErrorNeighborCluster = cluster;
                maxErrorsClustersEdgeItr = (_gEdges.begin()+i);
            }
        }
    }

    // Insere um novo no na posicao media entre greaterErrorCluster e greaterErrorNeighborCluster:
    float* newClusterPos = new float[_gDim];
    int size;
    float* greaterErrorClusterValues = NULL;
    float* greaterErrorNeighborClusterValues = NULL;
    greaterErrorCluster->getPos( size, greaterErrorClusterValues );
    greaterErrorNeighborCluster->getPos( size, greaterErrorNeighborClusterValues );
    for( int i=0 ; i<_gDim ; i++ )
        newClusterPos[i] = (greaterErrorClusterValues[i] + greaterErrorNeighborClusterValues[i])/2.0;

    // Decrescer erros acumulados de greaterErrorCluster e greaterErrorNeighborCluster, multiplicando-os por _gAlpha;
    greaterErrorCluster->setErrorInf( maxError * _gAlpha );
    greaterErrorNeighborCluster->setErrorInf( maxNeighborError * _gAlpha );
    // Criando o novo cluster:
    Cluster* newCluster = new Cluster( _gDim, newClusterPos, maxError * _gAlpha, 0, -1, NULL, NULL );
    _gCenters.push_back( newCluster );

    // Deleta o vetor auxiliar de posicoes do novo vetor:
    delete newClusterPos;

    // Remove a aresta ligando greaterErrorCluster e greaterErrorNeighborCluster:
    _gEdges.erase(maxErrorsClustersEdgeItr);

    // Insere duas novas arestas: newCluster-greaterErrorCluster e newCluster-greaterErrorNeighborCluster:
    GNG_Edge* newEdge = new GNG_Edge( newCluster, greaterErrorCluster, 0 );
    _gEdges.push_back( newEdge );

    GNG_Edge* newNeighborEdge = new GNG_Edge( newCluster, greaterErrorNeighborCluster, 0 );
    _gEdges.push_back( newNeighborEdge );

}


/**
    * Recebe uma amostra e encontra o cluster mais proximo segundo o criterio de distancia euclideana.
    */
int GNG_CPP::getBMU( float* smp, Cluster* &retCluster, Cluster* &retClusterNeighbor,  Cluster* &retClusterSecNeighbor,
                     float& error, float& errorNeighbor, float& errorSecNeighbor )
{
    if( smp == NULL )
        return -1;

    // Distancia ao primeiro cluster:
    float* clusterPos = NULL;
    int size;
    Cluster* cluster = _gCenters[0];
    cluster->getPos( size, clusterPos );
    float thisDist = fEuclideanDist( smp, clusterPos );

    // Objeto que recebe os clusters passo a passo, armazenando ponteiros para os mais 2 proximos:
    BMU_SegBMU gBMU_SegBMU( cluster, thisDist, NULL );

    for( size_t i=1 ; i<_gCenters.size() ; i++ )
    {
        Cluster* thiscluster = _gCenters[i];
        thiscluster->getPos( size, clusterPos );
        float thisDist = fEuclideanDist( smp, clusterPos );
        gBMU_SegBMU.UpDate( thiscluster, thisDist );
    }

    // Retorna os tres mais proximos:
    retCluster = gBMU_SegBMU.cl1;
    retClusterNeighbor = gBMU_SegBMU.cl2;
    retClusterSecNeighbor = gBMU_SegBMU.cl3;
    error = gBMU_SegBMU.dist_cl1;
    errorNeighbor = gBMU_SegBMU.dist_cl2;
    errorSecNeighbor = gBMU_SegBMU.dist_cl3;

    return 0;
}


/**
    * Recebe uma amostra e encontra o cluster mais proximo segundo o criterio de distancia euclideana.
    * Permite modificar a dimensao das amostras durante a procura.
    */
int GNG_CPP::getBMU( float* smp, Cluster* &retCluster, float& error, int usedSize )
{
    if( smp == NULL )
        return -1;

    if( usedSize > _gDim )
        return -1;

    // Distancia ao primeiro cluster:
    float* clusterPos = NULL;
    int size;
    Cluster* cluster = _gCenters[0];
    cluster->getPos( size, clusterPos );
    float thisDist = fEuclideanDist( smp, clusterPos, usedSize );

    for( size_t i=1 ; i<_gCenters.size() ; i++ )
    {
        Cluster* thiscluster = _gCenters[i];
        thiscluster->getPos( size, clusterPos );
        float thisDistInt = fEuclideanDist( smp, clusterPos, usedSize );
        if( thisDistInt < thisDist )
        {
            thisDist = thisDistInt;
            cluster = thiscluster;
        }
    }
    retCluster = cluster;
    error = thisDist;
    return 0;
}


/**
    * Obtem a distancia entre dois clusters como o numero de distancias medias entre o cluster referencia (o primeiro) e o segundo cluster.
    * Tais clusters nao precisam necessariamente pertencer �  instancia.
    */
float GNG_CPP::fNumMeansDistance( int cluster1Pos, int cluster2Pos )
{
    Cluster* c1 = _gCenters[cluster1Pos];
    float* c1Values=NULL;
    int c1size;
    c1->getPos( c1size, c1Values );

    Cluster* c2 = _gCenters[cluster2Pos];
    float* c2Values=NULL;
    int c2size;
    c2->getPos( c2size, c2Values );

    float c1RelativeDistance;
    c1->getRelativeDistance(c1RelativeDistance);
    return (fEuclideanDist( c1Values, c2Values ) / c1RelativeDistance);
}


/**
    * Obtem a distancia entre dois clusters como o numero de distancias medias entre o cluster referencia (o primeiro) e o segundo cluster.
    */
float GNG_CPP::fNumMeansDistance( Cluster* c1, Cluster* c2 )
{
    float* c1Values=NULL;
    int c1size;
    c1->getPos( c1size, c1Values );

    float* c2Values=NULL;
    int c2size;
    c2->getPos( c2size, c2Values );

    float c1RelativeDistance;
    c1->getRelativeDistance(c1RelativeDistance);
    return (fEuclideanDist( c1Values, c2Values ) / c1RelativeDistance);
}


/**
    * Ordena o vetor de nos pela distancia media do cluster aos seus vizinhos topologicos. Insere o Id do cluster a partir da sua nova posicao no vetor.
    */
void GNG_CPP::orderingByNeighborsDist()
{
    // Para cada cluster, encontra todos os seus vizinhos topologicos:
    for( unsigned int i=0 ; i<_gCenters.size() ; i++ )
    {
        float meanDistance = 0;
        int numNeighbors = 0;
        Cluster* cluster = _gCenters[i];
        float* clusterValues=NULL;
        int size;
        cluster->getPos( size, clusterValues );

        for( unsigned int j=0 ; j<_gEdges.size() ; j++ )
        {
            if( (_gEdges[j])->isNode( cluster ) )  //< Caso cluster seja um dos vertices da aresta:
            {
                numNeighbors++;     //< Incrementa o numero de vizinhos.
                Cluster* neighbor = (_gEdges[j])->getOtherNode( cluster );       //< Pega o outro no da aresta.
                float* neighborValues=NULL;
                neighbor->getPos( size, neighborValues );
                meanDistance += fEuclideanDist( clusterValues, neighborValues );
            }
        }

        // Insere a distancia media como distancia relativa:
        cluster->setRelativeDistance( meanDistance / numNeighbors );
    }

    std::multimap<float, Cluster*> clustersMultiMap;
    // Insere os clusters no multimap ordenados por erro:
    for( unsigned int i=0 ; i<_gCenters.size() ; i++ )
    {
        Cluster* cluster = _gCenters[i];
        float relDist;
        cluster->getRelativeDistance( relDist );
        clustersMultiMap.insert( std::pair<float, Cluster*>( relDist, cluster ) );
    }
    // Insere os clusters ordenados no vetor:
    std::multimap<float, Cluster*>::iterator clustersMultiMapItr;
    clustersMultiMapItr = clustersMultiMap.begin();
    int cont=0;
    for( ; clustersMultiMapItr != clustersMultiMap.end() ; ++clustersMultiMapItr, cont++ )
    {
        Cluster* cluster = (*clustersMultiMapItr).second;
        cluster->setClusterId( cont + _gClustersIdInit );
        // Insere esse cluster no vetor de clusters:
        _gCenters[cont] = cluster;
    }
    // Limpa o Multimap:
    clustersMultiMap.clear();
}


/**
    * Calcula a distancia entre os clusters atraves do numero de arestas percorridas entre um cluster e outro.
    */
int GNG_CPP::findDistancesByEdges()
{
    // Esvaziando a matriz que mantem as distancias entre as arestras:
    _gDistanceByEdges.clear();
    // Alocando a memoria necessaria para essa mesma matriz:
    int dimMatrix = _gCenters.size();
    for( int i=0 ; i<dimMatrix ; i++ )
    {
        std::vector<int> thisLine;
        thisLine.reserve(dimMatrix);
        thisLine.resize(dimMatrix);
        _gDistanceByEdges.push_back(thisLine);
    }

    // Criando uma instancia da classe que permite encontrar as distancias por arestas:
    Graph* tmpGraph = new Graph( _gEdgesMatrix );
    for( unsigned int i=0 ; i<_gCenters.size() ; i++ )
    {
        int ret = tmpGraph->BFS( i, _gDistanceByEdges[i] );
        if( ret == -1 )
            return -1;
    }

    delete tmpGraph;
    return 0;
}


/**
    * Cria um vetor descrevendo as arestas de ligacao entre os clusters, baseado no id numerico dos clusters.
    */
void GNG_CPP::createNumericEdgesMatrix( std::vector<std::vector<int> >& edgesMatrix )
{
    // Esvaziando a matriz:
    edgesMatrix.clear();
    // Faz o mesmo na copia interna da classe:
    _gEdgesMatrix.clear();

    int dimMatrix = _gCenters.size();
    for( int i=0 ; i<dimMatrix ; i++ )
    {
        std::vector<int> thisLine;
        thisLine.reserve(dimMatrix);
        thisLine.resize(dimMatrix);
        edgesMatrix.push_back(thisLine);

        // O mesmo na copia interna:
        std::vector<int> ithisLine;
        ithisLine.reserve(dimMatrix);
        ithisLine.resize(dimMatrix);
        _gEdgesMatrix.push_back(ithisLine);

        for( int j=0 ; j<dimMatrix ; j++ )
        {
            thisLine[j] = 0;
            ithisLine[j] = 0;
        }
    }

    for( unsigned int j=0 ; j<_gEdges.size() ; j++ )
    {
        Cluster* c1;
        Cluster* c2;
        (_gEdges[j])->getClusters( c1, c2 );

        int c1Id, c2Id;
        c1->getClusterId( c1Id );
        c2->getClusterId( c2Id );

        // Diminuindo do valor do clusterId inicial:
        c1Id = c1Id - _gClustersIdInit;
        c2Id = c2Id - _gClustersIdInit;
        edgesMatrix[c1Id][c2Id] = 1;
        edgesMatrix[c2Id][c1Id] = 1;
        // O mesmo na copia interna:
        _gEdgesMatrix[c1Id][c2Id] = 1;
        _gEdgesMatrix[c2Id][c1Id] = 1;
    }
}


/**
    * Recria o vetor de arestas a partir da matriz booleana.
    */
int GNG_CPP::createEdgesVectorFromMatrix()
{
    int dimMatrix = _gEdgesMatrix.size();
    for( int i=0 ; i<dimMatrix ; i++ )
    {
        for( int j=0 ; j<i ; j++ )
        {
            if( _gEdgesMatrix[i][j] == 1 )
            {
                Cluster* c1 = _gCenters[i];
                Cluster* c2 = _gCenters[j];

                GNG_Edge* newEdge = new GNG_Edge( c1, c2, 0 );
                _gEdges.push_back( newEdge );
            }
        }
    }
    return 0;
}


/**
    * Funcao auxiliar no processo de treinamento.
    * Utilizada tanto pela funcao de treinamento que recebe um vetor de amostras, quanto pela que recebe
    * uma instancia de VolumeSamples.
    */
void GNG_CPP::auxiliaryTrainStep( float* smp, bool& trained, unsigned int& numTotalTrains )
{
    // Encontra os dois clusters mais proximos da amostra:
    Cluster* s1 = NULL;
    Cluster* s2 = NULL;
    Cluster* s3 = NULL;
    float s1error, s2error, s3error;
    getBMU( smp, s1, s2, s3, s1error, s2error, s3error );

    // Incrementa de 1 as idades de todas as arestras que saem de s1:
    for( unsigned int i=0 ; i<_gEdges.size() ; i++ )
    {
        (_gEdges[i])->incAge( s1 );
    }

    // Adiciona o erro ao somatorio de erros do cluster:
//#define UNITARI_ERROR
#ifdef UNITARI_ERROR
    s1->incErrorInf( 1 );
#endif

//#define PERCENTAGE_ERROR
#ifdef PERCENTAGE_ERROR
    float error;
    int nSamples;
    s1->getErrorInf( error, nSamples );
    float scale = error/nSamples;
    s1->incErrorInf( s1error/scale );
#else
    s1->incErrorInf( s1error );
#endif

    // Move s1 com gEb:
    float* s1Values=NULL;
    int size;
    s1->getPos( size, s1Values );
    for( int i=0 ; i<_gDim ; i++ )
        s1Values[i] = s1Values[i] + (_gEb * (smp[i] - s1Values[i]) );

    // E move seus vizinhos topologicos de acordo com gEn. Alem disso, armazena a aresta que liga s1 e s2 caso exista:
    GNG_Edge* s1_s2_edge = NULL;
    for( unsigned int j=0 ; j<_gEdges.size() ; j++ )
    {
        if( (_gEdges[j])->isNode( s1 ) )  //< Caso s1 seja um dos vertices da aresta:
        {
            Cluster* neighbor = (_gEdges[j])->getOtherNode( s1 );       //< Pega o outro no da aresta.
            if( neighbor != NULL )
            {
                // Move neighbor com gEn:
                float*neighborValues=NULL;
                neighbor->getPos( size, neighborValues );
                for( int i=0 ; i<_gDim ; i++ )
                    neighborValues[i] = neighborValues[i] + (_gEn * (smp[i] - neighborValues[i]) );
                if( neighbor == s2 )    //< Caso neighbor seja s2, armazena ponteiro para essa aresta:
                    s1_s2_edge = _gEdges[j];
            }
        }
    }

    // Caso s1_s2_edge exista, fazer s1_s2_edge->age = 0. Caso nao exista, deve ser criada:
    if( s1_s2_edge != NULL )
        s1_s2_edge->ResetAge();
    else        //< Caso a aresta nao exista, ela devera ser criada:
    {
        GNG_Edge* newEdge = new GNG_Edge( s1, s2, 0 );
        _gEdges.push_back( newEdge );
    }

    // Remover todas as arestras com age > _gAmax:
    std::vector<int> edgesDeleted;
    std::vector<GNG_Edge*>::iterator gEdgesItr;
    gEdgesItr = _gEdges.begin();
    int cont=0;
    for( gEdgesItr = _gEdges.begin() ; gEdgesItr != _gEdges.end() ; ++gEdgesItr, cont++ )
    {
        if( (*gEdgesItr)->getAge() > _gAmax )   //< Caso age seja maior que _gAmax, deleta a aresta:
        {
            edgesDeleted.push_back(cont);
        }
    }

    // Retirando todas as arestas encontradas:
    for( int c=edgesDeleted.size()-1 ; c>=0 ; c-- )
    {
        std::vector<GNG_Edge*>::iterator gEdgesItr  = _gEdges.begin()+edgesDeleted[c];
        std::vector<GNG_Edge*>::iterator gEdgesItr2 = _gEdges.begin()+edgesDeleted[c-1];
        std::vector<GNG_Edge*>::iterator gEdgesItr3 = _gEdges.begin()+edgesDeleted[c+1];
        _gEdges.erase( _gEdges.begin()+edgesDeleted[c] );
    }

    // Remove todos os clusters que nao possuem arestas:
    std::vector<int> clustersDeleted;
    for( unsigned int i=0 ; i<_gCenters.size() ; i++ )
    {
        Cluster* thisCluster = _gCenters[i];
        bool edge = false;
        for( unsigned int j=0 ; j<_gEdges.size() ; j++ )
        {
            if( (_gEdges[j])->isNode( thisCluster ) )
            {
                edge = true;
                break;
            }
        }
        if( edge == false )   //< Caso o cluster nao possua arestas, e incluso no vetor de clusters a serem deletados.
            clustersDeleted.push_back( i );
    }

    // Retirando todos os clusters encontrados:
    for( int c=clustersDeleted.size()-1 ; c>=0 ; c-- )
    {
        _gCenters.erase( _gCenters.begin()+clustersDeleted[c] );
    }

    // Caso _gLambda treinamentos ja tenha sido executados, inclui novo no:
    if( (numTotalTrains > 0) && ((numTotalTrains % _gLambda) == 0)  && ((int)_gCenters.size() < _gNumClusters) )
    {
        addNode();
        std::cout << "Novo no adicionado. " << "Total: " << _gCenters.size() << " Treinos Executados: " << numTotalTrains << std::endl;
    }

    // Decrescer o erro de todos os nos multiplicando-os  por _gD;
    for( unsigned int j=0 ; j<_gCenters.size() ; j++ )
    {
        Cluster* thisCluster = _gCenters[j];
        float error;
        int nSamples;
        thisCluster->getErrorInf( error, nSamples );
        thisCluster->setErrorInf( error * _gD );
    }

    // Incrementa o contador de treinamentos:
    numTotalTrains++;

    // Ja existe numero desejado de clusters e _gLambda treinamentos ja foram executados depois da ultima inclusao:
    if( ((int)_gCenters.size() == _gNumClusters) && ((numTotalTrains % _gLambda) == 0 ) )
    {
        trained = true;
    }
}


/**
  * O treinamento pode ser feito passo a passo caso o vetor inteiro, contendo todas as amostras de treino, nao esteja disponível.
  * Isso podera ser feito atraves da utilizacao de duas funcoes: a primeira (initTrain) inicializa as variaveis de treinamento, devendo
  * receber duas amostras iniciais. A segunda sera a funcao "auxiliaryTrainStep", que devera ser chamada a cada amostra a ser treinada,
  * iterando ate que retorne seu parametro "bool& trained" como true. Isso indica que o treinamento tera sido finalizado.
  */
int GNG_CPP::initTrain( int gLambda, float gEb, float gEn, float gAlpha, int gAmax, float gD,
                    int& numExecutedTrains, float* sp1,  float* sp2, int nneurons )
{
    /* Gerando uma semente nova, para os nmeros aleatrios: */
    srand( (unsigned)time( NULL ) );

    // Caso nao existam amostras, retorna erro:
    if( (sp1 == NULL) || (sp2 == NULL) )
        return -1;

    // Permite reconfigurar o numero de neuronios que deverao ser formados:
    if( (nneurons != -1) && (nneurons > _gNumClusters) )
        _gNumClusters = nneurons;

    // Insere os parametros na estrutura:
    _gLambda = gLambda;
    _gEb = gEb;
    _gEn = gEn;
    _gAlpha = gAlpha;
    _gAmax = gAmax;
    _gD = gD;


    // Caso o numero de neuronios desejado seja somente 1 retorna erro:
    if( (nneurons == 1) )
        return -1;

    // Caso o numero de neuronios desejado seja maior que 1 e ainda nao tenha sido executado nenhum treinamento, inicia variaveis:
    if( _gCenters.size() < 1 )
    {
        // Cria os dois primeiros clusters:
        createFirstTwoClusters( sp1, sp2 );
    }

    return 0;
}


/**
    * Executa o treinamento recebendo um vetor de treinamento que contem todas as amostras que deverao ser treinadas.
    */
int GNG_CPP::allSamplesTrain( std::vector<float*> samples, int gLambda, float gEb, float gEn, float gAlpha, int gAmax, float gD,
                    int& numExecutedTrains, int nneurons )
{
    if( samples.size() == 0 )
        return -1;

    /* Gerando uma semente nova, para os nmeros aleatorios: */
    srand( (unsigned)time( NULL ) );

    float* sp1 = samples[0];
    float* sp2 = samples[samples.size()-1];
    // Inicializando todas as variaveis de treinamento:
    initTrain( gLambda, gEb, gEn, gAlpha, gAmax, gD, numExecutedTrains, sp1, sp2, nneurons );

    int numSamples = samples.size();
    unsigned int numTotalTrains = 0;
    bool trained = false;
    while( !trained )
    {
        // Escolhe uma entrada de acordo com a distribuiçăo de probabilidades:
        float prob = ((float)((float)rand()/(float)RAND_MAX));
        int ind = (int)(prob*(numSamples-1));
        float* smp = samples[ind];

        // Funcao que executa um passo do treinamento:
        auxiliaryTrainStep( smp, trained, numTotalTrains );
    }
    numExecutedTrains = numTotalTrains;

    return 0;
}


/**
    * Executa o treinamento.
    */
int GNG_CPP::allSamplesTrain( Volume_Smp* samples, int gLambda, float gEb, float gEn, float gAlpha, int gAmax, float gD,
                    int& numExecutedTrains, int nneurons, Volume_Smp::SamplesType smpType )
{
    /* Gerando uma semente nova, para os nmeros aleatorios: */
    srand( (unsigned)time( NULL ) );

    // Permite reconfigurar o numero de neuronios que deverao ser formados:
    if( (nneurons != -1) && (nneurons > _gNumClusters) )
        _gNumClusters = nneurons;

    // Insere os parametros na estrutura:
    _gLambda = gLambda;
    _gEb = gEb;
    _gEn = gEn;
    _gAlpha = gAlpha;
    _gAmax = gAmax;
    _gD = gD;


    // Caso o numero de neuronios desejado seja somente 1, retorna erro
    if( (nneurons == 1) )
        return -1;

    // Caso ainda nao tenha sido executado nenhum treinamento, inicia variaveis:
    if( _gCenters.size() < 1 )
    {
        // Com isso, podemos obter a amostra:
        float* sp1 = samples->getRandomSample( -1, -1, -1, -1, 0.01f, smpType );

        // Copia os valores de sp1 em sp2 (nao existe persistencia nesses dados retornados):
        float* sp2 = new float[samples->_vSmp_Size];
        for( int i=0 ; i<samples->_vSmp_Size ; i++ )
            sp2[i] = sp1[i];
        // Atualiza o valor de sp1:
        sp1 = samples->getRandomSample( -1, -1, -1, -1, 0.01f, smpType );

        // Cria os dois primeiros clusters:
        createFirstTwoClusters( sp1, sp2 );
        // Deleta o vetor auxiliar:
        delete[] sp2;
    }

    unsigned int numTotalTrains = 0;
    bool trained = false;
    while( !trained )
    {
        float* smp = samples->getRandomSample( -1, -1, -1, -1, 0.01f, smpType );
        if( smp == NULL )
            continue;

        // Funcao que executa um passo do treinamento:
        auxiliaryTrainStep( smp, trained, numTotalTrains );
    }
    numExecutedTrains = numTotalTrains;

    return 0;
}


/**
    * Termina a etapa de treinamento, calculando informacoes importantes mas que nao fazem parte do algoritmo
    * padrao de treinamento do GNG. Cria a matriz numerica de arestas, ordena o vetor de vertices (), classifica
    * o vetor de amostras recebido nos clusters formados, e com isso calcula estatísticas importantes, tais como
    * media, variancia, matriz de covariancias entre os componentes do cluster, entre outras informacoes.
    * Caso o vetor de amostras seja vazio, somente ordena os nos e cria as matrizes de arestas e
    * de distancia por arestas (sem calcular as estatísticas).
    * @param samples Vetor contendo todas as amostras que serao utilizadas nos calculos.
    * @param edgesMatrix Matriz de arestas numericas entre os clusters, a ser retornada.
    * @return Retorna -1 em caso de erro, ou 0 tenha somente ordenado os nos e criado as matrizes de arestas e
    * de distancia por arestas (sem calcular as estatísticas), ou 1 caso tenha tambem calculado medias, varianciaes e
    * matrizes de covariancia dos clusters utilizando as amostras.
    */
int GNG_CPP::finalizeTrain( std::vector<float*> samples, std::vector<std::vector<int> >& edgesMatrix,
                            std::vector<std::vector<float*> >& classifiedSamplesVector,
                            bool retClassifiedSamples )
{
    // E preciso que tenhamos pelo menos 2 neuronios no grafo:
    if( _gCenters.size() < 1 )
        return -1;

    // Ordena os nos pela distancia media do no aos seus vizinhos topologicos:
    orderingByNeighborsDist();
    // Cria a matriz de arestas, baseada no numero identificador do cluster dentro do vetor de clusters ja ordenados por erro:
    createNumericEdgesMatrix( edgesMatrix );

    // Constroi a matriz de distancias por arestas:
    findDistancesByEdges();

    // Caso existam menos amostras do que clusters, retorna sem calcular as estatísticas:
    if( samples.size() < _gCenters.size() )
        return 0;

    // O ultimo passo consiste em calcular medias e desvios-padrao dos clusters em relacao � s suas amostras:
    int _gCenterssize = _gCenters.size();
    //std::vector<std::vector<float*> > eachClusterSamplesVector;
    classifiedSamplesVector.reserve( _gCenterssize );
    classifiedSamplesVector.resize( _gCenterssize );

    // Classifica cada amostra:
    unsigned int i;
    // std::vector<float*> samples
    for( i = 0 ; i < samples.size() ; i++ )
    {
        // Encontra os dois clusters mais proximos da amostra:
        Cluster* s1 = NULL;
        Cluster* s2 = NULL;
        Cluster* s3 = NULL;
        float s1error, s2error, s3error;
        getBMU( samples[i], s1, s2,  s3, s1error, s2error, s3error );
        int clusterIndex;
        s1->getClusterId( clusterIndex );
        clusterIndex = clusterIndex  - _gClustersIdInit;
        classifiedSamplesVector[clusterIndex].push_back( samples[i] );
    }

    // Now it is possible to find the means and variances:
    for( i = 0; i < _gCenters.size() ; i++ )
    {
        Cluster* thisCluster = _gCenters[i];
        int clusterId;
        thisCluster->getClusterId( clusterId );
        clusterId = clusterId - _gClustersIdInit;
        getStatistics( classifiedSamplesVector[clusterId], _gDim, thisCluster->_clMean, thisCluster->_clVar );
        // And it is possible to get the cluster's covariance matrix:
        //      getCovarianceMatrix( classifiedSamplesVector[clusterId], _gDim, thisCluster->_clCenter,
        //                           thisCluster->_clCovarMatrix, thisCluster->_clCovarMatrixInv );

//        getCovarianceMatrix( classifiedSamplesVector[clusterId], _gDim,
//                             thisCluster->_clCovarMatrix, thisCluster->_clCovarMatrixInv );
        //std::cout << i << ". " << eachClusterSamplesVector[i].size() << std::endl;
    }

    return 1;
}


/**
    * Termina a etapa de treinamento, calculando informacoes importantes mas que nao fazem parte do algoritmo
    * padrao de treinamento do GNG. Cria a matriz numerica de arestas, ordena o vetor de vertices (), classifica
    * o vetor de amostras recebido nos clusters formados, e com isso calcula estatísticas importantes, tais como
    * media, variancia, matriz de covariancias entre os componentes do cluster, entre outras informacoes.
    * Caso o vetor de amostras seja vazio, somente ordena os nos e cria as matrizes de arestas e
    * de distancia por arestas (sem calcular as estatísticas).
    * @param samples Vetor contendo todas as amostras que serao utilizadas nos calculos.
    * @param edgesMatrix Matriz de arestas numericas entre os clusters, a ser retornada.
    * @return Retorna -1 em caso de erro, ou 0 tenha somente ordenado os nos e criado as matrizes de arestas e
    * de distancia por arestas (sem calcular as estatísticas), ou 1 caso tenha tambem calculado medias, varianciaes e
    * matrizes de covariancia dos clusters utilizando as amostras.
    */
int GNG_CPP::finalizeTrain( Volume_Smp* samples, int numSamplesUsed, std::vector<std::vector<int> >& edgesMatrix,
                            std::vector<std::vector<float*> >& classifiedSamplesVector,
                            bool retClassifiedSamples )
{
    // E preciso que tenhamos pelo menos 2 neuronios no grafo:
    if( _gCenters.size() < 1 )
        return -1;

    // Ordena os nos pela distancia media do no aos seus vizinhos topologicos:
    orderingByNeighborsDist();
    // Cria a matriz de arestas, baseada no numero identificador do cluster dentro do vetor de clusters ja ordenados por erro:
    createNumericEdgesMatrix( edgesMatrix );

    // Constroi a matriz de distancias por arestas:
    findDistancesByEdges();

    // O ultimo passo consiste em calcular medias e desvios-padrao dos clusters em relacao � s suas amostras:
    int _gCenterssize = _gCenters.size();

    std::vector<float> meanErrorsVector;
    meanErrorsVector.assign( _gCenterssize, 0.0f );

    classifiedSamplesVector.reserve( _gCenterssize );
    classifiedSamplesVector.resize( _gCenterssize );

    // Classifica cada amostra:
    int i;
    for( i = 0 ; i < numSamplesUsed ; i++ )
    {
        // Encontra os dois clusters mais proximos da amostra:
        Cluster* s1 = NULL;
        Cluster* s2 = NULL;
        Cluster* s3 = NULL;
        float s1error, s2error, s3error;

        // Escolhe uma entrada de acordo com a distribuicao de probabilidades:
        float* smp = samples->getRandomSample();

        getBMU( smp, s1, s2,  s3, s1error, s2error, s3error );
        int clusterIndex;
        s1->getClusterId( clusterIndex );
        clusterIndex = clusterIndex  - _gClustersIdInit;
        classifiedSamplesVector[clusterIndex].push_back( smp );
        meanErrorsVector[clusterIndex] += s1error;
    }

    // Now it is possible to find the means and variances:
    for( unsigned int i = 0; i < _gCenters.size() ; i++ )
    {
        Cluster* thisCluster = _gCenters[i];
        int clusterId;
        thisCluster->getClusterId( clusterId );
        clusterId = clusterId - _gClustersIdInit;
        thisCluster->setErrorInf( meanErrorsVector[clusterId] / ((classifiedSamplesVector[clusterId]).size() ) );
        getStatistics( classifiedSamplesVector[clusterId], _gDim, thisCluster->_clMean, thisCluster->_clVar );
    }

    return 1;
}


/**
    * Funcao auxiliar. Converte de clusterId para o índice do cluster num vetor.
    */
int GNG_CPP::getClusterPosFromId( int clusterId )
{
    return clusterId - _gClustersIdInit;
}


/**
    * Funcao auxiliar. Converte de clusterId para o índice do cluster num vetor.
    */
Cluster* GNG_CPP::getClusterFromClusterId( int clusterId )
{
    int vectorPos = clusterId - _gClustersIdInit;
    if( (vectorPos >= (int)_gCenters.size()) || (vectorPos < 0) )
        return NULL;
    Cluster* thisCluster = _gCenters[vectorPos];
    return thisCluster;
}


/**
    * Funcao auxiliar. Recebe um vetor contendo um conjunto de amostras classificadas em cada cluster (pelo clusterId)
    * e encontra, com base nesse vetor, os valores de medias e variancias.
    */
int GNG_CPP::getStatisticsFromVector( std::vector<std::vector<float*> >& eachClusterSamplesVector )
{
    // Caso o vetor nao possua o numero de clusters:
    if( eachClusterSamplesVector.size() != _gCenters.size() )
        return -1;
    // Caso vetor de amostras ou de clusters seja vazio:
    if( (eachClusterSamplesVector.size() == 0) || (_gCenters.size() == 0) )
        return -1;

    // Caso contrario, encontra medias e variancias:
    for( int i = 0; i < (int)_gCenters.size() ; i++ )
    {
        Cluster* thisCluster = _gCenters[i];
        int clusterId;
        thisCluster->getClusterId( clusterId );
        clusterId = clusterId - _gClustersIdInit;
        if( clusterId != i )
            return -1;

        getStatistics( eachClusterSamplesVector[clusterId], _gDim, thisCluster->_clMean, thisCluster->_clVar );
        // Calcula tambem a matriz de covariancia para o cluster:
        getCovarianceMatrix( eachClusterSamplesVector[clusterId], _gDim, thisCluster->_clCenter,
                             thisCluster->_clCovarMatrix, thisCluster->_clCovarMatrixInv );
    }

    // AQUI É O MOMENTO DE REMOVER OS OUTLIERS!!!!

    // Nesse momento, o treinamento ja foi executado, e as amostras ja foram classificadas em seus respectivos cluters.
    // Essa e a parte que permite retirar os outliers. Isso e feito para cada cluster separadamente, evitando utilizacao
    // indevida de memoria:
//    removeOutLiers( thisGNG_InfNode->eachClusterSamplesVector, spuGng_Cpp, samples );


    return 0;
}


/**
    * Salva uma instancia de GNG depois de treinada.
    */
int GNG_CPP::save( const char *filename )
{
    if( filename == NULL )
        return -1;

    std::string s;
    std::stringstream out;
    out << filename;
    out << ".gng";
    s = out.str( );

    FILE *fp;
    fp = fopen( s.c_str( ), "wt" );
    if( fp == NULL )
        return -1;

    if( fprintf( fp, "GROWING_NEURAL_GAS_DATA_1.0\n" ) < 0 )
        return -1;

    // Insere os parametros no arquivo:
    if( fprintf( fp, "_gClustersIdInit = %d\n", _gClustersIdInit ) < 0 )
        return -1;
    if( fprintf( fp, "_gLambda = %d\n", _gLambda ) < 0 )
        return -1;
    if( fprintf( fp, "_gEb = %f\n", _gEb ) < 0 )
        return -1;
    if( fprintf( fp, "_gEn = %f\n", _gEn ) < 0 )
        return -1;
    if( fprintf( fp, "_gAlpha = %f\n", _gAlpha ) < 0 )
        return -1;
    if( fprintf( fp, "_gAmax = %d\n", _gAmax ) < 0 )
        return -1;
    if( fprintf( fp, "_gD = %f\n", _gD ) < 0 )
        return -1;
    if( fprintf( fp, "_gCentersSize = %d\n", (int)_gCenters.size() ) < 0 )
        return -1;

    // Agora serao salvas as arestras (atraves da matriz de arestas criada por createNumericEdgesMatrix() despois do treinamento):
    int dimMatrix = _gCenters.size();
    for( int i=0 ; i<dimMatrix ; i++ )
    {
        for( int j=0 ; j<dimMatrix ; j++ )
            if( fprintf( fp, "%d\t", (_gEdgesMatrix[i][j]) ) < 0 )
                return -1;
        fprintf( fp, "\n" );
    }



    // Fecha o arquivo que contem esses dados. Os clusters serao salvos em arquivo separado:
    fclose( fp );

    // Salvando o arquivo de clusters. A extensao dos arquivos de clusters sera ".clt":
    std::string s3;
    std::stringstream out3;
    out3 << filename;
    out3 << ".clt";
    s3 = out3.str( );

    // Abrindo o arquivo para escrita:
    fp = fopen( s3.c_str( ), "wt" );
    if( fp == NULL )
        return -1;

    int num_clusters = _gCenters.size();
    fprintf( fp, "GROWING_NEURAL_GAS_DATA_CLUSTERS_1.0\n" );
    if( fprintf( fp, "NUM_CLUSTERS = %d\n", num_clusters ) < 0 )
        return -1;

    // Inserindo os clusters no arquivo:
    for( unsigned int i=0 ; i<_gCenters.size() ; i++ )
    {
        Cluster* cluster = _gCenters[i];
        if( cluster->saveCluster( fp ) == -1 )
            return -1;
    }

    // Fecha o arquivo de dados do gng:
    fclose( fp );
    return 0;
}


/**
    * Carrega uma instancia previamente treinada a partir de um arquivo.
    */
int GNG_CPP::load( const char *filename )
{
    if( filename == NULL )
        return -1;

    // Lendo o arquivo dos demais dados:
    std::string s;
    std::stringstream out;
    out << filename;
    out << ".gng";
    s = out.str( );

    FILE *fp2=NULL;
    fp2 = fopen( s.c_str( ), "rt" );
    if( fp2 == NULL )
        return -1;

    if( fscanf( fp2, "GROWING_NEURAL_GAS_DATA_1.0\n" )  != 0)
        return -1;

    // Recupera os parametros do arquivo:
    if( fscanf( fp2, "_gClustersIdInit = %d\n", &_gClustersIdInit ) != 1 )
        return -1;
    if( fscanf( fp2, "_gLambda = %d\n", &_gLambda ) != 1 )
        return -1;
    if( fscanf( fp2, "_gEb = %f\n", &_gEb ) != 1 )
        return -1;
    if( fscanf( fp2, "_gEn = %f\n", &_gEn ) != 1 )
        return -1;
    if( fscanf( fp2, "_gAlpha = %f\n", &_gAlpha ) != 1 )
        return -1;
    if( fscanf( fp2, "_gAmax = %d\n", &_gAmax ) != 1 )
        return -1;
    if( fscanf( fp2, "_gD = %f\n", &_gD ) != 1 )
        return -1;

    int dimMatrix = 0;
    if( fscanf( fp2, "_gCentersSize = %d\n", &dimMatrix ) != 1 )
        return -1;

    // Agora serao lidas as arestras (matriz de arestas):
    // Faz o mesmo na copia interna da classe:

    _gEdgesMatrix = std::vector<std::vector<int> > (dimMatrix,std::vector<int>(dimMatrix,0));

    for( int i=0 ; i<dimMatrix ; i++ )
    {
        for( int j=0 ; j<dimMatrix ; j++ )
        {
            int thisValue = -1;
            if( fscanf( fp2, "%d\t", &thisValue ) != 1 ) return -1;
            if( thisValue == 0 )
                _gEdgesMatrix[i][j] = 0;
            if( thisValue == 1 )
                _gEdgesMatrix[i][j] = 1;
        }
        if( fscanf( fp2, "\n" )  != 0) return -1;
    }
    if( fscanf( fp2, "\n" )  != 0) return -1;

    // Fecha o arquivo de dados do gng:
    fclose( fp2 );

    // Lendo o arquivo de clusters. A extensao dos arquivos de clusters sera ".clt":
    std::string s3;
    std::stringstream out3;
    out3 << filename;
    out3 << ".clt";
    s3 = out3.str( );

    // Abrindo o arquivo para escrita:

    fp2 = fopen( s3.c_str( ), "rt" );
    if( fp2 == NULL )
        return -1;

    int num_clusters = -1;
    if( fscanf( fp2, "GROWING_NEURAL_GAS_DATA_CLUSTERS_1.0\n" )  != 0)
        return -1;
    if( fscanf( fp2, "NUM_CLUSTERS = %d\n", &num_clusters ) != 1 )
        return -1;

    // Armazena o numero de clusters:
    _gNumClusters = num_clusters;

    // Alocando o espaco necessario ao vetor de clusters:
    _gCenters.reserve( num_clusters );
    _gCenters.resize( num_clusters );

    // Recriando o mapa contendo os clusters ordenados por erro:
    int i=0;
    for( ; i<num_clusters ; i++ )
    {
        Cluster* cluster = NULL;
        cluster = new Cluster( fp2 );
        if( cluster == NULL )
            return -1;
        int clusterId;
        cluster->getClusterId( clusterId );
        clusterId = clusterId - _gClustersIdInit;
        _gCenters[clusterId] = cluster;
    }

    // Armazenando _gDim:
    Cluster* cluster0 = _gCenters[0];
    cluster0->getClusterSamplesSize( _gDim );

    // Fecha o arquivo de clusters:
    fclose( fp2 );

    // Recriando o vetor de arestas a partir da matrix booleana:
    if( createEdgesVectorFromMatrix() == -1 )
        return -1;

    // Constroi a matriz de distancias por arestas:
    findDistancesByEdges();

    return 0;
}


/**
    * Retorna a dimensao espacial dos clusters.
    */
int GNG_CPP::getSamplesSize()
{
    if( _gCenters.size() == 0 )
        return -1;
    int size;
    Cluster* cluster0 = _gCenters[0];
    cluster0->getClusterSamplesSize( size );
    return size;
}

}
