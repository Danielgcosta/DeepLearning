/*
 * File:   Cluster.h
 * Author: Aurélio
 *
 * Created on September 22, 2009, 3:43 PM
 */

#ifndef _CLUSTER_H
#define	_CLUSTER_H

#include <vector>
#include <math.h>
#include <stdio.h>
#include "lusystem.h"
#include "../data/stl_utils.h"

#ifdef USING_OPENCV
#include "cv.h"
#include "cxcore.h"
#endif

namespace MSA
{

class Cluster;

/**
* Funcão de multiplicacao entre duas matrizes.
* @param ma Número de linhas da matriz a.
* @param na Número de colunas da matriz a.
* @param mb Número de linhas da matriz b.
* @param nb Número de colunas da matriz b.
* @param a Matriz a.
* @param b Matriz b.
* @param mr Matriz resultante
* @return Retorna -1 em caso de erro, ou 0 caso ok.
*/
  static int matrixMultiplication(int ma, int na, int mb, int nb, double** a, double** b, double**& mr)
{
  int i, j , v;

  if( (a==NULL) || (b==NULL) || (mr==NULL) )
    return -1;

  if (na != mb)
    return -1;

  for (i = 0 ; i < ma; i++ )
    for (j = 0; j < nb; j++)
      for (v = 0; v < na; v++)
        mr[i][j] = mr[i][j] + a[i][v] * b[v][j];

  return 0;
}


  /**
    * Funcão auxiliar. Verifica se duas matrizes são iguais.
    */
  static bool equal( double**a, double**b, int w, int h )
  {
    for( int i=0 ; i<w ; i++ )
      for( int j=0 ; j<h ; j++ )
      {
        if( a[i][j] != b[i][j] )
          return false;
      }
    return true;
  }


  static bool printMatix( double** a, int l, int c )
  {
    FILE* f = NULL;
    f = fopen( "out.txt", "at" );
    if( f == NULL )
      return -1;
    // Imprime a invCorrelationMatrix:
    for(int i=0;i<l;i++)
    {
      for(int j=0;j<c;j++)
        if( fprintf( f, "%.6lf, ", a[i][j] ) < 0 )
          return false;
      if( fprintf( f, "\n" ) < 0 )
        return false;
      if( i == l-1 )
        if( fprintf( f, "\n" ) < 0 )
          return false;
    }
    fclose(f);
    return true;
  }


/**
  * Retorna a distância de Mahalanobis entre dois vetores. Recebe a inversa da matriz de
  * covariância relacionada à comparacão.
  * @param sampleVec1 primeiro vetor.
  * @param meanVec segundo vetor.
  * @param invCorrelationMatrix Inversa da matriz de correlacão envolvida.
  * @param size Dimensão dos vetores (e número de linhas e colunas da matriz).
  * @param auxMatrix Matriz auxiliar, evita realocacão de memória.
  * @return A funcão retorna -1 em caso de erro nos parâmetros, ou a distância obtida.
  */
static double fMahalanobisDist( float* sampleVec1, double* meanVec, double** invCorrelationMatrix, int size,
                                double** auxMatrix1, double** auxMatrix2 )
{
  // Caso vetores ou matriz sejam nulos, retorna erro:
  if( (sampleVec1==NULL) || (meanVec==NULL) || (invCorrelationMatrix==NULL) ||
      (auxMatrix1==NULL) || (auxMatrix2==NULL) )
    return -1;

  // Evitando alocacão. Admitindo o tamanho máximo de "size" como 1024 (na prática não chega a isso).
  if( size > 1024 )
    return -1;

  for( int i=0 ; i<size ; i++ )
  {
    for( int j=0 ; j<size ; j++ )
    {
      auxMatrix1[i][j] = 0;
      auxMatrix2[i][j] = 0;
    }
  }

  // Caso contrário, calcula a distância de Mahalanobis:
  for( int i=0 ; i<size ; i++ )
    auxMatrix2[0][i] = sampleVec1[i] - meanVec[i];

  int ret = matrixMultiplication(1, size, size, size, auxMatrix2, invCorrelationMatrix, auxMatrix1 );
  if( ret == -1 )
    return -1;

//    // Imprime a invCorrelationMatrix:
//    for(int i=0;i<size;i++) {
//      for(int j=0;j<size;j++)  std::cout << invCorrelationMatrix[i][j] << " ";
//      std::cout << std::endl;
//      if( i == size-1 ) std::cout << std::endl;
//    }

//    // Imprime a auxMatrix2:
//    for(int i=0;i<size;i++) {
//      for(int j=0;j<size;j++)  std::cout <<  auxMatrix2[i][j] << " ";
//      std::cout << std::endl;
//      if( i == size-1 ) std::cout << std::endl;
//    }

//    // Imprime a auxMatrix1:
//    for(int i=0;i<size;i++) {
//      for(int j=0;j<size;j++)  std::cout <<  auxMatrix1[i][j] << " ";
//      std::cout << std::endl;
//      if( i == size-1 ) std::cout << std::endl;
//    }

  double retDist = 0;
  for( int i=0 ; i<size ; i++ )
    retDist += auxMatrix1[0][i] * auxMatrix2[0][i];

  if( retDist < 0 )
    ret=1;
  return sqrt(retDist);
}


/**
  * Funcão auxiliar. Recebe uma matriz quadrada e retorna a sua inversa.
 * @param inMatrix Matriz quadrada para a qual obter a inversa.
 * @param invMatrx Matriz inversa, retornada caso não existam erros.
 * @param size Número de dimensơes da matriz.
 * @param inAuxMatrix Matriz auxiliar (opcional) que guardará valores temporários durante os
 *        cálculos. Caso seja recebida, evita realocacão e posterior delecão de memória.
 * @param inIndx Vetor auxiliar (opcional) que guardará valores temporários durante os
 *        cálculos. Caso seja recebido, evita realocacão e posterior delecão de memória.
 * @param inB Vetor auxiliar (opcional) que guardará valores temporários durante os
 *        cálculos. Caso seja recebido, evita realocacão e posterior delecão de memória.
  */
static int inverseMatrix( double** inMatrix, double** invMatrix, int size,
                          double** inAuxMatrix=NULL, int* inIndx=NULL, double* inB=NULL )
{
  if( (inMatrix==NULL) || (invMatrix == NULL) )
    return -1;

  int *indx = NULL;
  double *b = NULL;
  double** auxMatrix=NULL;

  // Aloca memória para a matriz auxiliar, caso não tenha sido recebida:
  if( inAuxMatrix != NULL )
    auxMatrix = inAuxMatrix;
  else
    {
      auxMatrix = new double*[size];
      for( int i=0 ; i<size ; i++ )
        auxMatrix[i] = new double[size];
    }

  // Aloca memória para o vetor de índices caso não tenha sido recebido:
  if( inIndx != NULL )
    indx = inIndx;
  else
    indx = new int[size];

  // Aloca memória para o vetor de b's caso não tenha sido recebido:
  if( inB != NULL )
    b = inB;
  else
    b = new double[size];

  // Inicializa os vetores e resolvendo cada sistema, para encontrar a matriz:
  for( int i=0 ; i<size ; i++ )
  {
    for( int j=0 ; j<size ; j++ )
    {
      indx[j] = 0;
      b[j] = 0;
    }
    for( int i2=0 ; i2<size ; i2++ )
      for( int j2=0 ; j2<size ; j2++ )
        auxMatrix[i2][j2] = inMatrix[i2][j2];

    b[i] = 1;
    int ret = LuDcmp(auxMatrix, size, indx);
    // Caso o valor de retorno não seja 1, houve erro na resolucão do sistema:
    if( ret != 1 )
    {
      // Libera a memória de vetores e matriz, caso tenha sido preciso alocar:
      if( inAuxMatrix == NULL )
      {
        for( int i=0 ; i<size ; i++ )
          delete[] auxMatrix[i];
        delete[] auxMatrix;
      }
      if( inIndx == NULL )
          delete[] indx;
      if( inB == NULL )
          delete[] b;
      return -1;
    }
    // Caso o valor de retorno tenha sido o correto (1), resolve o sistema e insere o resultado na coluna:
    LuBksb( auxMatrix, size, indx, b );
    for( int j=0 ; j<size ; j++ )
      invMatrix[j][i] = b[j];
  }

  // Libera a memória de vetores e matriz, caso tenha sido preciso alocar:
  if( inAuxMatrix == NULL )
  {
    for( int i=0 ; i<size ; i++ )
      delete[] auxMatrix[i];
    delete[] auxMatrix;
  }
  if( inIndx == NULL )
      delete[] indx;
  if( inB == NULL )
      delete[] b;

  return 0;
}



#ifdef USING_OPENCV
/**
  * Funcão auxiliar. Calcula a matriz de covariâncias de um grupo de vetores recebido, floating point.
  */
template <typename type1>
static int getCovarianceMatrix( std::vector<type1*>& samplesVector, int samplesSize,
                                double**& covarMatrix, double**& covarMatrixInverse  )
{
    // Número de amostras contidas no vetor:
    int samplesVectorsize = samplesVector.size();

    if( samplesSize <= 0 )
      return -1;

    if( (samplesVectorsize < 1) || (covarMatrix == NULL) )
      return -1;

    CvMat* output = cvCreateMat( samplesSize, samplesSize, CV_32FC1 );
    CvMat* outputInverse = cvCreateMat( samplesSize, samplesSize, CV_32FC1 );
    CvMat* meanvec = cvCreateMat( 1, samplesSize, CV_32FC1 );

    cvZero( output );
    cvZero( outputInverse );
    cvZero( meanvec );

    CvMat** input = new CvMat*[samplesVectorsize];
    for(int i=0; i<samplesVectorsize; i++)
    {
        input[i] = cvCreateMat( 1, samplesSize, CV_32FC1 );
        for( int j=0 ; j<samplesSize ; j++ )
        {
            float value = (float)(samplesVector[i][j]);
            cvmSet( input[i], 0, j, value );
        }
    }
    // Calcula a matriz de covariâncias:
    cvCalcCovarMatrix( (const void **) input, samplesVectorsize, output, meanvec, CV_COVAR_NORMAL /*| CV_COVAR_SCALE*/ );

    // Obtendo a inversa da matriz de covariâncias:
    cvInvert( output, outputInverse, CV_SVD );

    // Copiando os valores para as variáveis a serem retornadas:
    for( int i=0 ; i<samplesSize ; i++ )
    {
        for( int j=0 ; j<samplesSize ; j++ )
        {
            covarMatrix[i][j] = cvmGet( output, i, j );
            covarMatrixInverse[i][j] = cvmGet( outputInverse, i, j );
        }
    }

    // Deletando estruturas de dados temporárias:
    cvReleaseMat( &output );
    cvReleaseMat( &meanvec );
    for( int i=0; i<samplesVectorsize; i++ )
        cvReleaseMat( &input[i] );
}
#endif

/**
  * Funcão auxiliar. Recebe dois vetores em R1, floating point, e calcula a covariância entre eles.
  * caso as médias não sejam recebidas a funcão retorna erro.
  */
template <typename type1>
static int getCovarianceMatrix( std::vector<type1*>& samplesVector, int samplesSize, float* meanVector,
                                double**& covarMatrix, double**& covarMatrixInverse  )
{
  // Número de amostras contidas no vetor:
  int samplesVectorsize = samplesVector.size();

  if( samplesSize <= 0 )
    return -1;

  if( (samplesVectorsize < 1) || (meanVector == NULL) || (covarMatrix == NULL) )
    return -1;

  for( int i=0 ; i<samplesSize ; i++ )
  {
    for( int j=0 ; j<samplesSize ; j++ )
    {
      //if( i >= j )
      {
        long double covar=0;
        for( int k=0 ; k<samplesVectorsize ; k ++ )
        {
          covar += ( (samplesVector[k][i]-meanVector[i])*(samplesVector[k][j]-meanVector[j]) );
        }
        covar = covar/samplesVectorsize;
        covarMatrix[i][j] = covar;
//        covarMatrix[j][i] = covar;
      }
    }
  }

//  printMatix( covarMatrixInverse, samplesSize, samplesSize );

  // Calcula a inversa da matriz de covariâncias:
  if( inverseMatrix( covarMatrix, covarMatrixInverse, samplesSize ) == -1 )
    return -1;

//  printMatix( covarMatrixInverse, samplesSize, samplesSize );

  return 0;
}


/**
  * Funcão auxiliar, calcula média e variância de cada um dos componentes do vetor recebido.
  */
template <typename type1>
static int getStatistics( std::vector<type1*>& samplesVector, int samplesSize, float*& meanVector, float*& varVector )
{
  int samplesVectorsize = samplesVector.size();

  if( samplesSize <= 0 )
    return -1;

  if( (samplesVectorsize < 1) || (meanVector == NULL) || (varVector == NULL) )
    return -1;


  std::vector<double> tmpMeanVector;
  tmpMeanVector.reserve( samplesSize );
  tmpMeanVector.resize( samplesSize );

  std::vector<double> tmpVarVector;
  tmpVarVector.reserve( samplesSize );
  tmpVarVector.resize( samplesSize );

  for( int j=0 ; j<samplesSize ; j++ )
  {
    meanVector[j] = 0;
    varVector[j] = 0;
    tmpMeanVector[j] = 0;
    tmpVarVector[j] = 0;
  }

  // Médias de cada componente:
  for( int i=0 ; i<samplesVectorsize ; i++ )
  {
    if( samplesVector[i] == NULL )
      return -1;
    for( int j=0 ; j<samplesSize ; j++ )
      tmpMeanVector[j] += samplesVector[i][j];
  }

  for( int j=0 ; j<samplesSize ; j++ )
    tmpMeanVector[j] = tmpMeanVector[j]/samplesVectorsize;

  // Variâncias de cada componente:
  for( int i=0 ; i<samplesVectorsize ; i++ )
  {
    if( samplesVector[i] == NULL )
      return -1;
    for( int j=0 ; j<samplesSize ; j++ )
      tmpVarVector[j] += ( (samplesVector[i][j]-tmpMeanVector[j])*(samplesVector[i][j]-tmpMeanVector[j]) );
  }
  for( int j=0 ; j<samplesSize ; j++ )
    tmpVarVector[j] = sqrt(tmpVarVector[j]/samplesVectorsize);

  // Copiando os valores para o vetor a ser retornado:
  for( int j=0 ; j<samplesSize ; j++ )
  {
    meanVector[j] = (float)(tmpMeanVector[j]);
    varVector[j] = (float)(tmpVarVector[j]);
  }
  return 0;
}


/**
 * Classe que comporta o conceito de um cluster genérico. A idéia central que motivou
 * a existência dessa classe é que os mais diversos algoritmos de agrupamento de dados
 * (data clustering) tais como k-means, Neural Gas, Growing Neural Gas, Self-Organizing Maps
 * entre outros produzem como resultado final um vetor de clusters, que pode ser armazenado
 * sem a existência da instância do alguritmo que o gerou. A classe Cluster, portanto,
 * armazena informacões a respeito de Clusters para futura utilizacão.
 */
class Cluster
{
public:
  int _clId; ///< Identificador numérico desse cluster.
  int _clSize; ///< Dimensão.
  float* _clCenter; ///< Centro do cluster.
  float _clError; ///< Erro total do cluster (não necessariamente disponível).
  int _clNumSamples; ///< Número de amostras classificadas nesse cluster (não necessariamente disponível).
  float _clRelativeDistance; ///< Distância relativa do cluster aos seus vizinhos (média dos n vizinhos mais próximos).
  float* _clMean;            ///< Média de cada componente, de todas as amostras pertencentes ao cluster.
  float* _clVar;             ///< Variância de cada componente, das amostras pertencentes ao cluster.
  double**_clCovarMatrix;     ///< Matriz de covariância dos componentes das amostras pertencentes ao cluster.

  std::vector<float*> _clSamples; ///< Vetor de amostras classificadas no cluster, caso disponível.

  // Inversa da matriz de covariâncias, utilizada somente caso necessário.
  // Não será salva no caso de armazenarmos uma instância em disco:
  double**_clCovarMatrixInv;

  // Duas matrizes auxiliares, são alocadas para evitar realocacão a todo momento:
  double** _clAuxMatrix1;
  double** _clAuxMatrix2;


  /**
    * Retorna a distância de Mahalanobis entre dois vetores. Caso o segundo vetor seja nulo, é utilizado
    * o clusterCenter
    * @param sampleVec1 primeiro vetor.
    * @param sampleVec2 segundo vetor.
    * @return A funcão retorna -1 em caso de erro nos parâmetros, ou a distância obtida.
    */
  double getMahalanobisDist( float* sampleVec1, float* sampleVec2=NULL )
  {
      if( sampleVec1 == NULL )
          return -1;
      if( sampleVec2 == NULL )
          sampleVec2 = _clCenter;

      if( _clSize > 1024 )
          return -1;
      double tmpVec[1024];
      double* tmpVecPt = &(tmpVec[0]);
      for( int i=0 ; i<_clSize ; i++ )
          tmpVecPt[i] = (double)sampleVec2[i];
      double ret = fMahalanobisDist( sampleVec1, tmpVecPt, _clCovarMatrixInv,  _clSize, _clAuxMatrix1, _clAuxMatrix2 );
      return ret;
  }


#ifdef USING_OPENCV
  void PrintMat(CvMat *A)
  {
      int i, j;
      for (i = 0; i < A->rows; i++)
      {
          std::cout << std::endl;
          switch (CV_MAT_DEPTH(A->type))
          {
          case CV_32F:
              for (j = 0; j < A->cols; j++)
                  std::cout << (float)cvGetReal2D(A, i, j) << " " ;
              break;
          case CV_64F:
              for (j = 0; j < A->cols; j++)
                  std::cout << (float)cvGetReal2D(A, i, j) << " " ;
              break;
          case CV_8U:
          case CV_16U:
              for(j = 0; j < A->cols; j++)
                  std::cout << (int)cvGetReal2D(A, i, j) << " " ;
              break;
          default:
              break;
          }
      }
      std::cout << std::endl;
  }


  /**
    * Distortion.
    */
  double distortionCalc( CvMat* A, CvMat* B, CvMat* T, CvMat* Aux1, CvMat* Aux2, CvMat* Aux3 )
  {
      // Fazendo Aux = A - B:
      int samplesSize = A->cols;
      for( int j=0 ; j<samplesSize ; j++ )
          cvmSet( Aux1, 0, j, (cvmGet(A, 0, j) - cvmGet(B, 0, j)) );

//      PrintMat( A );
//      PrintMat( B );
//      PrintMat( Aux1 );
//      PrintMat( T );

      // Encontrando a transposta:
      cvTranspose( Aux1, Aux3 );
//      PrintMat( Aux3 );
      // T * Aux3 -> Aux3:
      cvMatMul( T, Aux3, Aux2 );
//      PrintMat( Aux2 );

      // inner Aux2 Aux1:
      double res = 0;
      for( int j=0 ; j<samplesSize ; j++ )
           res += (cvmGet(Aux2, j, 0) * cvmGet(Aux3, j, 0));

      return res;
  }


  /**
    * Funcão auxiliar. Recebe dois vetores em R1, floating point, e calcula a covariância entre eles.
    * caso as médias não sejam recebidas a funcão retorna erro.
    */
  double getClusterEstimatedDistortion( std::vector<float*>& samplesVector, int samplesSize, double** covarMatrixInv = NULL )
  {
    // Número de amostras contidas no vetor:
    int samplesVectorsize = samplesVector.size();

    if( samplesSize <= 0 )
      return -1;

    if( samplesVectorsize < 1 )
      return -1;

    // Caso a matriz recebida seja nula, utiliza a matriz de covariâncias do cluster:
    if( covarMatrixInv == NULL )
        covarMatrixInv = _clCovarMatrixInv;

    // Criando os dois vetores necessários no cálculo:
    CvMat* A = cvCreateMat( 1, samplesSize, CV_32FC1 );
    CvMat* B = cvCreateMat( 1, samplesSize, CV_32FC1 );

    // Criando a matriz necessária (matriz de covariâncias inversa):
    CvMat* T = cvCreateMat( samplesSize, samplesSize, CV_32FC1 );
    for( int i=0 ; i<samplesSize; i++ )
        for( int j=0 ; j<samplesSize ; j++ )
             cvmSet( T, i, j, covarMatrixInv[i][j] );

    // Copiando o centroide para a estrutura necessária:
    for( int j=0 ; j<samplesSize ; j++ )
        cvmSet( B, 0, j, _clCenter[j]) ;


    // Criando vetores auxiliares:
    CvMat* Aux1 = cvCreateMat( samplesSize, samplesSize, CV_32FC1 );
    CvMat* Aux2 = cvCreateMat( samplesSize, samplesSize, CV_32FC1 );
    CvMat* Aux3 = cvCreateMat( samplesSize, samplesSize, CV_32FC1 );
    cvZero( Aux1 );
    cvZero( Aux2 );
    cvZero( Aux3 );

    // Calculando o somatório das distorcões:
    double distortion = 0;
    for( int i=0 ; i<samplesVectorsize ; i++ )
    {
        cvZero( Aux1 );
        cvZero( Aux2 );
        cvZero( Aux3 );
        for( int j=0 ; j<samplesSize ; j++ )
            cvmSet( A, 0, j, (float)(samplesVector[i][j]) );
        distortion += cvMahalonobis( A, B, T );
//        distortion += distortionCalc( A, B, T, Aux1, Aux2, Aux3 );
    }

    // Deletando estruturas de dados temporárias:
    cvReleaseMat( &A );
    cvReleaseMat( &B );
    cvReleaseMat( &T );

    cvReleaseMat( &Aux1 );
    cvReleaseMat( &Aux2 );
    cvReleaseMat( &Aux3 );

    return distortion/samplesVectorsize;
  }
#endif

  /**
   * Função auxiliar que permite salvar o conteúdo do cluster no arquivo de ponteiro recebido.
   * @param f Ponteiro para o arquivo sendo escrito.
   * @return Retorna -1 em caso de erro, ou 0 caso ok.
   */
  int saveCluster( FILE *f )
  {
    // Casos de retorno:
    if( f == NULL )
      return -1;

    if( fprintf( f, "_clId: %d\n", _clId ) < 0 ) return -1;
    if( fprintf( f, "_clSize: %d\n", _clSize ) < 0 ) return -1;
    if( fprintf( f, "_clError: %f\n", _clError ) < 0 ) return -1;
    if( fprintf( f, "_clNumSamples: %d\n", _clNumSamples ) < 0 ) return -1;
    if( fprintf( f, "_clRelativeDistance: %f\n", _clRelativeDistance ) < 0 ) return -1;

    // Salva os vetores de média e variância:
    if( fprintf( f, "_clMean, _clVar:\n" ) < 0 ) return -1;
    int i;
    for( i = 0; i < _clSize; i++ )
      if( fprintf( f, "%f, %f\n", _clMean[i], _clVar[i] ) < 0 ) return -1;

    // Salva a matriz de covariâncias:
    if( fprintf( f, "_clCovarMatrix:\n" ) < 0 ) return -1;
    int j;
    for( i = 0; i < _clSize; i++ )
    {
      for( j = 0; j < _clSize; j++ )
        if( fprintf( f, "%lf, ", _clCovarMatrix[i][j] ) < 0 ) return -1;
      if( fprintf( f, "\n" ) < 0 ) return -1;
    }

    // Salva o vetor de pesos:
    if( fprintf( f, "_clCenter:\n" ) < 0 ) return -1;
    for( i = 0; i < _clSize; i++ )
      if( fprintf( f, "%f\n", _clCenter[i] ) < 0 ) return -1;
    return 0;
  }


  /**
   * Funcão que permite ler o conteúdo de um cluster a partir de um arquivo.
   * @param f Ponteiro para o arquivo sendo escrito.
   * @return Retorna -1 em caso de erro, ou 0 caso ok.
   */
  int loadCluster( FILE*&f )
  {
    // Casos de retorno:
    if( f == NULL )
      return -1;

    if( fscanf( f, "_clId: %d\n", &_clId ) != 1 ) return -1;
    if( fscanf( f, "_clSize: %d\n", &_clSize ) != 1 ) return -1;
    if( fscanf( f, "_clError: %f\n", &_clError ) != 1 ) return -1;
    if( fscanf( f, "_clNumSamples: %d\n", &_clNumSamples ) != 1 ) return -1;
    if( fscanf( f, "_clRelativeDistance: %f\n", &_clRelativeDistance ) != 1 ) return -1;

    // Vetores de média e variância:
    _clMean = new float[_clSize];
    _clVar = new float[_clSize];
    if( fscanf( f, "_clMean, _clVar:\n" ) != 0 ) return -1;
    int i;
    for( i = 0; i < _clSize; i++ )
    {
      if( fscanf( f, "%f, %f\n", &( _clMean[i] ), &( _clVar[i] ) ) != 2 ) return -1;
      float a = sqrt( _clVar[i] );
      a = a + 1e-4;
    }

    // Matriz de covariâncias:
    _clCovarMatrix = new double*[_clSize];
    if( fscanf( f, "_clCovarMatrix:\n" ) != 0 ) return -1;
    int j;
    for( i = 0; i < _clSize; i++ )
    {
      _clCovarMatrix[i] = new double[_clSize];
      for( j = 0; j < _clSize; j++ )
        if( fscanf( f, "%lf, ", &( _clCovarMatrix[i][j] ) ) != 1 ) return -1;
      if( fscanf( f, "\n" ) != 0 ) return -1;
    }

    // Vetor de pesos:
    _clCenter = new float[_clSize];
    if( fscanf( f, "_clCenter:\n" ) != 0 ) return -1;
    for( i = 0; i < _clSize; i++ )
      if( fscanf( f, "%f\n", &( _clCenter[i] ) ) != 1 ) return -1;

    return 0;
  }


  /**
    * Retorna uma cópia do cluster atual.
    */
  Cluster* copy()
  {
    Cluster* retCopy = new Cluster( _clSize, _clCenter, _clError, _clNumSamples, _clId, _clMean, _clVar,
                                    _clCovarMatrix );
//    // Calcula a inversa da matriz de covariâncias:
//    inverseMatrix( retCopy->_clCovarMatrix, retCopy->_clCovarMatrixInv, _clSize );
    return retCopy;
  }


  /**
   * Destrutor.
   */
  ~Cluster( )
  {
      delete[] _clCenter;
      delete[] _clMean;
      delete[] _clVar;

      int i = 0;
      for(; i < _clSize; i++ )
      {
          delete[] _clCovarMatrix[i];
          delete[] _clCovarMatrixInv[i];
          delete[] _clAuxMatrix1[i];
          delete[]  _clAuxMatrix2[i];
      }
      delete[] _clCovarMatrix;
      delete[] _clCovarMatrixInv;
      delete[] _clAuxMatrix1;
      delete[] _clAuxMatrix2;

      vectorFreeMemory( _clSamples );
  }


  /**
   * Construtor.
   */
  Cluster( int size, float*& center, float error, int numSamples, int id, float* mean=NULL,
           float* var=NULL, double** covarMatrix=NULL )
  {
    _clSize = size;
    _clCenter = new float[size];
    _clMean = new float[size];
    _clVar = new float[size];
    _clCovarMatrix = new double*[size];
    _clCovarMatrixInv = new double*[size];

    int i = 0;
    for(; i < size; i++ )
    {
      // Vetor de posicão:
      _clCenter[i] = center[i];
      // Vetor de médias:
      if( mean != NULL )
        _clMean[i] = mean[i];
      else
        _clMean[i] = 0;
      // Vetor de variâncias:
      if( var != NULL )
        _clVar[i] = var[i];
      else
        _clVar[i] = 0;

      // Matriz de covariâncias:
      _clCovarMatrix[i] = new double[size];
      _clCovarMatrixInv[i] = new double[size];
      int j = 0;
      if( covarMatrix != NULL)
      {
        for(; j < size; j++ )
        {
          _clCovarMatrix[i][j] = covarMatrix[i][j];
          _clCovarMatrixInv[i][j] = 0;
        }
      }
      else
      {
        for(; j < size; j++ )
        {
          _clCovarMatrix[i][j] = 0;
          _clCovarMatrixInv[i][j] = 0;
        }
      }
    }

    _clError = error;
    _clNumSamples = numSamples;
    _clId = id;

    // Alocando espaco para as matrizes auxiliares de uso geral:
    _clAuxMatrix1 = new double*[size];
    _clAuxMatrix2 = new double*[size];
    for( int i=0 ; i<size ; i++ )
    {
      _clAuxMatrix1[i] = new double[size];
      _clAuxMatrix2[i] = new double[size];
    }
  }


  /**
   * Construtor.
   */
  Cluster( FILE*& f )
  {
    // Carrega o cluster:
    loadCluster( f );
    // Inversa da matriz de covariâncias:
    _clCovarMatrixInv = new double*[_clSize];
    int i = 0;
    for(; i < _clSize; i++ )
    {
      _clCovarMatrixInv[i] = new double[_clSize];
      int j = 0;
      for(; j < _clSize; j++ )
        _clCovarMatrixInv[i][j] = 0;
    }

    // Alocando espaco para as matrizes auxiliares de uso geral:
    _clAuxMatrix1 = new double*[_clSize];
    _clAuxMatrix2 = new double*[_clSize];
    for( int i=0 ; i<_clSize ; i++ )
    {
      _clAuxMatrix1[i] = new double[_clSize];
      _clAuxMatrix2[i] = new double[_clSize];
    }
  }


  /**
   * Retorna a posicão espacial do cluster.
   */
  void getPos( int& size, float*& center )
  {
    size = _clSize;
    center = _clCenter;
  }


  /**
   * Retorna o erro e o número de amostras que foram levadas em conta no seu
   * cálculo. numSamples = -1 indica que não existe esse dado.
   */
  void getErrorInf( float& error, int& numSamples )
  {
    error = _clError;
    numSamples = _clNumSamples;
  }


  /**
   * Obtém o número identificador do cluster.
   */
  void getClusterId( int& id )
  {
    id = _clId;
  }


  /**
   * Obtém a dimensão (espaco dimensional) das amostras do cluster.
   */
  void getClusterSamplesSize( int& samplesSize )
  {
    samplesSize = _clSize;
  }


  /**
   * Insere o número identificador do cluster.
   */
  void setClusterId( int id )
  {
    _clId = id;
  }


  /**
   * Insere o erro e o número de amostras que foram levadas em conta no seu
   * cálculo. numSamples = -1 indica que não existe esse dado.
   */
  void setErrorInf( float error, int numSamples=-1 )
  {
    _clError = error;
    if( numSamples != -1 )
      _clNumSamples = numSamples;
  }


  /**
   * Soma um novo valor de erro ao erro já existente.
   */
  void incErrorInf( float error )
  {
    _clError += error;
    _clNumSamples++;
    _clRelativeDistance = _clError / _clNumSamples;
  }


  /**
   * Insere o valor da distância relativa do cluster.
   */
  void setRelativeDistance( float clRelativeDistance )
  {
    _clRelativeDistance = clRelativeDistance;
  }


  /**
    * Retorna o vetor de médias obtido através das amostras classificadas no cluster.
    */
  float* getMeansVector()
  {
    return _clMean;
  }


  /**
    * Retorna o vetor de variâncias obtido através das amostras classificadas no cluster.
    */
  float* getVariancesVector()
  {
    return _clVar;
  }


  /**
    * Retorna a matriz de covariâncias obtida através das amostras classificadas no cluster.
    */
  double** getCovariancesMatrix()
  {
    return _clCovarMatrix;
  }



  /**
   * Obtém a distância relativa.
   */
  void getRelativeDistance( float& clRelativeDistance )
  {
    clRelativeDistance = _clRelativeDistance;
  }


  /**
   * Insere uma amostra que foi classificada dentro desse cluster.
   */
  int setSample( float*& sample, int sampleSize )
  {
    if( ( sample == NULL ) || ( sampleSize != _clSize ) )
      return -1;
    _clSamples.push_back( sample );
    return 0;
  }

};

}

#endif	/* _CLUSTER_H */
