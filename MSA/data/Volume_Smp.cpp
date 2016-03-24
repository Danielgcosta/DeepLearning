

#include "Volume_Smp.h"
#include "DVPVolumeData.h"
#include "DeepLearningNetwork.h"

using namespace MSA;

namespace MSA
{

int Volume_Smp::getSPCAFeatures( std::vector<float*> samples, int sampleDim, int nFeatures, std::vector<float*>& features )
{
	if( samples.size() == 0 )
		return -1;

	if( nFeatures >= sampleDim )
		return -1;


	// Encontrando valores de mínimo e máximo dos voxels:
	double min, max;
	min = std::numeric_limits<double>::max();
	max = -std::numeric_limits<double>::max();
	for( int i=0 ; i<(int)samples.size() ; i++ )
	{
		for( int j=0 ; j<sampleDim ; j++ )
		{
			if( min > samples[i][j] )
				min = (double)samples[i][j];
			if( max < samples[i][j] )
				max = (double)samples[i][j];
		}
	}

	// Criando uma matriz que contenha as amostras de entrada, com média 0 e valores entre 0 e 1:
	double** samplesMatrix = new double*[samples.size()];
	double*samplesVector = new double[sampleDim*samples.size()];
	for( int i=0 ; i<(int)samples.size() ; i++ )
	{
		samplesMatrix[i] = &(samplesVector[i*sampleDim]);
		for( int j=0 ; j<sampleDim ; j++ )
		{
			samplesMatrix[i][j] = (double)(samples[i][j]);
			// Escalonando os vetores entre 0 e 1:
			scaleIn( 1.0, 0.0, max, min, samplesMatrix[i][j] );
		}
	}

	// Fazendo a matriz com média 0:
	toMeanZero( samplesMatrix, samples.size(),  sampleDim );


	// Criando espaco para a matriz que irá conter os novos vetores de
	// características gerados pelo SPCA. Para manter o tamanho das amostras, alocamos os vetores com tamanho
	// sampleDim ao invés de nFeatures, e então completamos as amostras com ZERO:
	float* matfeaturesVecSPCAfloat = new float[samples.size() * sampleDim];

	for( unsigned int i=0 ; i< (samples.size() * (unsigned int)sampleDim) ; i++ )
	{
		matfeaturesVecSPCAfloat[i] = 0.0f;
	}

	// Reserva memoria para o vetor de features:
	features.reserve( (int)samples.size() );
	features.resize( (int)samples.size() );

	for( int i=0 ; i<(int)samples.size() ; i++ )
	{
		float* tmp = &(matfeaturesVecSPCAfloat[i*sampleDim/*nFeatures*/]);
		features[i] = tmp;
	}

	// Iniciando a parte de calculo do SPCA:
	SPCA* SPCAtraining = new SPCA( sampleDim );
	double* alphasDiff = new double[sampleDim];
	double greaterDiff = 0;

	int featuresSPCA_Cont = 0;
	while( featuresSPCA_Cont < nFeatures )
	{
		// Copia os alphas pra dentro do vetor auxiliar:
		CopyVector( SPCAtraining->_spca_alphas, alphasDiff, sampleDim );

		// Calcula a primeira iteracão do SPCA:
		toMeanZero( samplesMatrix, samples.size(), sampleDim );

		SPCAtraining->SPCA_Next_Alpha( samplesMatrix, samples.size() );
		greaterDiff = GreaterDiff( SPCAtraining->_spca_alphas, alphasDiff, sampleDim );

		int numIterations = 0;
		for( ; numIterations<18 ; numIterations++ )
		{
			CopyVector( SPCAtraining->_spca_alphas, alphasDiff, sampleDim );
			SPCAtraining->SPCA_Next_Alpha( samplesMatrix, samples.size() );
			greaterDiff = GreaterDiff( SPCAtraining->_spca_alphas, alphasDiff, sampleDim );
			std::cout << "		Iterations: " << numIterations << " GreaterDiff: " << greaterDiff << std::endl;

			if( greaterDiff <= 1e-05 )
				break;
		}
		std::cout << "Iterations: " << numIterations << std::endl;


		// Nesse ponto, posso pegar o valor da característica para todas as amostras:
		for( int i=0 ; i<(int)samples.size() ; i++ )
		{
			features[i][featuresSPCA_Cont] = (float)(SPCAtraining->SPCA_Y( samplesMatrix[i] ));
		}

		// Salva os arquivos de características nFeatures dimensões:
		if( featuresSPCA_Cont == (nFeatures-1) )
		{
			char spcaFeaturesFileN[] = "SPCA_Features";
			char* spcaFeaturesFileNPt = &(spcaFeaturesFileN[0]);

			char spcaFeaturesFileName[128];
			sprintf( spcaFeaturesFileName, "%s_%d", spcaFeaturesFileNPt, featuresSPCA_Cont+1 );
			Save_File( features, nFeatures, spcaFeaturesFileName );
		}

		// Realizando o "deflaten":
		SPCAtraining->SPCA_entries_deflation( samplesMatrix, samples.size() );

		featuresSPCA_Cont++;
	}

	// Deleta os dados alocados:
	delete alphasDiff;
	delete SPCAtraining;
	delete samplesVector;
	delete[] samplesMatrix;

	return 0;
}


int Volume_Smp::getDEEP_LEARNINGFeatures( std::vector<float*> samples, int sampleDim, int nFeatures, std::vector<float*>& features )
{
	if( samples.size() == 0 )
		return -1;

	if( nFeatures >= sampleDim )
		return -1;


	// Encontrando valores de mi�nimo e maximo dos voxels:
	float min, max;
	min = std::numeric_limits<float>::max();
	max = std::numeric_limits<float>::min();
	for( int i=0 ; i<(int)samples.size() ; i++ )
	{
		for( int j=0 ; j<sampleDim ; j++ )
		{
			if( min > samples[i][j] )
				min = samples[i][j];
			if( max < samples[i][j] )
				max = samples[i][j];
		}
	}

	// Criando uma matriz que contenha as amostras de entrada, com  valores entre 0.1 e 0.9:
	Matrix samplesMatrix( samples.size(), sampleDim );
	for( unsigned int i=0 ; i<samples.size() ; i++ )
	{
		for( unsigned int j=0 ; j<(unsigned int)sampleDim ; j++ )
		{
			float value = samples[i][j];
			// Escalonando os vetores entre 0 e 1:
			scaleIn( (float)0.9, (float)0.1, max, min, value );
			samplesMatrix.set( i, j, (double)(value) );
		}
	}

	// Encontrando o vetor de layersSizes contendo o tamanho de cada uma das camadas da rede:
	std::vector<unsigned int> layersSizes;
	unsigned int layerDim = (unsigned int)sampleDim;

	// Inserindo num de neuronios na camada de entrada:
	layersSizes.push_back( layerDim );
	while( layerDim > (unsigned int)(nFeatures) )
	{
		layerDim = Rounded(layerDim * 0.60f);
		layersSizes.push_back( layerDim );
	}
	// Inserindo num de neuronios na camada de saída:
	layersSizes.push_back( (unsigned int)nFeatures );


	// Criando a rede de DeepLearning:
	DeepLearningNetwork* myDeepLearningNetwork = new DeepLearningNetwork( layersSizes );
	myDeepLearningNetwork->train( samplesMatrix );

	// Criando espaco para a matriz que irá conter os novos vetores de características gerados pelo DeepLearning.
	// Para manter o tamanho das amostras, alocamos os vetores com tamanho sampleDim ao invés de nFeatures,
	// e então completamos as amostras com ZERO:
	float* matfeaturesVecDL = new float[samples.size() * sampleDim];
	for( unsigned int i=0 ; i< (samples.size() * (unsigned int)sampleDim) ; i++ )
		matfeaturesVecDL[i] = 0.0f;

	// Reserva memoria para o vetor de features:
	features.reserve( (int)samples.size() );
	features.resize( (int)samples.size() );

	for( int i=0 ; i<(int)samples.size() ; i++ )
	{
		float* tmp = &(matfeaturesVecDL[i*sampleDim]);
		features[i] = tmp;
	}

	return 0;
}


/**
 * Funcao auxiliar. Recebe o tipo de amostras que deverao ser treinadas, e de posse disso e do numero de features,
 * calcula e armazena os vetores de features para todas as amostras do tipo requerido.
 */
int Volume_Smp::get_ExF_Features( )
{
	// Inserindo as amostras na estrutura de dados correspondente:
	_vSmp_ExF_FeaturesDataGrid3.resize(_vSmp_Dim.x);
	for (int i = 0; i < _vSmp_Dim.x; i++)
	{
		_vSmp_ExF_FeaturesDataGrid3[i].resize(_vSmp_Dim.y);
		for (int j = 0; j < _vSmp_Dim.y; j++)
		{
			_vSmp_ExF_FeaturesDataGrid3[i][j].resize(_vSmp_Dim.z);
			for (int k = 0; k < _vSmp_Dim.z; k++)
				_vSmp_ExF_FeaturesDataGrid3[i][j][k] = -1;
		}
	}

	// Variavel que armazena o numero de amostras desse tipo existentes no volume;
	int numSamples = 0;
	std::vector<float*> samples;
	for( int x = 0; x < _vSmp_Dim.x; x++ )
	{
		for( int y = 0; y < _vSmp_Dim.y; y++ )
		{
			for( int z = 0; z < _vSmp_Dim.z; z++ )
			{
				float centralAmpVoxel = 0;
				float* thisSample = getSample_( x, y, z, centralAmpVoxel, _vSmp_ExF_SmpType );
				if( thisSample != NULL )
				{
					samples.push_back( thisSample );
					_vSmp_ExF_FeaturesDataGrid3[x][y][z] =  numSamples;
					numSamples++;
				}
			}
		}
	}

	// Obtendo o vetor de features a partir do vetor de amostras:
	int ret = -1;

	if(_spuSamplesPosProcessingType == PCA )
	{
		ret = getSPCAFeatures( samples, getSamplesSize(), _vSmp_ExF_nFeatures, _vSmp_ExF_Features );
		if( ret != 0 )
			return -1;
	}
	if( _spuSamplesPosProcessingType == DEEP_LEARNING_MLP )
	{
		ret = getDEEP_LEARNINGFeatures( samples, getSamplesSize(), _vSmp_ExF_nFeatures, _vSmp_ExF_Features );
		if( ret != 0 )
			return -1;
	}

	return 0;
}



/**
 * Retorna a dimensao das amostras que serao retornadas pela instancia.
 */
int Volume_Smp::getSamplesSize( )
{
	return _vSmp_Size;
}


/**
 * Retorna ponteiros para o vetor que contem as posicoes onde existem amostras de todos os tipos possiveis.
 */
const std::vector<std::vector< int > >& Volume_Smp::getSamplesPositions()
{
	return _spuSamplesPositions;
}


/**
 * Retorna ponteiros para o vetor que contem as posicoes 3D onde existem amostras de todos os tipos possiveis.
 */
const std::vector<std::vector<std::vector<std::vector<bool> > > >& Volume_Smp::getSamplesPositions3D()
{
	return _spuSamplesPositions3D;
}


/**
 * Funcao auxiliar. Retorna se uma coordenada e de pico do tipo definido.
 */
bool Volume_Smp::isPeak( int x, int y, int z, Volume_Smp::SamplesType type )
{
	// Verificando se e realmente um voxel de pico:
	float valueAnt;
	float valueAct;
	float valuePos;

	void* retVoxel = NULL;
	_vSmp_GeneralVolumeDataSeismic->getVoxel( x, y, z-1, retVoxel );
	float* retVoxelf = (float*)retVoxel;
	valueAnt = retVoxelf[0];

	retVoxel = NULL;
	_vSmp_GeneralVolumeDataSeismic->getVoxel( x, y, z, retVoxel );
	retVoxelf = (float*)retVoxel;
	valueAct = retVoxelf[0];

	retVoxel = NULL;
	_vSmp_GeneralVolumeDataSeismic->getVoxel( x, y, z+1, retVoxel );
	retVoxelf = (float*)retVoxel;
	valuePos = retVoxelf[0];

	bool ok = false;
	if( type == Volume_Smp::NEGATIVE_PEAKS )
		if( (valueAct<0) && (valueAct<=valueAnt) && (valueAct<=valuePos) )
			ok = true;
	if( type == Volume_Smp::POSITIVE_PEAKS )
		if( (valueAct>0) && (valueAct>=valueAnt) && (valueAct>=valuePos) )
			ok = true;

	if( ok == false )
		std::cout << " " << valueAct << " " << valueAnt << " " << valuePos  << std::endl;

	return ok;
}


/**
 * Funcao auxiliar. Retorna se o vetor recebido eh do tipo definido.
 */
bool Volume_Smp::isOfType( float* vector, int sampleSize, Volume_Smp::SamplesType type )
{
	// Verificando se e realmente um voxel de pico:
	float valueAnt = vector[(sampleSize/2)-1];
	float valueAct = vector[(sampleSize/2)];
	float valuePos = vector[(sampleSize/2)+1];

	bool ok = false;
	if( (type == Volume_Smp::NEGATIVE_PEAKS) || (type == Volume_Smp::NEGATIVE_PEAKS_OP) )
		if( (valueAct<=0) && (valueAct<=valueAnt) && (valueAct<=valuePos) )
			ok = true;
	if( (type == Volume_Smp::POSITIVE_PEAKS) || (type == Volume_Smp::POSITIVE_PEAKS_OP) )
		if( (valueAct>0) && (valueAct>=valueAnt) && (valueAct>=valuePos) )
			ok = true;
	if( type == Volume_Smp::NEG_TO_POS_AMP )
		if( (valueAct<=0) && (valuePos>0) )
			ok = true;
	if( type == Volume_Smp::POS_TO_NEG_AMP )
		if( (valueAct>0) && (valuePos<0) )
			ok = true;

	if( ok == false )
		std::cout << "Amostra nao eh do tipo requerido " << valueAct << " " << valueAnt << " " << valuePos  << std::endl;

	return ok;
}


/**
 * Funcío que testa se as amostras criadas estío corretas.
 */
int Volume_Smp::testSamplesPositions()
{
	std::vector< int > spuSamplesPositions;
	std::vector<std::vector<std::vector<bool> > > spuSamplesPositions3D;

	int numErrors=0;
	// Para cada um dos tipos de amostras:
	for( int i=NEGATIVE_PEAKS ; i<=POS_TO_NEG_AMP ; i++ )
	{
		SamplesType type = (SamplesType)i;
		spuSamplesPositions   = _spuSamplesPositions[(int)type];
		spuSamplesPositions3D = _spuSamplesPositions3D[(int)type];

		// Testando se para cada amostra, ela e pico. Isso permite verificar se as posicoes das amostras estío corretas:
		for( int cont=2 ; cont<(int)spuSamplesPositions.size()-2 ; cont++ )
		{
			int x, y, z;
			int internalPos = spuSamplesPositions[cont];
			if( _vSmp_GeneralVolumeDataSeismic->internalPosToXYZ( x, y, z, internalPos ) == 0 )
			{
				// Caso essa mesma amostra nío exista na matriz 3D, incrementa o numero de erros:
				if( spuSamplesPositions3D[x][y][z] == false )
					numErrors++;

				if( (z<1) || (z>=(_vSmp_Dim.z-1)) )
					continue;

				// Obtendo os três valores, e verificando se a amostra e realmente do tipo correta:
				void* retVoxel = NULL;
				_vSmp_GeneralVolumeDataSeismic->getVoxel( x, y, z-1, retVoxel );
				float* retVoxelf = (float*)retVoxel;
				float valueAnt = retVoxelf[0];

				retVoxel = NULL;
				_vSmp_GeneralVolumeDataSeismic->getVoxel( x, y, z, retVoxel );
				retVoxelf = (float*)retVoxel;
				float valueAct = retVoxelf[0];

				retVoxel = NULL;
				_vSmp_GeneralVolumeDataSeismic->getVoxel( x, y, z+1, retVoxel );
				retVoxelf = (float*)retVoxel;
				float valuePos = retVoxelf[0];

				if( type == NEGATIVE_PEAKS )
				{
					if( (valueAct>0) || (valueAct>valueAnt) || (valueAct>valuePos) )
						numErrors++;
				}
				if( type == POSITIVE_PEAKS )
				{
					if( (valueAct<0) || (valueAct<valueAnt) || (valueAct<valuePos) )
						numErrors++;
				}
				if( type == NEG_TO_POS_AMP )
				{
					if( (valueAct<0) || (valueAnt>0) || (valueAct<valueAnt) )
						numErrors++;
				}
				if( type == POS_TO_NEG_AMP )
				{
					if( (valueAct>0) || (valueAnt<0) || (valueAct>valueAnt) )
						numErrors++;
				}
			}
		}

		// Testando a versao 3D do vetor de coordenadas:
		// Cria uma matriz 3D de booleans, indicando onde existem amostras:
		for(int x=0; x<_vSmp_Dim.x; x++)
		{
			for(int y=0; y<_vSmp_Dim.y; y++)
			{
				for(int z=0; z<_vSmp_Dim.z; z++)
				{
					if( spuSamplesPositions3D[x][y][z] == true )
					{
						if( (z<1) || (z>=(_vSmp_Dim.z-1)) )
							continue;

						// Obtendo os tríªs valores, e verificando se a amostra e realmente do tipo correta:
						void* retVoxel = NULL;
						_vSmp_GeneralVolumeDataSeismic->getVoxel( x, y, z-1, retVoxel );
						float* retVoxelf = (float*)retVoxel;
						float valueAnt = retVoxelf[0];

						retVoxel = NULL;
						_vSmp_GeneralVolumeDataSeismic->getVoxel( x, y, z, retVoxel );
						retVoxelf = (float*)retVoxel;
						float valueAct = retVoxelf[0];

						retVoxel = NULL;
						_vSmp_GeneralVolumeDataSeismic->getVoxel( x, y, z+1, retVoxel );
						retVoxelf = (float*)retVoxel;
						float valuePos = retVoxelf[0];

						if( type == NEGATIVE_PEAKS )
						{
							if( (valueAct>0) || (valueAct>valueAnt) || (valueAct>valuePos) )
								numErrors++;
						}
						if( type == POSITIVE_PEAKS )
						{
							if( (valueAct<0) || (valueAct<valueAnt) || (valueAct<valuePos) )
								numErrors++;
						}
						if( type == NEG_TO_POS_AMP )
						{
							if( (valueAct<0) || (valueAnt>0) || (valueAct<valueAnt) )
								numErrors++;
						}
						if( type == POS_TO_NEG_AMP )
						{
							if( (valueAct>0) || (valueAnt<0) || (valueAct>valueAnt) )
								numErrors++;
						}
					}
				}
			}
		}
	}

	if( numErrors != 0 )
	{
		std::cout << "Vol_Samples::testTrainSamplesPositions(). Erro na criacío das amostras!!!" << std::endl;
		return -1;
	}
	return 0;
}


/**
 * Encontra todos as amostras de treinamento importantes (picos positivos e negativos e pontos de mudanca de sinal).
 * @param inVolume Volume de entrada, a partir do qual gerar as amostras.
 * @param spuSamplesType Tipo das amostras (POS_TO_NEG_AMP, NEG_TO_POS_AMP, POSITIVE_PEAKS, ou NEGATIVE_PEAKS).
 * @param spuSamplesPositions Posicoes de cada uma das amostras do tipo requerido dentro do vetor de voxels do volume.
 * @param spuSamplesPositions3D O mesmo de spuSamplesPositions mas armazenando as coordenada 3D do volume.
 * @return Retorna -1 em caso de erro, ou 0 caso ok.
 */
int Volume_Smp::getTrainSamplesPositions(  SamplesType spuSamplesType,
		std::vector< int >& spuSamplesPositions,
		std::vector<std::vector<std::vector<bool> > >& spuSamplesPositions3D )
{
	// Caso o volume nío exista, retorna erro:
	if( _vSmp_GeneralVolumeDataSeismic == NULL )
		return -1;

	// Cria uma matriz 3D de booleans, indicando onde existem amostras:
	spuSamplesPositions3D.resize(_vSmp_Dim.x);
	for (int i = 0; i < _vSmp_Dim.x; i++)
	{
		spuSamplesPositions3D[i].resize(_vSmp_Dim.y);
		for (int j = 0; j < _vSmp_Dim.y; j++)
		{
			spuSamplesPositions3D[i][j].resize(_vSmp_Dim.z);
			for (int k = 0; k < _vSmp_Dim.z; k++)
				spuSamplesPositions3D[i][j][k] = false;
		}
	}

	// Limpa o vetor:
	spuSamplesPositions.clear();

	// OBTENDO VETORES DE TREINAMENTO, DE AMOSTRAS POSITIVAS E NEGATIVAS. SOMENTE OS PICOS DOS TRACOS SERíO UTILIZADOS:
	// Obtendo dois vetores de treinamento, contendo somente os picos dos tracos:
	std::vector<int> negPeaks;
	std::vector<int> posPeaks;
	// OBTENDO VETORES DE TREINAMENTO, DE POSICOES EM QUE AS APLITUDES MUDAM DE SINAL:
	// Dois vetores de treinamento, com í­ndices onde amplitudes mudam de sinal (positivo => negativo e negativo => positivo):
	std::vector<int> negToPos;
	std::vector<int> posToNeg;

	for( int x=0 ; x<_vSmp_Dim.x ; x++ )
	{
		for( int y=0 ; y<_vSmp_Dim.y ; y++ )
		{
			// Caso as amostras sejam de í­ndices onde amplitudes mudam de sinal (positivo => negativo):
			if( spuSamplesType == POS_TO_NEG_AMP )
			{
				// getTracePeaks retorna todos os picos de amplitudes do traco, em dois vetores: picos negativos e picos positivos:
				_vSmp_GeneralVolumeDataSeismic->getTraceSignChanges( x, y, negToPos, posToNeg );

				// Amostras com í­ndices onde amplitudes mudam de sinal (positivo => negativo):
				for( int z=0 ; z<(int)posToNeg.size() ; z++ )
				{
					// Caso essa seja valida, converte de 3D para coordenada interna do vetor de voxels do volume e armazena a posicío:
					int internalPos = -1;
					if(  _vSmp_GeneralVolumeDataSeismic->xyzToInternalPos( x, y, (_vSmp_Dim.z-posToNeg[z]-1), internalPos ) == 0 )
					{
						spuSamplesPositions.push_back( internalPos );
						spuSamplesPositions3D[x][y][(_vSmp_Dim.z-posToNeg[z]-1)] = true;
					}
				}
				// Limpa os vetores temporarios:
				negToPos.clear( );
				posToNeg.clear( );
			}

			// Caso as amostras sejam de í­ndices onde amplitudes mudam de sinal (negativo => positivo):
			if( spuSamplesType == NEG_TO_POS_AMP )
			{
				// getTracePeaks retorna todos os picos de amplitudes do traco, em dois vetores: picos negativos e picos positivos:
				_vSmp_GeneralVolumeDataSeismic->getTraceSignChanges( x, y, negToPos, posToNeg );

				// Amostras com í­ndices onde amplitudes mudam de sinal (negativo => positivo):
				for( int z=0 ; z<(int)negToPos.size() ; z++ )
				{
					// Caso essa seja valida, converte de 3D para coordenada interna do vetor de voxels do volume e armazena a posicío:
					int internalPos = -1;
					if(  _vSmp_GeneralVolumeDataSeismic->xyzToInternalPos( x, y, (_vSmp_Dim.z-negToPos[z]-1), internalPos ) == 0 )
					{
						spuSamplesPositions.push_back( internalPos );
						spuSamplesPositions3D[x][y][(_vSmp_Dim.z-negToPos[z]-1)] = true;
					}
				}
				// Limpa os vetores temporarios:
				negToPos.clear( );
				posToNeg.clear( );
			}

			// Caso as amostras sejam de picos positivos:
			if( spuSamplesType == POSITIVE_PEAKS )
			{
				// getTracePeaks retorna todos os picos de amplitudes do traco, em dois vetores: picos negativos e picos positivos:
				_vSmp_GeneralVolumeDataSeismic->getTracePeaks( x, y, negPeaks, posPeaks );

				// Amostras de pico positivas:
				for( int z=0 ; z<(int)posPeaks.size() ; z++ )
				{
					// Caso essa seja valida, converte de 3D para coordenada interna do vetor de voxels do volume e armazena a posicío:
					int internalPos = -1;
					if(  _vSmp_GeneralVolumeDataSeismic->xyzToInternalPos( x, y, (_vSmp_Dim.z-posPeaks[z]-1), internalPos ) == 0 )
					{
						spuSamplesPositions.push_back( internalPos );
						spuSamplesPositions3D[x][y][(_vSmp_Dim.z-posPeaks[z]-1)] = true;
					}
				}
				// Limpa os vetores temporarios:
				negPeaks.clear( );
				posPeaks.clear( );
			}

			// Caso as amostras sejam de picos negativos:
			if( spuSamplesType == NEGATIVE_PEAKS )
			{
				// getTracePeaks retorna todos os picos de amplitudes do traco, em dois vetores: picos negativos e picos positivos:
				_vSmp_GeneralVolumeDataSeismic->getTracePeaks( x, y, negPeaks, posPeaks );

				// Amostras de pico negativas:
				for( int z=0 ; z<(int)negPeaks.size() ; z++ )
				{
					// Caso essa seja valida, converte de 3D para coordenada interna do vetor de voxels do volume e armazena a posicío:
					int internalPos = -1;
					if(  _vSmp_GeneralVolumeDataSeismic->xyzToInternalPos( x, y, (_vSmp_Dim.z-negPeaks[z]-1), internalPos ) == 0 )
					{
						spuSamplesPositions.push_back( internalPos );
						spuSamplesPositions3D[x][y][(_vSmp_Dim.z-negPeaks[z]-1)] = true;
					}
				}
				// Limpa os vetores temporarios:
				negPeaks.clear( );
				posPeaks.clear( );
			}
		}
	}
	return 0;
}




/**
 * Funcao auxiliar. Recebe uma coordenada 3d e a direcao de percorrimento no traco. Retorna,
 * a partir da posicao recebida, a proxima posicao em que ocorre um pico (negativo ou positivo), ou
 * -1 em caso de nao haver esse proximo pico.
 */
int Volume_Smp::getNextPeak( int x, int y, int z, int dir )
{
	if( (dir != 1) && (dir != -1 ) )
		return -1;

	int negTp = (int)NEGATIVE_PEAKS;
	int posTp = (int)POSITIVE_PEAKS;

	// Caso a amostra dessa posicao nao seja pico, retorna:
	if( (_spuSamplesPositions3D[posTp][x][y][z] == false) && (_spuSamplesPositions3D[negTp][x][y][z] == false) )
		return -1;

	for(int k=1; k<_vSmp_Dim.z; k++ )
	{
		int pos = z+(k*dir);

		if( pos >= _vSmp_Dim.z )
			return -1;
		if( pos < 0 )
			return -1;

		if( _spuSamplesPositions3D[posTp][x][y][pos] == true )
			return pos;
		if( _spuSamplesPositions3D[negTp][x][y][pos] == true )
			return pos;
	}
	return -1;
}



/**
 * Constroi uma amostra dos tipos NEGATIVE_PEAKS_OP ou POSITIVE_PEAKS_OP.
 */
int Volume_Smp::getOP_Sample( int x, int y, int z, float& centralAmpVoxel, SamplesType type )
{
	if( (type != NEGATIVE_PEAKS_OP) && (type != POSITIVE_PEAKS_OP) )
		return -1;

	SamplesType correctedType;
	if( type == NEGATIVE_PEAKS_OP )
		correctedType = NEGATIVE_PEAKS;
	else // if( type == POSITIVE_PEAKS_OP )
		correctedType = POSITIVE_PEAKS;

	// Caso a amostra dessa posicao nao seja do tipo requerido, retorna:
	if( _spuSamplesPositions3D[(int)correctedType][x][y][z] == false )
		return -1;

	int h_vSmp_Size = _vSmp_Size/2;

	void* retVoxel = NULL;
	_vSmp_GeneralVolumeDataSeismic->getVoxel( x, y, z, retVoxel );
	float* retVoxelf = (float*)retVoxel;
	float value = retVoxelf[0];

	// Pega o valor da amostra central:
	_vSmp_SampleVector[h_vSmp_Size] = value;
	centralAmpVoxel = value;

	// Constroi a amostra e retorna:
	int pos = h_vSmp_Size-1;
	int zNeg = z;
	while( pos >= 0 )
	{
		int zNepTmp = getNextPeak( x, y, zNeg, -1 );
		if( zNepTmp == -1 )
			return -1;

		_vSmp_GeneralVolumeDataSeismic->getVoxel( x, y, zNepTmp, retVoxel );
		retVoxelf = (float*)retVoxel;
		value = retVoxelf[0];
		_vSmp_SampleVector[pos] = value;
		pos--;
		zNeg = zNepTmp;
	}
	pos = h_vSmp_Size+1;
	int zPos = z;
	while( pos < _vSmp_Size )
	{
		int zPosTmp = getNextPeak( x, y, zPos, 1 );
		if( zPosTmp == -1 )
			return -1;

		_vSmp_GeneralVolumeDataSeismic->getVoxel( x, y, zPosTmp, retVoxel );
		retVoxelf = (float*)retVoxel;
		value = retVoxelf[0];
		_vSmp_SampleVector[pos] = value;
		pos++;
		zPos = zPosTmp;
	}
	return 0;
}


/**
 * Construtor.
 */
Volume_Smp::Volume_Smp( )
{
	_vSmp_MinStdDevForTrain = 0.0f;
	_vSmp_GeneralVolumeDataSeismic = NULL;
	_vSmp_Size = -1;
	_vSmp_In_Cross_Time_Steps = Vector3Di( 0, 0, 0 );
	_vSmp_3DCoordsMultiplier =  Vector3Dd( 0, 0, 0 );
	_vSmp_SamplesFeatures = VXL_VALUES_ONLY;

	_spuSamplesPosProcessingType = OTHER_POS_PROS_TYPE;
	_vSmp_ExF_SmpType  = UNDEFINED;
}


/**
 * Construtor.
 * @param vSmp_GeneralVolumeDataSeismic Ponteiro para o volume de entrada.
 * @param vSmp_Size Dimensao que tera cada uma das amostras a serem criadas.
 * @param vSmp_In_Cross_Time_Steps InLineStep, CrossLineStep, e TimeStep.
 * @param vSmp_3DCoordsMultiplier Vetor que identifica o fator de multiplicacao a ser aplicado � s coordenadas 3D da amostra.
 * Caso algum dos valores do vetor seja diferente de 0, a amostra retornada ira conter o(s) valor(es) da coordenada correspondente,
 * multiplicada pelo seu fator e transformada para a mesma escada das amplitudes das amostras, ao final do vetor de amostras.
 */
Volume_Smp::Volume_Smp( DVPVolumeData* vSmp_GeneralVolumeDataSeismic, int vSmp_Size, Vector3Di vSmp_In_Cross_Time_Steps,
		float vSmp_MinStdDevForTrain /*= 0.1f*/, SamplesFeatures vSmp_SamplesFeatures /*=VXL_VALUES_ONLY*/, Vector3Dd vSmp_3DCoordsMultiplier )
{
	_vSmp_MinStdDevForTrain = vSmp_MinStdDevForTrain;
	_vSmp_GeneralVolumeDataSeismic = vSmp_GeneralVolumeDataSeismic;
	_vSmp_Size = vSmp_Size;
	_vSmp_In_Cross_Time_Steps = vSmp_In_Cross_Time_Steps;
	_vSmp_3DCoordsMultiplier = vSmp_3DCoordsMultiplier;
	_vSmp_SamplesFeatures = vSmp_SamplesFeatures;


	_spuSamplesPosProcessingType = PCA;
	_vSmp_ExF_SmpType  = UNDEFINED;
	_vSmp_ExF_nFeatures = 4;


	// Caso os fatores de multiplicacao sejam diferentes de zero, as amostras terao que ser do tipo VXL_VALUES_AND_3D:
	if( (_vSmp_3DCoordsMultiplier.x == 0) && (_vSmp_3DCoordsMultiplier.y == 0) && (_vSmp_3DCoordsMultiplier.z == 0) )
		if( _vSmp_SamplesFeatures != VXL_VALUES_ONLY )
		{
			std::cout << "Volume_Smp::Volume_Smp: amostras VXL_VALUES_ONLY de tipos conflitantes. ERRO!!!" << std::endl;
			return;
		}
	if( (_vSmp_3DCoordsMultiplier.x != 0) || (_vSmp_3DCoordsMultiplier.y != 0) || (_vSmp_3DCoordsMultiplier.z != 0) )
		if( _vSmp_SamplesFeatures != VXL_VALUES_AND_3D )
		{
			std::cout << "Volume_Smp::Volume_Smp: amostras VXL_VALUES_AND_3D de tipos conflitantes. ERRO!!!" << std::endl;
			return;
		}


	// Ponteiro para a instância da classe que contem os dados do volume seja nulo, ou valores invalidos, retorna erro:
	if( ( _vSmp_GeneralVolumeDataSeismic == NULL ) || ( vSmp_Size == -1 ) )
	{
		_vSmp_Size = -1;
		return;
	}

	// Dados do volume sismico:
	// Dados especificos vindos do volume de entrada (dimensoes, tipo de voxel, posicao espacial 3D):	
	SoDataSet::DataType entryVolumeType;	
	Vector3Di entryVolumeDim;
	_vSmp_GeneralVolumeDataSeismic->getDataChar( entryVolumeType, entryVolumeDim );

	_vSmp_Dim.x = entryVolumeDim.x;
	_vSmp_Dim.y = entryVolumeDim.y;
	_vSmp_Dim.z = entryVolumeDim.z;
	//	_vSmp_GeneralVolumeDataSeismic->getDimensions( _vSmp_Dim.x, _vSmp_Dim.y, _vSmp_Dim.z );

	// O vetor que mantera as amostras caso sejam do tipo VXL_VALUES_AND_3D sera alocado de qualquer forma:
	_vSmp_SampleVector = new float[_vSmp_Size + 3];      //< Teremos no maximo 3 coordenadas (3D).

	// Valores de mínimo e maximo das amplitudes dos voxels:
	double minAmp, maxAmp;

	// Em primeiro lugar, encontra min e max dentre todos os voxels:
	_vSmp_GeneralVolumeDataSeismic->getMinMax( minAmp, maxAmp );

	double mean, stdDev;
	_vSmp_GeneralVolumeDataSeismic->statistics( mean, stdDev );

	// Armazena valores referentes ao volume sismico de entrada (min e max de amplitude, media e desvio-padrao):
	double vSmp_minAmp = minAmp;
	double vSmp_maxAmp = maxAmp;
	_vSmp_mean = mean;
	_vSmp_stdDev = stdDev;


	std::cout << "_vSmp_minAmp: " << vSmp_minAmp << std::endl;
	std::cout << "_vSmp_maxAmp: " << vSmp_maxAmp << std::endl;
	std::cout << "_vSmp_mean: " << _vSmp_mean << std::endl;
	std::cout << "_vSmp_stdDev: " << _vSmp_stdDev << std::endl;

	// Caso existam amostras 3D que deverao ser adicionadas � s amostras:
	// Para o eixo X:
	if( _vSmp_3DCoordsMultiplier.x != 0)
	{
		// Encontrando o fator de multiplicacao utilizado para escalonar essa coordenada numa amostra qualquer:
		double maxScale = ((maxAmp - minAmp) * _vSmp_3DCoordsMultiplier.x);
		_vSmp_3DCoordsMinMaxScale.x =  maxScale / _vSmp_Dim.x;
	}
	// Para o eixo Y:
	if( _vSmp_3DCoordsMultiplier.y != 0)
	{
		// Encontrando o fator de multiplicacao utilizado para escalonar essa coordenada numa amostra qualquer:
		double maxScale = ((maxAmp - minAmp) * _vSmp_3DCoordsMultiplier.y);
		_vSmp_3DCoordsMinMaxScale.y =  maxScale / _vSmp_Dim.y;
	}
	// Para o eixo Z:
	if( _vSmp_3DCoordsMultiplier.z != 0)
	{
		// Encontrando o fator de multiplicacao utilizado para escalonar essa coordenada numa amostra qualquer:
		double maxScale = ((maxAmp - minAmp) * _vSmp_3DCoordsMultiplier.z);
		_vSmp_3DCoordsMinMaxScale.z =  maxScale / _vSmp_Dim.z;
	}


	// Criando cada uma das versoes de samplesPositions:
	_spuSamplesPositions.reserve( 4 );
	_spuSamplesPositions.resize( 4 );
	// Versao 3D:
	_spuSamplesPositions3D.reserve( 4 );
	_spuSamplesPositions3D.resize( 4 );

	// Para NEGATIVE_PEAKS:
//	if( getTrainSamplesPositions( NEGATIVE_PEAKS,
//			_spuSamplesPositions[(int)NEGATIVE_PEAKS], _spuSamplesPositions3D[(int)NEGATIVE_PEAKS] ) == -1 )
//		std::cout << "Volume_Smp::Volume_Smp(). Erro na criacao das amostras!!!" << std::endl;
//	// Para POSITIVE_PEAKS:
//	if( getTrainSamplesPositions( POSITIVE_PEAKS,
//			_spuSamplesPositions[(int)POSITIVE_PEAKS], _spuSamplesPositions3D[(int)POSITIVE_PEAKS] ) == -1 )
//		std::cout << "Volume_Smp::Volume_Smp(). Erro na criacao das amostras!!!" << std::endl;
	// Para NEG_TO_POS_AMP:
//	if( getTrainSamplesPositions( NEG_TO_POS_AMP,
//			_spuSamplesPositions[(int)NEG_TO_POS_AMP], _spuSamplesPositions3D[(int)NEG_TO_POS_AMP] ) == -1 )
//		std::cout << "Volume_Smp::Volume_Smp(). Erro na criacao das amostras!!!" << std::endl;
//	// Para POS_TO_NEG_AMP:
//	if( getTrainSamplesPositions( POS_TO_NEG_AMP,
//			_spuSamplesPositions[(int)POS_TO_NEG_AMP], _spuSamplesPositions3D[(int)POS_TO_NEG_AMP] ) == -1 )
//		std::cout << "Volume_Smp::Volume_Smp(). Erro na criacao das amostras!!!" << std::endl;

	// Caso seja necess�rio, procede com o processo de extra��o de features:
	if( (_spuSamplesPosProcessingType == PCA ) || (_spuSamplesPosProcessingType == DEEP_LEARNING_MLP ) )
		get_ExF_Features();
}


/**
 * Destrutor.
 */
Volume_Smp::~Volume_Smp( )
{
	delete[] _vSmp_SampleVector;
}


/**
 * Retorna o ponteiro para uma amostra, uma vez recebida suas coordenadas.
 * @param x Coordenada x da coordenada central da amostra.
 * @param y Coordenada y da coordenada central da amostra.
 * @param z Coordenada z da coordenada central da amostra.
 * @return Retora NULL caso a posicao da amostra seja invalida, ou um ponteiro para o voxel INICIAL da amostra.
 */
float* Volume_Smp::getSample_( int x, int y, int z, float& centralAmpVoxel, SamplesType samplesType )
{
	// Caso a coordenada nao seja valida, retorna NULL:
	bool isValid = _vSmp_GeneralVolumeDataSeismic->isValidCoord( x, y, z );
	if( isValid == false )
		return NULL;

	// Coordenadas invalidas:
	if( z < (_vSmp_Size / 2 + 1) )
		return NULL;
	if( z + ((_vSmp_Size / 2) + 1) >= _vSmp_Dim.z )
		return NULL;

	// Caso a amostra seja do tipo NEGATIVE_PEAKS_OP ou POSITIVE_PEAKS_OP:
	if( (samplesType == NEGATIVE_PEAKS_OP) || (samplesType == POSITIVE_PEAKS_OP) )
	{
		int ret = getOP_Sample( x, y, (_vSmp_Dim.z-z-1), centralAmpVoxel, samplesType );
		if( ret == -1 )
			return NULL;
		if( ret == 0 )
			return _vSmp_SampleVector;
	}

	// Caso estejamos trabalhando com os outros tipos de amostras:
	if( samplesType != UNDEFINED )
	{
		int type = (int)samplesType;
		int internalPos = -1;
		// Caso a posicao da amostra dada por (x, y, z) seja invalida, retorna NULL:
		if( _vSmp_GeneralVolumeDataSeismic->xyzToInternalPos( x, y, (_vSmp_Dim.z-z-1), internalPos ) == -1 )
			return NULL;

		// Pega o numero de amostras do tipo requerido:
		int numSamples = (_spuSamplesPositions[type]).size()-2;
		if( internalPos > _spuSamplesPositions[type][numSamples] )
			return NULL;

		// Caso a amostra na posicao (x, y, z) nao seja do tipo desejado, retorna NULL:
		if( _spuSamplesPositions3D[type][x][y][(_vSmp_Dim.z-z-1)] == false )
			return NULL;
	}

	// Obtem o numero traco:
	int traceNumber;
	if( _vSmp_GeneralVolumeDataSeismic->xyToTraceNumber( x, y, traceNumber ) == -1 )
		return NULL;

	void* traceData = NULL;
	if( _vSmp_GeneralVolumeDataSeismic->getSegyTraceData( traceNumber, traceData ) == -1 )
		return NULL;


	int local_vSmp_Size = _vSmp_Size;
	if( _vSmp_SamplesFeatures == VXL_VALUES_AND_3D )
		local_vSmp_Size = local_vSmp_Size-4;    //< Era pra ser 3, 4 e para manter as amostras com tamanho i�mpar. Assim, adicionamos 0.0 no ultimo valor.

	float* traceDataf = (float*) traceData;
	float* sampleInit = &traceDataf[z];
	centralAmpVoxel = sampleInit[0];
	sampleInit -= local_vSmp_Size / 2;

	// Caso estejamos no modo que retorna somente valores de voxels:
	if( _vSmp_SamplesFeatures == VXL_VALUES_ONLY )
		return sampleInit;


	// Caso em que estamos no modo em que retornamos tambem coordenadas 3D das amostras escalonadas:
	if( _vSmp_SamplesFeatures == VXL_VALUES_AND_3D )
	{
		// Primeiramente copia as amostras dos voxels para o vetor interno a ser retornado:
		int i=0;
		for(  ; i<local_vSmp_Size ; i++ )
			_vSmp_SampleVector[i] = sampleInit[i];
		// Feito isso, e preciso incluir as ultimas coordenadas:
		// Para o eixo X:
		if( _vSmp_3DCoordsMultiplier.x != 0)
		{
			// Inclui o valor da coordenada X:
			_vSmp_SampleVector[i] = x * _vSmp_3DCoordsMinMaxScale.x;
			i++;
		}
		// Para o eixo Y:
		if( _vSmp_3DCoordsMultiplier.y != 0)
		{
			// Inclui o valor da coordenada Y:
			_vSmp_SampleVector[i] = y * _vSmp_3DCoordsMinMaxScale.y;
			i++;
		}
		// Para o eixo Z:
		if( _vSmp_3DCoordsMultiplier.z != 0)
		{
			// Inclui o valor da coordenada Z:
			_vSmp_SampleVector[i] = z * _vSmp_3DCoordsMinMaxScale.z;
		}
		i++;
		_vSmp_SampleVector[i] = 0.0;

		// Retorna o vetor (nao existe persistência nesse caso:
		return _vSmp_SampleVector;
	}
	// Caso contrario, retorna erro:
	return NULL;
}


/**
 * Retorna o ponteiro para uma amostra, uma vez recebida suas coordenadas.
 * @param x Coordenada x da coordenada central da amostra.
 * @param y Coordenada y da coordenada central da amostra.
 * @param z Coordenada z da coordenada central da amostra.
 * @return Retora NULL caso a posicao da amostra seja invalida, ou um ponteiro para o voxel INICIAL da amostra.
 */
float* Volume_Smp::getSampleCenter( int x, int y, int z )
{
	// Caso a coordenada nao seja valida, retorna NULL:
	bool isValid = _vSmp_GeneralVolumeDataSeismic->isValidCoord( x, y, z );
	if( isValid == false )
		return NULL;

	// Coordenadas invalidas:
	if( z < (_vSmp_Size / 2 + 1) )
		return NULL;
	if( z + ((_vSmp_Size / 2) + 1) >= _vSmp_Dim.z )
		return NULL;

	// Caso contrario, obtem o numero traco:
	int traceNumber;
	if( _vSmp_GeneralVolumeDataSeismic->xyToTraceNumber( x, y, traceNumber ) == -1 )
		return NULL;

	void* traceData = NULL;
	if( _vSmp_GeneralVolumeDataSeismic->getSegyTraceData( traceNumber, traceData ) == -1 )
		return NULL;

	float* traceDataf = (float*) traceData;
	float* sampleInit = &traceDataf[z];

	return sampleInit;
}


/**
 * Retorna o ponteiro para uma amostra escolhida randomicamente, positiva ou negativa.
 * @return Retora NULL caso os dados da classe sejam invalidos, ou um ponteiro para o voxel INICIAL da amostra.
 */
float* Volume_Smp::getRandomSample_( int xMin /*=-1*/, int yMin /*=-1*/ , int xMax /*=-1*/, int yMax /*=-1*/, float peaksProb /*=0.4f*/, SamplesType samplesType /*=UNDEFINED*/ )
{
	// Coordenadas mí­nimas da vizinhanca:
	int xm = (xMin== -1)? 0 : xMin;
	int ym = (yMin== -1)? 0 : yMin;
	// Coordenadas maximas da vizinhanca:
	int xM = (xMax== -1)? (_vSmp_Dim.x-1) : xMax;
	int yM = (yMax== -1)? (_vSmp_Dim.y-1) : yMax;

	while( 1 )
	{
		int x, y, z;
		float centralAmpVoxel;
		float* sampleInit = NULL;

		// Caso treinando sem aumentar a probabilidade de obter picos, ou a probabilidade de nao tenha sido atingida:
		if( samplesType == UNDEFINED )
		{
			// Gerando a posicao da amostra aleatoriamente:
			x = RandInt( xm, xM );
			y = RandInt( ym, yM );
			z = RandInt( (_vSmp_Size / 2 + 1), (_vSmp_Dim.z - ((_vSmp_Size / 2) -1)) );

			sampleInit = getSample_( x, y, z, centralAmpVoxel, samplesType );
			if( sampleInit == NULL )
				continue;
			else
				return sampleInit;

		}
		else    //< Caso contrario, teremos que encontrar um pico positivo ou negativo para treinar:
		{
			int type = (int)samplesType;

			if( samplesType == NEGATIVE_PEAKS_OP )
				type = (int)NEGATIVE_PEAKS;
			if( samplesType == POSITIVE_PEAKS_OP )
				type = (int)POSITIVE_PEAKS;

			int numSamples = (_spuSamplesPositions[type]).size()-2;
			int thisSampleIndex = Rounded((RandFloat() * numSamples));
			int internalPos = _spuSamplesPositions[type][thisSampleIndex];
			if( _vSmp_GeneralVolumeDataSeismic->internalPosToXYZ( x, y, z, internalPos ) == -1 )
				continue;

			// Caso a coordenada z da amostra nao esteja nos limites por tamanho das amostras, continua:
			if( (z < (_vSmp_Size / 2 + 1)) || (z >= (_vSmp_Dim.z - ((_vSmp_Size / 2) + 1))) )
				continue;

			// Caso nao tenha sido possivel encontrar a amostra, continua:
			sampleInit = getSample_( x, y, z, centralAmpVoxel, samplesType );
			if( sampleInit == NULL )
				continue;
		}

		if( _vSmp_In_Cross_Time_Steps.x > 1 )
		{
			x = x / _vSmp_In_Cross_Time_Steps.x;  //< Divisao de inteiros (joga o resto fora).
			x = x * _vSmp_In_Cross_Time_Steps.x;
		}

		if( _vSmp_In_Cross_Time_Steps.y > 1 )
		{
			y = y / _vSmp_In_Cross_Time_Steps.y;  //< Divisao de inteiros (joga o resto fora).
			y = y * _vSmp_In_Cross_Time_Steps.y;
		}

		if( _vSmp_In_Cross_Time_Steps.z > 1 )
		{
			z = z / _vSmp_In_Cross_Time_Steps.z;  //< Divisao de inteiros (joga o resto fora).
			z = z * _vSmp_In_Cross_Time_Steps.z;
		}

		// Confere novamente se a amostra eh do tipo requerido:
		if( (sampleInit != NULL) && ( isOfType( sampleInit, _vSmp_Size, samplesType ) == false ) )
			return NULL;

		if( (samplesType == NEG_TO_POS_AMP) || (samplesType == POS_TO_NEG_AMP) )
			return sampleInit;

		if( (samplesType == NEGATIVE_PEAKS) || (samplesType == POSITIVE_PEAKS) )
		{
			if( (sampleInit != NULL) && ( fabs(centralAmpVoxel) > (_vSmp_mean + (_vSmp_MinStdDevForTrain * _vSmp_stdDev)) ) )
				return sampleInit;
		}
		if( (samplesType == NEGATIVE_PEAKS_OP) || (samplesType == POSITIVE_PEAKS_OP) )
		{
			if( (sampleInit != NULL) && ( fabs(centralAmpVoxel) > (_vSmp_mean + (_vSmp_MinStdDevForTrain * _vSmp_stdDev)) ) )
				return sampleInit;
		}
	}
	return NULL;
}


/**
 * Retorna o ponteiro para uma amostra, uma vez recebida suas coordenadas.
 * @param x Coordenada x da coordenada central da amostra.
 * @param y Coordenada y da coordenada central da amostra.
 * @param z Coordenada z da coordenada central da amostra.
 * @return Retora NULL caso a posicao da amostra seja invalida, ou um ponteiro para o voxel INICIAL da amostra.
 */
float* Volume_Smp::getSample( int x, int y, int z, float& centralAmpVoxel, SamplesType samplesType )
{
	if(_spuSamplesPosProcessingType == PCA )
	{
		float* thisSample = getSample_( x, y, z, centralAmpVoxel, samplesType );
		if( thisSample != NULL )
			return _vSmp_ExF_Features[_vSmp_ExF_FeaturesDataGrid3[x][y][z]];
	}
	if(_spuSamplesPosProcessingType == DEEP_LEARNING_MLP )
	{
		float* thisSample = getSample_( x, y, z, centralAmpVoxel, samplesType );
		if( thisSample != NULL )
			return _vSmp_ExF_Features[_vSmp_ExF_FeaturesDataGrid3[x][y][z]];
	}

	return getSample_( x, y, z, centralAmpVoxel, samplesType );
}


/**
 * Retorna o ponteiro para uma amostra escolhida randomicamente, positiva ou negativa.
 * @return Retora NULL caso os dados da classe sejam invalidos, ou um ponteiro para o voxel INICIAL da amostra.
 */
float* Volume_Smp::getRandomSample( int xMin, int yMin, int xMax, int yMax, float peaksProb, SamplesType samplesType )
{
	if(_spuSamplesPosProcessingType == PCA )
	{
		int numSamples = (int)(_vSmp_ExF_Features.size())-2;
		int thisSampleIndex = Rounded((RandFloat() * numSamples));
		return _vSmp_ExF_Features[thisSampleIndex];
	}
	if(_spuSamplesPosProcessingType == DEEP_LEARNING_MLP )
	{
		int numSamples = (int)(_vSmp_ExF_Features.size())-2;
		int thisSampleIndex = Rounded((RandFloat() * numSamples));
		return _vSmp_ExF_Features[thisSampleIndex];
	}

	return getRandomSample_( xMin, yMin, xMax, yMax, peaksProb, samplesType );
}

}
