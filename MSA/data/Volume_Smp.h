#ifndef VOL_SAMPLES_H
#define VOL_SAMPLES_H

#include "DVPVolumeData.h"

#include "../../math/Vector.h"

#include "SPCA.h"
#include "DataGrid3.h"

namespace MSA
{

/**
  * Essa classe se propoe a ser uma forma simples de criacao das amostras de entrada. Para isso,
  * a classe recebe como entrada um ponteiro para o volume de entrada (somente voxels do tipo float sao permitidos).
  * A classe tambem recebe como entrada a dimensao que tera cada uma das amostras a serem criadas.
  * Caso as amostras sendo criadas sejam do tipo VXL_VALUES_AND_3D, N�O
  */
class Volume_Smp
{
    
public:

    // Enumerator que define se as amostras terao completacao:
    enum SamplesFeatures
    {
        VXL_VALUES_ONLY=0,		//< Amostras contendo somente valores dos voxels.
        VXL_VALUES_AND_3D		//< Amostras contendo valores dos voxels e as coordenadas 3D.
    };

    /**
    * Enumerator que define o tipo de amostra.
    * As amostras ficam, nos tracos, na seguinte ordem:
    *      ...
    *  NEGATIVE_PEAKS
    *
    *  NEG_TO_POS_AMP ---\
    *                     Podem ser a mesma coordenada.
    *  POSITIVE_PEAKS ---/
    *
    *  POS_TO_NEG_AMP ---\
    *                     Podem ser a mesma coordenada.
    *  NEGATIVE_PEAKS ---/
    *     ...
    */
    enum SamplesType
    {
        NEGATIVE_PEAKS=0,   //< Amostras contendo todos os valores do traco, mas criadas somente nos pontos de picos negativos.
        POSITIVE_PEAKS,     //< Amostras contendo todos os valores do traco, mas criadas somente nos pontos de picos positivos.
        NEG_TO_POS_AMP,     //< Posicoes de mudancas de valores de amplitudes: voxel anterior negativo -> voxel atual positivo.
        POS_TO_NEG_AMP,     //< Posicoes de mudancas de valores de amplitudes: voxel anterior positivo -> voxel atual negativo.
        NEGATIVE_PEAKS_OP,	//< Amostras formadas somente de maximos e minimos locais, criadas somente para os pontos de picos negativos.
        POSITIVE_PEAKS_OP,	//< Amostras formadas somente de maximos e minimos locais, criadas somente para os pontos de picos positivos.
        UNDEFINED
    };



    /**
    * Enumerator que define o tipo de pós-processamento de amostra.
    *      ...
    *  TRACE_SHAPES
    *  PCA
    *  DEEP_LEARNING_MLP.
    *     ...
    */
    enum PosProcessingType
    {
    	TRACE_SHAPES=0,		//< Amostras contendo todos os valores do traco.
    	PCA,				//< Amostras contendo features obtidas atraves de PCA a partir dos valores do traco.
    	DEEP_LEARNING_MLP,	//< Amostras contendo features obtidas atraves de DEEP_LEARNING_MLP a partir dos valores do traco.
		OTHER_POS_PROS_TYPE
    };


    int _vSmp_Size;				//< Dimensao das amostras, COM A INCLUSAO DAS COORDENADAS 3D caso o tipo de amostras seja VXL_VALUES_AND_3D.
    int _vSmp_OP_Size;			//< Borda minima nos tracos, so utilizado para amostras tipo NEGATIVE_PEAKS_OP ou POSITIVE_PEAKS_OP.

    Vector3Dd _vSmp_3DCoordsMultiplier;			//< Multiplicadores de cada coordenada 3D.
    Vector3Dd _vSmp_3DCoordsMinMaxScale;			//< Valores de minimo e maximo a ser aplicado nas coordenadas 3D.
    SamplesFeatures _vSmp_SamplesFeatures;		//< Define o tipo de amostras.
    float* _vSmp_SampleVector;					//< Vetor auxiliar que devera conter as amostras.

    float _vSmp_MinStdDevForTrain;				//< Define o minimo de amplitude de um pico para que entre no treinamento.
    DVPVolumeData* _vSmp_GeneralVolumeDataSeismic;

    // Demais informacoes relacionadas ao volume sismico de entrada (min e max de amplitudes, media e desvio-padrao):
    double _vSmp_mean, _vSmp_stdDev;

    // Dados do volume sismico:
    Vector3Di _vSmp_Dim;

    // InLineStep, CrossLineStep, e TimeStep:
    Vector3Di _vSmp_In_Cross_Time_Steps;

    // Esse vetor contera todas as posicoes de _vSmp_GeneralVolumeDataSeismic onde existem amostras do tipo definido:
    std::vector<std::vector< int > > _spuSamplesPositions;
    // O mesmo, porem contendo as posicoes em 3D, permitindo consulta rapida:
    std::vector<std::vector<std::vector<std::vector<bool> > > > _spuSamplesPositions3D;

    PosProcessingType _spuSamplesPosProcessingType;

    // Vari�veis e fun��es utilizadas somente para o caso de utiliza��o dos m�todos de extra��o de features:
    SamplesType _vSmp_ExF_SmpType;
	int _vSmp_ExF_nFeatures;																									//< Numero de features a serem obtidas.
	std::vector<float*> _vSmp_ExF_Features;																		//< Contem todos os vetores de caracteristicas obtidos pelo PCA.
	std::vector<std::vector<std::vector<int> > > _vSmp_ExF_FeaturesDataGrid3;		//< Vetores de _vSmp_ExFeatures_Features posicionados onde existem amostras do tipo basico.


	int getSPCAFeatures( std::vector<float*> samples, int sampleDim, int nFeatures, std::vector<float*>& features );

	int getDEEP_LEARNINGFeatures( std::vector<float*> samples, int sampleDim, int nFeatures, std::vector<float*>& features );

	int get_ExF_Features( );


    /**
     * Retorna a dimensao das amostras que serao retornadas pela instancia.
     */
    int getSamplesSize();


    /**
      * Retorna ponteiros para o vetor que contem as posicoes onde existem amostras de todos os tipos possiveis.
      */
    const std::vector<std::vector< int > >& getSamplesPositions();


    /**
      * Retorna ponteiros para o vetor que contem as posicoes 3D onde existem amostras de todos os tipos possiveis.
      */
    const std::vector<std::vector<std::vector<std::vector<bool> > > >& getSamplesPositions3D();



    /**
    * Funcao auxiliar. Retorna se uma coordenada e de pico do tipo definido.
    */
    bool isPeak( int x, int y, int z, Volume_Smp::SamplesType type );


    /**
    * Funcao auxiliar. Retorna se o vetor recebido eh do tipo definido.
    */
    bool isOfType( float* vector, int sampleSize, Volume_Smp::SamplesType type );



    /**
    * Funcío que testa se as amostras criadas estío corretas.
    */
    int testSamplesPositions();


    /**
  * Encontra todos as amostras de treinamento importantes (picos positivos e negativos e pontos de mudanca de sinal).
  * @param inVolume Volume de entrada, a partir do qual gerar as amostras.
  * @param spuSamplesType Tipo das amostras (POS_TO_NEG_AMP, NEG_TO_POS_AMP, POSITIVE_PEAKS, ou NEGATIVE_PEAKS).
  * @param spuSamplesPositions Posicoes de cada uma das amostras do tipo requerido dentro do vetor de voxels do volume.
  * @param spuSamplesPositions3D O mesmo de spuSamplesPositions mas armazenando as coordenada 3D do volume.
  * @return Retorna -1 em caso de erro, ou 0 caso ok.
  */
    int getTrainSamplesPositions(  SamplesType spuSamplesType,
                                   std::vector< int >& spuSamplesPositions,
                                   std::vector<std::vector<std::vector<bool> > >& spuSamplesPositions3D );



    /**
     * Funcao auxiliar. Recebe uma coordenada 3d e a direcao de percorrimento no traco. Retorna,
     * a partir da posicao recebida, a proxima posicao em que ocorre um pico (negativo ou positivo), ou
     * -1 em caso de nao haver esse proximo pico.
     */
    int getNextPeak( int x, int y, int z, int dir );



    /**
     * Constroi uma amostra dos tipos NEGATIVE_PEAKS_OP ou POSITIVE_PEAKS_OP.
     */
    int getOP_Sample( int x, int y, int z, float& centralAmpVoxel, SamplesType type );


    /**
    * Construtor.
    */
    Volume_Smp( );


    /**
    * Construtor.
    * @param vSmp_GeneralVolumeDataSeismic Ponteiro para o volume de entrada.
    * @param vSmp_Size Dimensao que tera cada uma das amostras a serem criadas.
    * @param vSmp_In_Cross_Time_Steps InLineStep, CrossLineStep, e TimeStep.
    * @param vSmp_3DCoordsMultiplier Vetor que identifica o fator de multiplicacao a ser aplicado � s coordenadas 3D da amostra.
    * Caso algum dos valores do vetor seja diferente de 0, a amostra retornada ira conter o(s) valor(es) da coordenada correspondente,
    * multiplicada pelo seu fator e transformada para a mesma escada das amplitudes das amostras, ao final do vetor de amostras.
    */
    Volume_Smp( DVPVolumeData* vSmp_GeneralVolumeDataSeismic, int vSmp_Size, Vector3Di vSmp_In_Cross_Time_Steps = Vector3Di(0, 0, 0) ,
    		float vSmp_MinStdDevForTrain = 0.1f, SamplesFeatures vSmp_SamplesFeatures=VXL_VALUES_ONLY, Vector3Dd vSmp_3DCoordsMultiplier = Vector3Dd(0, 0, 0) );


    /**
    * Destrutor.
    */

    ~Volume_Smp( );
    
    
    /**
    * Retorna o ponteiro para uma amostra, uma vez recebida suas coordenadas.
    * @param x Coordenada x da coordenada central da amostra.
    * @param y Coordenada y da coordenada central da amostra.
    * @param z Coordenada z da coordenada central da amostra.
    * @return Retora NULL caso a posicao da amostra seja invalida, ou um ponteiro para o voxel INICIAL da amostra.
    */
    float* getSample_( int x, int y, int z, float& centralAmpVoxel, SamplesType samplesType=UNDEFINED );


    /**
    * Retorna o ponteiro para uma amostra, uma vez recebida suas coordenadas.
    * @param x Coordenada x da coordenada central da amostra.
    * @param y Coordenada y da coordenada central da amostra.
    * @param z Coordenada z da coordenada central da amostra.
    * @return Retora NULL caso a posicao da amostra seja invalida, ou um ponteiro para o voxel INICIAL da amostra.
    */
    float* getSample( int x, int y, int z, float& centralAmpVoxel, SamplesType samplesType=UNDEFINED );


    /**
    * Retorna o ponteiro para uma amostra escolhida randomicamente, positiva ou negativa.
    * @return Retora NULL caso os dados da classe sejam invalidos, ou um ponteiro para o voxel INICIAL da amostra.
    */
    float* getRandomSample_( int xMin=-1, int yMin=-1, int xMax=-1, int yMax=-1, float peaksProb=0.4f, SamplesType samplesType=UNDEFINED );


    /**
    * Retorna o ponteiro para uma amostra escolhida randomicamente, positiva ou negativa.
    * @return Retora NULL caso os dados da classe sejam invalidos, ou um ponteiro para o voxel INICIAL da amostra.
    */
    float* getRandomSample( int xMin=-1, int yMin=-1, int xMax=-1, int yMax=-1, float peaksProb=0.4f, SamplesType samplesType=UNDEFINED );


    /**
    * Retorna o ponteiro para uma amostra, uma vez recebida suas coordenadas.
    * @param x Coordenada x da coordenada central da amostra.
    * @param y Coordenada y da coordenada central da amostra.
    * @param z Coordenada z da coordenada central da amostra.
    * @return Retora NULL caso a posicao da amostra seja invalida, ou um ponteiro para o voxel INICIAL da amostra.
    */
    float* getSampleCenter( int x, int y, int z );


};

}

#endif // VOL_SAMPLES_H
