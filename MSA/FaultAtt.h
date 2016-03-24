/*
 * FaultAtt.h
 *
 *  Created on: May 12, 2014
 *      Author: aurelio
 */

#ifndef FAULTATT_H_
#define FAULTATT_H_

#include <stdio.h>

//#include "SeismicProcessingControler.h"

#include "v3o2_sdk.h"
#include "SDKVolume.h"
#include "../math/Vector.h"

namespace MSA
{
    
class DVPVolumeData;

class FaultAtt
{
    
public:

    /**
     * Construtor recebendo o SDK do v3o2 e um volume.
     * @param sdk SDK do v3o2.
     * @param volume Volume de entrada para o método.
     */
    FaultAtt( v3o2_SDK* sdk, SDKVolume* volume );

    /**
     * Destrutor padrão.
     */
    virtual ~FaultAtt();

    /**
     * Calcula o atributo MSA.
     */
    void MSA_Calc( );

    /**
     * Recebe dois volumes, sendo que o primeiro e um sub-volume do segundo. Recebe tambem uma
     * coordenada de mapa de um voxel do sub-volume. Pega o valor do voxel do sub-volume e o escreve no volume
     * de saida.
     */
    bool copyVoxelsFromSubVolume( Vector3Di mapCoord, DVPVolumeData*& entrySubVolume, DVPVolumeData*& outPutVolume );
    

    /**
     * Funcao que permite encontrar atributos de falha a partir de volumes sismicos de entrada, utilizando para isso
     * a subdivisao do volume original em diferentes volumes de entrada.
     */
    bool createAtt( DVPVolumeData*& _spuEntryVolume, DVPVolumeData*& _spuFaultVolume );

    // Parametros de treinamento do GNG:
    int fap_Lambda;
    
    float fap_Eb;
    
    float fap_En;
    
    float fap_Alpha;
    
    float fap_D;
    
    int fap_Amax;

    /**
     * Dimensao vertical das amostras.
     */
    int fap_SamplesSize;
        
    /**
     * Numero de neuronios criados.
     */
    float fap_NneuronsMulti;

    /**
     * Valor minimo de amplitude para que uma amostra seja treinada.
     */
    float fap_MinStdDevForTrain;
        
    /**
     * Numero de divisoes verticais do volume.
     */
    int fap_DivZ;					
        
    /**
     * Define o tamanho das amostras no volume de clusters.
     */
    int fap_Window;
    
    /**
     * Define da vizinhanca horizontal.
     */
    int fap_HDist;

    /**
     * Volume de entrada (original) vindo do v3o2.
     */
    SDKVolume* _sdkInputVolume;
    
    /**
     * Volume de saída (com os atributos calculados) que será pendurado na árvore do v3o2.
     */
    SDKVolume* _sdkAttributeVolume;
    
    /**
     * Arquivo de entrada no formato DVP.
     */
    DVPVolumeData* _dvpInputVolume;
    
    /**
     * Arquivo de saída no formato DVP.
     */
    DVPVolumeData* _dvpAttributeVolume;

protected:

    /**
     * SDK do v3o2.
     */
    v3o2_SDK* _sdk;
    
private:
    
    /**
     * Imprime as informações de um volume SDK.
     */
    void printVolumeInfo( SDKVolume* volume );
};

}
#endif /* FAULTATT_H_ */
