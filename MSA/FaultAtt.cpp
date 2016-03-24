/*
 * FaultAtt.cpp
 *
 *  Created on: May 12, 2014
 *      Author: aurelio
 */

#include "FaultAtt.h"
#include "data/dipVolume.h"
#include "data/DVPVolumeData.h"
#include "processing/SeismicProcessingController.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdlib.h>
#include <cstdlib>


namespace MSA
{

FaultAtt::FaultAtt( v3o2_SDK* sdk, SDKVolume* volume ) : _sdk( sdk )
{
    if( volume == NULL )
    {
        printf("\n*** FaultAtt: não existe volume carregado. ***\n");
        return;
    }

    _sdkInputVolume = volume;
    _dvpInputVolume = 0;
    _dvpAttributeVolume = 0;
}


FaultAtt::~FaultAtt()
{
    
}


void FaultAtt::printVolumeInfo( SDKVolume* volume )
{
    std::cout << std::endl << "*** Dados do Volume: " << volume->nome() << " ***" << std::endl;
    
    Vector3Di nInline, nCrossline;
    Vector3Df nTime;
    volume->grade( nInline.x, nInline.y, nInline.z, nCrossline.x, nCrossline.y, nCrossline.z, nTime.x, nTime.y, nTime.z );
    
    int numInline, numCrosslines, numTime;
    numInline = numCrosslines = numTime = 0;
    volume->getDimensoes( numInline, numCrosslines, numTime );
    std::cout << "Inline( inicial, final, incremento) - Num: (" << nInline.x << ", " <<  nInline.y << ", " << nInline.z << ") - " << numInline << std::endl;
    std::cout << "Crossline( inicial, final, incremento): (" << nCrossline.x << ", " <<  nCrossline.y << ", " << nCrossline.z << ") - " << numCrosslines << std::endl;
    std::cout << "Time( inicial, final, incremento): (" << nTime.x << ", " <<  nTime.y << ", " << nTime.z << ") - " << numTime << std::endl;

    Vector3Dd p1, p2, p3;
    volume->p1( p1.x, p1.y );
    volume->p2( p2.x, p2.y );
    volume->p3( p3.x, p3.y );
    std::cout << "P1("<< std::setprecision(15) << p1.x << ", " <<  p1.y << ")" << std::endl;
    std::cout << "P2("<< std::setprecision(15) << p2.x << ", " <<  p2.y << ")" << std::endl;
    std::cout << "P3("<< std::setprecision(15) << p3.x << ", " <<  p3.y << ")" << std::endl;
}


void FaultAtt::MSA_Calc( )
{
    fap_Lambda = 45000; // 45000;
    fap_Eb = 0.015f;
    fap_En = 0.0015f;
    fap_Alpha = 0.5f;
    fap_D = 0.995f;
    fap_Amax = 1750;
    fap_SamplesSize = 35;
    fap_NneuronsMulti = 0.5f;
    fap_MinStdDevForTrain = 0.0f;
    fap_DivZ = 1;
    fap_Window = 1;
    fap_HDist = 2;

    // Cria um volume DVP equivalente ao volume de entrada recebido:
    printVolumeInfo( _sdkInputVolume );
    _dvpInputVolume = new DVPVolumeData( _sdkInputVolume );

    // Dimensoes do volume de entrada:
    int ilFrom, ilTo, ilStep;
    int clFrom, clTo, clStep;
    float tFrom, tTo, tStep;
    _sdkInputVolume->grade( ilFrom, ilTo, ilStep, clFrom, clTo, clStep, tFrom, tTo, tStep );
    double p1x, p1y;
    _sdkInputVolume->p1( p1x, p1y );
    double p2x, p2y;
    _sdkInputVolume->p2( p2x, p2y );
    double p3x, p3y;
    _sdkInputVolume->p3( p3x, p3y );
    double p4x, p4y;
    _sdkInputVolume->p4( p4x, p4y );

    // Nome do volume de entrada:
    const char *nome = _sdkInputVolume->nome();

    std::string s;
    std::stringstream out;
    out << nome;
    s = out.str( );
    out << "_MSA";
    s = out.str( );

    _sdkAttributeVolume = _sdk->criaVolumeFloat( s.c_str( ), ilFrom, ilTo, ilStep, clFrom, clTo, clStep, tFrom, tTo, tStep, p1x, p1y, p2x, p2y, p3x, p3y );
    _sdkAttributeVolume->setQuantizador();

    float* fap_outVolumeVoxels = NULL;
    fap_outVolumeVoxels = _sdkAttributeVolume->getVoxels( &fap_outVolumeVoxels );
    if( fap_outVolumeVoxels == NULL )
    {
        printf( "nao conseguiu obter os voxels do volume \n" );
    }

    // Gerando o volume de falha se saida:
    _dvpAttributeVolume = new DVPVolumeData( _dvpInputVolume, fap_outVolumeVoxels );

    // Calcula o volume de atributo de descontinuidade preenchendo os voxels do volume fap_volumeAttDVP:
    createAtt( _dvpInputVolume, _dvpAttributeVolume );
        
    // Adiciona o volume ao v3o2:
    _sdkAttributeVolume->setQuantizador();
    _sdk->associaVolume( _sdkAttributeVolume );
}


bool FaultAtt::copyVoxelsFromSubVolume( Vector3Di mapCoord, DVPVolumeData*& entrySubVolume, DVPVolumeData*& outPutVolume )
{
    // Caso um dos volumes seja nulo, retorna:
    if( (entrySubVolume == NULL) || (outPutVolume == NULL) )
            return false;

    // Dados especificos vindos do sub-volume(dimensoes, tipo de voxel, posicao espacial 3D)
    SoDataSet::DataType entrySubVolumeType;
    Vector3Di entrySubVolumeDim;
    entrySubVolume->getDataChar( entrySubVolumeType, entrySubVolumeDim );


    // Converte a coordenada de mapa recebida para coordenada de grid no sub-volume:	
    Vector3Df mapCoordf( mapCoord.x, mapCoord.y, mapCoord.z );
    Vector3Di indexCoord = entrySubVolume->geoRefToGrid( mapCoordf );

    // Obtem o valor do voxel no sub-volume:
    void* retVoxel = NULL;
    entrySubVolume->getVoxel( indexCoord.x, indexCoord.y, indexCoord.z, retVoxel );
    float* retVoxelf = (float*)retVoxel;
    float retVoxelfValue = retVoxelf[0];

    // Caso seja um voxel de falha, nao devera ser inserido:
    if( retVoxelfValue == 0 )
    {
        return false;
    }

    // Converte a coordenada de mapa recebida para coordenada de grid no volume de saida:
    Vector3Di indexCoord2 = outPutVolume->geoRefToGrid( mapCoordf  );

    // Insere o valor do voxel no volume de saida:
    outPutVolume->setVoxel( indexCoord2.x, indexCoord2.y, indexCoord2.z, (void*) (&(retVoxelfValue)) );                

    return true;
}



bool FaultAtt::createAtt( DVPVolumeData*& _spuEntryVolume, DVPVolumeData*& _spuFaultVolume )
{
    if( _spuEntryVolume == 0 )
    {
        std::cout << "createFaultAtt(). Erro no volume entrada!!!" << std::endl;
        return false;
    }

    std::cout << "Usando FaultAtt::createAtt()" << std::endl;

    SoDataSet::DataType type;
    Vector3Di dim;
    _spuEntryVolume->getDataChar( type, dim );

    _spuFaultVolume->P1 = _spuEntryVolume->P1;
    _spuFaultVolume->P2 = _spuEntryVolume->P2;
    _spuFaultVolume->P3 = _spuEntryVolume->P3;

    Vector3Di _spuSubVolumesSize;

    // Configurando os lados do sub-volume para que tenhamos somente um bloco.
    _spuSubVolumesSize.x = dim.x;
    _spuSubVolumesSize.y = dim.y;
    _spuSubVolumesSize.z = dim.z / fap_DivZ;

    // numX, numY e numZ armazenam o numero de sub-volumes em cada dimensao.
    int numX = dim.x / _spuSubVolumesSize.x;
    if( (dim.x % _spuSubVolumesSize.x) > 0 )  numX++;

    int numY = dim.y / _spuSubVolumesSize.y;
    if( (dim.y % _spuSubVolumesSize.y) > 0 )  numY++;

    int numZ = dim.z / _spuSubVolumesSize.z;
    if( (dim.z % _spuSubVolumesSize.z) > 0 )  numZ++;

    numX = 1;
    numY = 1;
    numZ = 1;

    std::cout << "Dimensoes do dado total: " << "X: " << dim.x << ", Y: " << dim.y << ", Z: " << dim.z << std::endl ;
    std::cout << "Dimensoes do dado parcial: " << "X: " << dim.x / numX << ", Y: " << dim.y / numY << ", Z: " << dim.z / numZ << std::endl ;

    // As variaveis min e max armazenarao as coordenadas da BBox do sub-volume dentro do volume pai:
    Vector3Di min, max;

    for( int i=0 ; i<numX ; i++ )
    {
        min.x = i*_spuSubVolumesSize.x - (int)(0.1f*_spuSubVolumesSize.x);
        min.x = MAX( 0, min.x );
        max.x = (i+1)*_spuSubVolumesSize.x + (int)(0.1f*_spuSubVolumesSize.x);
        max.x = MIN( dim.x, max.x );

        int k;
        k=0;
        for( ; k<numZ ; k++ )
        {
            min.z = k*_spuSubVolumesSize.z - (int)(0.35f*_spuSubVolumesSize.z);
            min.z = MAX( 0, min.z );
            max.z = (k+1)*_spuSubVolumesSize.z + (int)(0.35f*_spuSubVolumesSize.z);
            max.z = MIN( dim.z, max.z );

            int j;
            j=0;
            for( ; j<numY ; j++ )
            {
                std::cout << "Inicio de um novo Treinamento: " << "i: " << i << ", j: " << j << ", k: " << k <<  std::endl ;
                min.y = j*_spuSubVolumesSize.y - (int)(0.1f*_spuSubVolumesSize.y);
                min.y = MAX( 0, min.y );
                max.y = (j+1)*_spuSubVolumesSize.y + (int)(0.1f*_spuSubVolumesSize.y);
                max.y = MIN( dim.y, max.y );


                std::cout << "Caixa: ";
                std::cout << "X: " << min.x << ", " << max.x << " : " ;
                std::cout << "Y: " << min.y << ", " << max.y << " : " ;
                std::cout << "Z: " << min.z << ", " << max.z << " : " ;
                std::cout << std::endl ;

                // De posse das dimensoes, cria o novo volume:
                DVPVolumeData* thisSubVolume = 0;


                // Caso o sub-volume seja o proprio volume de entrada, simplesmente posiciona o ponteiro:
                if( (numX == 1) && (numY == 1) && (numZ == 1) )
                {
                    thisSubVolume = _spuEntryVolume;
                }
                else
                {
                    // TODO:
                    // thisSubVolume = new DVPVolumeData( _spuEntryVolume, min, max );
//                    if( _spuEntryVolume->getSubVolume( subVolume, thisSubVolume ) == false )
//                    {
//                        return -1;
//                    }
                }

                // Medindo o tempo de treinamento:
                time_t initTime;
                initTime = time (NULL);

                // Cria essa instancia de SPC:
                SeismicProcessingController* thisSPC = new SeismicProcessingController();
                thisSPC->setParameters( fap_Lambda, fap_Eb, fap_En, fap_Alpha, fap_D, fap_Amax,
                                fap_SamplesSize, fap_NneuronsMulti, fap_MinStdDevForTrain );
                thisSPC->Train( thisSubVolume, NULL );

                // Imprimindo o tempo de treinamento e classificacao:
                time_t endTime;
                endTime = time (NULL);
                std::cout << "z: " << k << " de: " << numZ << ". Levou: " << (endTime-initTime) << " seg." << std::endl;

                // Iniciando o processo de mapeamento do volume de falhas:
                DVPVolumeData* thisFaultVolume = 0;
                thisSPC->createFaultVolume2( fap_Window, fap_HDist, thisFaultVolume, true );

                SoDataSet::DataType type;
                Vector3Di dim;
                thisFaultVolume->getDataChar( type, dim );

                // Uma vez criado, esse sub-volume de falha deve ser copiado no volume de falha de saida:
                for( int i = 0; i < dim.x; i++ )
                {
                    for( int j = 0; j < dim.y; j++ )
                    {
                        for( int k = 0; k < dim.z; k++ )
                        {
                            void* retVoxel = NULL;
                            thisFaultVolume->getVoxel( i, j, k, retVoxel );
                            float* retVoxelf = (float*) retVoxel;                                            
                            float dvpValue = retVoxelf[0];

                            _spuFaultVolume->setVoxel( i, j, k, (void*) (&(dvpValue)) );

                            //copyVoxelsFromSubVolume( Vector3Di( i, j, k ), thisFaultVolume, _spuFaultVolume );
                        }
                    }
                }

                // Uma vez criado, esse sub-volume de falha deve ser copiado no volume de falha de saida:
                /*for( int i=ilFrom ; i<ilTo ; i++ )
                {
                    for( int j=clFrom ; j<clTo ; j++ )
                    {
                        for( int k=zlFrom ; k<zlTo ; k++ )
                        {
                                copyVoxelsFromSubVolume( Vector3Di( i, j, k ), thisFaultVolume, _spuFaultVolume );
                        }
                    }
                }*/

                // Deleta o subVolume de clusters:
                if( thisSubVolume != _spuEntryVolume )
                {
                    delete thisSubVolume;
                }

                // Deleta o subVolume de falhas:
                if( thisFaultVolume != NULL )
                {
                    delete thisFaultVolume;                                                                
                }
            }
        }
    }

    return 1;
}

}
