/* 
 * File:   MSA_DLCommand.cpp
 * Author: jnavarro
 * 
 * Created on October 6, 2015, 7:19 PM
 */

#include "MSA_DLCommand.h"
#include "FaultAtt.h"

#include <iostream>
#include <sstream>

MSA_DLCommand::MSA_DLCommand( v3o2_SDK* sdk ) : _sdk( sdk )
{
    
}


MSA_DLCommand::~MSA_DLCommand() 
{
    
}


void MSA_DLCommand::executar( SDKItemMenu *item )
{
    std::cout << "SKDItemMenu chamada...\n";
}

    
void MSA_DLCommand::executar( SDKItemMenu *item, SDKObjeto* objeto )
{    
    if( objeto == 0 )
    {
        printf("\n MSA_DLCommand::executar. Volume invalido. Abortando o cálculo.\n" );
        return;
    }

    SDKVolume* volume = NULL;
    volume = dynamic_cast<SDKVolume*>(objeto);
    if( volume == 0 )
    {
            printf("\n MSA_DLCommand::executar. SDKVolume* não recebido. Abortando o cálculo.\n" );
            return;
    }

    // Cria o objeto de calculo de atributo de descontinuidade:
    MSA::FaultAtt faultAtt( _sdk, volume );
    faultAtt.MSA_Calc();
}
