/* 
 * File:   MSA_DLCommand.h
 * Author: jnavarro
 *
 * Created on October 6, 2015, 7:19 PM
 */

#ifndef MSA_DLCOMMAND_H
#define	MSA_DLCOMMAND_H

// Includes do SDK
#include "SDKMenu.h"
#include "SDKSecao.h"
#include "v3o2_sdk.h"
#include "SDKVolume.h"


class MSA_DLCommand : public SDKComando
{
    
public:
    
    MSA_DLCommand( v3o2_SDK* sdk );
    
    ~MSA_DLCommand();

    /**
    * Implementa a acao que deve ser executada pelo comando
    * @param[in] item     Item que chamou o comando a ser executado
    */
    void executar( SDKItemMenu *item );

    /**
    * Implementa a acao que deve ser executada pelo comando
    * @param[in] item     Item que chamou o comando a ser executado
    * @param[im] objeto   Objeto responsavel por executar o cando
    */
    void executar(SDKItemMenu *item, SDKObjeto* objeto);

private:

    v3o2_SDK* _sdk;
};

#endif	/* MSA_DLCOMMAND_H */

