/* 
 * File:   MSA_DL.cpp
 * Author: jnavarro
 * 
 * Created on October 6, 2015, 7:09 PM
 */

// Includes sdk
#include "MSA_DL.h"
#include "SDKVolume.h"
#include "SDKMenu.h"
#include "MSA/MSA_DLCommand.h"

#include <iostream>

/* --------------------------------------------------------------------- */
extern "C" {

	#ifdef _WIN32
	__declspec(dllexport)
	#endif

		MSA_DL* CreatePlugin() {
		return new MSA_DL;
	}

	#ifdef _WIN32
	__declspec(dllexport)
	#endif

	void DestroyPlugin(MSA_DL* p) {
		delete p;
	}

};
/* --------------------------------------------------------------------- */

MSA_DL::MSA_DL( )
{        
    // Propriedades do Plugin
    _name = "Minimal Similarity Accumulation With Deep Learning\0";
    _title = "MSA-DL.\0";
    _description = "Cálculo de atributo de falha com Deep Learning.\0";
    _vendor = "Tecgraf\0";
    _major = 0;
    _minor = 0;
    _build = 1;
    _status = "beta\0";
}


MSA_DL::~MSA_DL( )
{

}


void MSA_DL::init( void* sdk )
{
    std::cout << "*** MSA_DL::init ***" << std::endl;
    
    _sdk = static_cast<v3o2_SDK*>( sdk );
    
    if ( _sdk != 0 )
    {
        // Cria o elemento de interface
        _interface = _sdk->sdkInterface( );

        // Cria novo item do menu do volume
        SDKMenu* menu = _interface->criaMenu( "MSA com DeepLearning" );

        // Cria instância do comando que será executado pelo item do menu
        MSA_DLCommand* command = new MSA_DLCommand( _sdk );

        // Cria subitem do menu com o comando associado
        SDKItemMenu* itemMenu = _interface->criaItemMenu( "Calcular MSA-DL", command );

        // Adciona item no menu
        menu->adiciona( itemMenu );

        // Registra o novo menu para o objeto volume do v3o2
        _sdk->registraMenuVolume( menu );
    }
}


void MSA_DL::start()
{	
    std::cout << "*** MSA_DL::start ***" << std::endl;
}


void MSA_DL::end()
{

}


const std::string& MSA_DL::name() const
{
    return _name;
}


const std::string& MSA_DL::title() const
{
    return _title;
}


const std::string& MSA_DL::description() const
{
    return _description;
}


const std::string& MSA_DL::vendor() const
{
    return _vendor;
}


void MSA_DL::version(int &major, int &minor, int &build) const
{
    major = _major;
    minor = _minor;
    build = _build;
}


const std::string& MSA_DL::status() const
{
    return _status;
}
