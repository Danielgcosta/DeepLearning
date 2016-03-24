/* 
 * File:   MSA_DL.h
 * Author: jnavarro
 *
 * Created on October 6, 2015, 7:08 PM
 */

#ifndef MSA_DL_H
#define	MSA_DL_H

#include "plugin.h"
#include "v3o2_sdk.h"

class MSA_DL : public Plugin
{
    
public:
    
    /**
     * Construtro padrão.
     */
    MSA_DL();
            
    /**
     * Destrutor padrão.
     */
    virtual ~MSA_DL();
    
    /** Métodos da interface de plugin. */

    void init(void *sdk);

    void start();

    void end();

    const std::string &name() const;

    const std::string &title() const;

    const std::string &description() const;

    const std::string &vendor() const;

    void version(int &major, int &minor, int &build) const;

    const std::string &status() const;
    
protected:

    std::string _name;
    
    std::string _title;
    
    std::string _description;
    
    std::string _vendor;
    
    int _major;
    
    int _minor;
    
    int _build;
    
    std::string _status;

private:

    /**
     * Elementos de interface
     */
    SDKInterface* _interface;

    /**
     * Objeto de dialogo entre o v3o2 e o plugin
     */
    v3o2_SDK* _sdk;

};

/*---------------------------------------------------------------------------*/
extern "C"
{
	FUNCTION_DECLSPEC
	MSA_DL *CreatePlugin( );

	FUNCTION_DECLSPEC
	void DestroyPlugin( MSA_DL *p );
};
/*---------------------------------------------------------------------------*/

#endif	/* MSA_DL_H */
