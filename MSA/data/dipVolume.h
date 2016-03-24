/*
 * dipVolume.h
 *
 *  Created on: 04/03/2015
 *      Author: Aurelio
 */

#ifndef DIPVOLUME_H_
#define DIPVOLUME_H_

#include "../../math/Vector.h"
#include "DataGrid3.h"
#include "utl.h"

#include <limits>
#include <cstdio>
#include <iostream>
#include <limits.h>
#include <string>
#include <map>

using namespace MSA;

const float DIP_DUMMY_VALUE = 0.0f;

namespace MSA
{

/**
 * Classe auxiliar temporaria utilizada para armazenar um volume de mergulho.
 */
class dipNode
{
public:
	Vector4Df  _dir;		//< Direcoes +x, +y, -x, -y.
	float _conf;				//< Valor da discontinuidade nesse ponto (quanto maior mais discontinuo).
	float _dummyValue;


	/**
	 * Salva esse no. Recebe o ponteiro para o arquivo.
	 * @param f Ponteiro para o arquivo onde o no deve ser salvo.
	 * @return Retorna -1 em caso de erro, ou 0 caso ok.
	 */
	int saveNode( FILE *&f )
	{
		// Casos de retorno:
		if( f == NULL )
			return -1;

		if( fprintf( f, "x: %f\n", _dir.x ) < 0 ) return -1;
		if( fprintf( f, "y: %f\n", _dir.y ) < 0 ) return -1;
		if( fprintf( f, "z: %f\n", _dir.z ) < 0 ) return -1;
		if( fprintf( f, "w: %f\n", _dir.w ) < 0 ) return -1;
		if( fprintf( f, "d: %f\n", _conf ) < 0 ) return -1;

		return 0;
	}


	/**
	 * Carrega esse no de um arquivo. Recebe o ponteiro para o arquivo
	 * @param f Ponteiro para o arquivo onde o grupo deve ser salvo.
	 * @return Retorna -1 em caso de erro, ou 0 caso ok.
	 */
	int loadNode( FILE*&f )
	{
		// Casos de retorno:
		if( f == NULL )
			return -1;

		float xPos, yPos, xNeg, yNeg;
		if( fscanf( f, "x: %f\n", &xPos ) != 1 ) return -1;
		if( fscanf( f, "y: %f\n", &yPos  ) != 1 ) return -1;
		if( fscanf( f, "z: %f\n", &xNeg ) != 1 ) return -1;
		if( fscanf( f, "w: %f\n", &yNeg) != 1 ) return -1;
		if( fscanf( f, "d: %f\n", &_conf ) != 1 ) return -1;

		_dir.x = xPos;
		_dir.y = yPos;
		_dir.z = xNeg;
		_dir.w = yNeg;

		return 0;
	}


	/**
	 * Construtor. Le o no a partir de um arquivo.
	 */
	dipNode( FILE*&f ) :
		_dummyValue( (float)DIP_DUMMY_VALUE )
	{
		loadNode( f );
	}


	/**
	 * Constructor
	 */
	dipNode( Vector4Df dir = Vector4Df( DIP_DUMMY_VALUE, DIP_DUMMY_VALUE,
			DIP_DUMMY_VALUE, DIP_DUMMY_VALUE ), float conf=std::numeric_limits<float>::max() ) :
				_dummyValue( DIP_DUMMY_VALUE )
	{
		_dir = dir;
		_conf = conf;
	}

	/**
	 * Insere nova configuracao para esse no.
	 */
	void setNode( Vector4Df dir, float conf )
	{
		_dir = dir;
		_conf = conf;
	}


	/**
	 * Configura os valores das direcoes do no.
	 */
	void setDir( Vector4Df dir )
	{
		_dir = dir;
	}


	/**
	 * Obtém os valores das direcoes do no.
	 */
	Vector4Df getDir()
	{
		return _dir;
	}


	/**
	 * Configura um dos valores das direcoes do no.
	 */
	void setDir( float vdir, int pos )
	{
		if( pos == 0 )
			_dir.x = vdir;
		if( pos == 1 )
			_dir.y = vdir;
		if( pos == 2 )
			_dir.z = vdir;
		if( pos == 3 )
			_dir.w = vdir;
	}


	/**
	 * Configura o valor de discontinuidade do no.
	 */
	void setConf( float conf )
	{
		_conf = conf;
	}


	/**
	 * Obtem o valor de discontinuidade do no.
	 */
	float getConf()
	{
		return _conf;
	}


	/**
	 * Retorna se o no e valido.
	 */
	bool isValid()
	{
		if( (_dir.x < -2.0f) || (_dir.x > 2.0f))
			return false;
		if( (_dir.y < -2.0f) || (_dir.y > 2.0f))
			return false;
		if( (_dir.z < -2.0f) || (_dir.z > 2.0f))
			return false;
		if( (_dir.w < -2.0f) || (_dir.w > 2.0f))
			return false;
		return true;
	}


	/**
	 * Destructor.
	 */
	~dipNode()
	{
	}

};



/**
 * Classe que permite manter informacoes de um volume de mergulho com peso nos nos.
 */
class weightedDipVolume
{
public:

	std::vector<std::vector<std::vector<dipNode*> > > _wdv_WeightedDipVolume;	//< Estrutura contendo os vetores de mergulho e os pesos em cada posicao.
	Vector3Di _wdv_Dim;	//< Dimensoes do volume sismico.
	FILE *_wdv_fp;


	/**
	 * Construtor.
	 */
	weightedDipVolume( Vector3Di wdv_Dim )
	{
		_wdv_fp = NULL;
		_wdv_Dim = wdv_Dim;
		// Criando a estrutura de dados que ira manter os dados de mergulho:
		_wdv_WeightedDipVolume.resize(_wdv_Dim.x);
		for (int i = 0; i < _wdv_Dim.x; i++)
		{
			_wdv_WeightedDipVolume[i].resize(_wdv_Dim.y);
			for (int j = 0; j < _wdv_Dim.y; j++)
			{
				_wdv_WeightedDipVolume[i][j].resize(_wdv_Dim.z);
				for (int k = 0; k < _wdv_Dim.z; k++)
					_wdv_WeightedDipVolume[i][j][k] = new dipNode();
			}
		}
	}


	/**
	 * Construtor.
	 */
	weightedDipVolume( std::string& filename )
	{
		loadWDV( filename );
	}


	/**
	 * Construtor.
	 */
	~weightedDipVolume( )
	{
		for (int i = 0; i < _wdv_Dim.x; i++)
		{
			for (int j = 0; j < _wdv_Dim.y; j++)
			{
				for (int k = 0; k < _wdv_Dim.z; k++)
				{
					dipNode* thisNode = _wdv_WeightedDipVolume[i][j][k];
					if( thisNode != NULL )
						delete thisNode;
				}
			}
		}
		_wdv_WeightedDipVolume.clear();
	}


	/**
	 * Retorna um ponteiro para um no.
	 */
	dipNode* getDipNode( Vector3Di pos )
	{
		dipNode* thisNode = _wdv_WeightedDipVolume[pos.x][pos.y][pos.z];
		return thisNode;
	}


	/**
	 * Salva um volume de mergulho com pesos
	 */
	bool saveWDV( std::string& filename )
	{
		_wdv_fp = fopen( filename.c_str( ), "wt" );
		if (_wdv_fp == NULL)
			return false;

		fprintf( _wdv_fp, "WDV_DATA_1.0\n" );
		fprintf( _wdv_fp, "%d, %d, %d\n",  ( _wdv_Dim.x ) ,  ( _wdv_Dim.y ) ,  ( _wdv_Dim.z ) );

		for (int i = 0; i < _wdv_Dim.x; i++)
		{
			for (int j = 0; j < _wdv_Dim.y; j++)
			{
				for (int k = 0; k < _wdv_Dim.z; k++)
				{
					dipNode* thisNode = _wdv_WeightedDipVolume[i][j][k];
					thisNode->saveNode( _wdv_fp );
				}
			}
		}
		fprintf( _wdv_fp, "\n" );

		fclose( _wdv_fp );
		return true;
	}


	/**
	 * Le um volume de mergulho com pesos
	 */
	bool loadWDV( std::string& filename )
	{
		// Lendo o arquivo dos dados:
		_wdv_fp = fopen( filename.c_str( ), "rt" );
		if (_wdv_fp == NULL)
		{
			std::cout << "::loadWDV ret -1. _wdv_fp=NULL: " << filename << std::endl;
			return false;
		}
		if( fscanf( _wdv_fp, "WDV_DATA_1.0\n" ) != 0 )
		{
			std::cout << "loadWDV:WDV_DATA_1.0 - Erro!!" << std::endl;
			return false;
		}

		int dimX, dimY, dimZ;
		if( fscanf( _wdv_fp,"%d, %d, %d\n", &dimX, &dimY, &dimZ ) != 3 )
		{
			std::cout << "loadWDV: dimX, dimY, dimZ - Erro!!" << std::endl;
			return false;
		}

		_wdv_Dim.x = dimX;
		_wdv_Dim.y = dimY;
		_wdv_Dim.z = dimZ;


		// Criando a estrutura de dados que ira manter os dados de mergulho:
		_wdv_WeightedDipVolume.resize(_wdv_Dim.x);
		for (int i = 0; i < _wdv_Dim.x; i++)
		{
			_wdv_WeightedDipVolume[i].resize(_wdv_Dim.y);
			for (int j = 0; j < _wdv_Dim.y; j++)
			{
				_wdv_WeightedDipVolume[i][j].resize(_wdv_Dim.z);
				for (int k = 0; k < _wdv_Dim.z; k++)
				{
					_wdv_WeightedDipVolume[i][j][k] = new dipNode();
					dipNode* thisNode = _wdv_WeightedDipVolume[i][j][k];
					thisNode->loadNode( _wdv_fp );
				}
			}
		}

		fclose( _wdv_fp );

		return true;
	}


	/**
	 * Calcula os valores de mergulho nas posicoes em que tal informacao nao existe.
	 */
	int setMeanDip( Vector3Di init, Vector3Di end, int dir )
	{
		// Caso nessas coordenadas nao exista a informacao de dir, retorna erro:
		dipNode* initNode = getDipNode( init );
		dipNode* endNode = getDipNode( end );

		if( initNode == NULL )
			return -1;
		if( endNode == NULL )
			return -1;

		// Caso a informacão de direcao nao exista em algum deles, retorna erro:
		Vector4Df initNodeDir = initNode->getDir();
		Vector4Df endNodeDir = endNode->getDir();

		float initNodeDirArray[4];
		initNodeDirArray[0] = initNodeDir.x;
		initNodeDirArray[1] = initNodeDir.y;
		initNodeDirArray[2] = initNodeDir.z;
		initNodeDirArray[3] = initNodeDir.w;
		float endNodeDirArray[4];
		endNodeDirArray[0] = endNodeDir.x;
		endNodeDirArray[1] = endNodeDir.y;
		endNodeDirArray[2] = endNodeDir.z;
		endNodeDirArray[3] = endNodeDir.w;

		if( initNodeDirArray[dir] == DIP_DUMMY_VALUE )
			return -1;
		if( endNodeDirArray[dir] == DIP_DUMMY_VALUE )
			return -1;

		float inc = (endNodeDirArray[dir] - initNodeDirArray[dir] ) / (end.z - init.z);
		for( int cont=0, i=init.z+1 ; i<end.z ; i++, cont++ )
		{
			dipNode* tmpNode = getDipNode( Vector3Di( init.x, init.y, i ) );
			Vector4Df tmpNodeDir = tmpNode->getDir();

			float tmpNodeDirArray[4];
			tmpNodeDirArray[0] = tmpNodeDir.x;
			tmpNodeDirArray[1] = tmpNodeDir.y;
			tmpNodeDirArray[2] = tmpNodeDir.z;
			tmpNodeDirArray[3] = tmpNodeDir.w;

			if( tmpNodeDirArray[dir] != DIP_DUMMY_VALUE )
			{
				std::cout << "setMeanDip: tmpNodeDirArray[dir] != DIP_DUMMY_VALUE  - Erro!!" << std::endl;
				return -1;
			}

			if( initNodeDirArray[dir] == endNodeDirArray[dir] )
				tmpNodeDirArray[dir] = initNodeDirArray[dir];
			else
				tmpNodeDirArray[dir] = initNodeDirArray[dir] + (cont*inc);

			// Inserindo o novo valor de direcao desse no:
			tmpNode->setDir( tmpNodeDirArray[dir], dir );

		}
		return 0;
	}


	/**
	 * Uma vez recebida a coordenada de um traco uma posicao inicial e um valor de direcao, retorna a proxima posicao valida,
	 * ou -2 caso a procura chegue ao final do traco. Retorna -1 em caso de erro,
	 */
	int findNextDip( int xCoord, int yCoord, int zCoord, int dir )
	{
		if( (xCoord<0) || (xCoord>=_wdv_Dim.x) )
			return -1;
		if( (yCoord<0) || (yCoord>=_wdv_Dim.y) )
			return -1;
		if( (zCoord<0) || (zCoord>=_wdv_Dim.z) )
			return -2;

		// Caso contrario, encontra cada pico e realiza os calculos aproximados de dip:
		for (int k = zCoord; k < _wdv_Dim.z; k++)
		{
			dipNode* thisNode = getDipNode( Vector3Di(xCoord, yCoord, k) );
			if( thisNode == NULL )
			{
				std::cout << "findNextDip: thisNode == NULL   - Erro!!" << std::endl;
				return -1;
			}

			// Caso a informacao de direcao nao exista em algum deles, retorna erro:
			Vector4Df thisNodeDir = thisNode->getDir();

			float thisNodeDirArray[4];
			thisNodeDirArray[0] = thisNodeDir.x;
			thisNodeDirArray[1] = thisNodeDir.y;
			thisNodeDirArray[2] = thisNodeDir.z;
			thisNodeDirArray[3] = thisNodeDir.w;

			if( thisNodeDirArray[dir] == DIP_DUMMY_VALUE )
				continue;
			else
				return k;
		}
		return -2;
	}


	/**
	 * Encontra os dips medios ao longo de todo um traco.
	 * Recebe as coordenadas 2D do traco em coordenadas de indice.
	 */
	int findDips( int xCoord, int yCoord, int dir )
	{
		if( (xCoord<0) || (xCoord>=_wdv_Dim.x) )
			return -1;
		if( (yCoord<0) || (yCoord>=_wdv_Dim.y) )
			return -1;

		// Caso contratio, encontra cada pico e realiza os calculos aproximados de dip:
		int k=0;
		bool findNext = true;
		int ret = findNextDip( xCoord, yCoord, k, dir );
		if( ret < 0 )
			return -1;
		// Caso exista um topo, k recebe esse valor:
		k = ret;
		while( findNext )
		{
			int topo = k;
			ret = findNextDip( xCoord, yCoord, (k+1), dir );
			if(ret == -2 )
				break;
			int base = ret;
			// Tendo topo e base, encontra os dips medios no intervalo:
			if( setMeanDip( Vector3Di( xCoord, yCoord, topo ), Vector3Di( xCoord, yCoord, base ), dir ) == -1 )
			{
				std::cout << "findDips: setMeanDip invalido - Erro!!" << std::endl;
				return -1;
			}
			k = base;
		}
		return 0;
	}


	/**
	 * Completa todos os valores de dip com base nos valores ja existentes na estrutura de dados.
	 */
	int findAllDips()
	{
		for (int i = 0; i < _wdv_Dim.x; i++)
		{
			for (int j = 0; j < _wdv_Dim.y; j++)
			{
				for( int dir=0 ; dir<4 ; dir++ )
				{
					findDips( i, j, dir ) ;
				}
			}
		}
		return 0;
	}



	/**
	 * Funcao de mapeamento de horizontes baseados em weightedDip
	 * Recebe um vector contendo a primeira posicao (semente) mapeada, e mapeia os voxels utilizando as
	 * informacoes  disponiveis.
	 */
	int map_WeightedDipHorizon( Vector3Di seed, float msaMax, std::vector<Vector3Df>& horizonRet )
	{
		horizonRet.clear();

		DataGrid3< Vector3Df > tmpHorizon;
		tmpHorizon.setDimensions(_wdv_Dim.x, _wdv_Dim.y, 1 );

		// Inicializando a estrutura de dados:
		for( int x = 0; x < _wdv_Dim.x; x++ )
			for( int y = 0; y < _wdv_Dim.y; y++ )
				tmpHorizon( x, y, 0 ) = Vector3Df( (float)x, (float)y, (float)-1 );

		float max = std::numeric_limits<float>::max();
		float msaThreshold = max - msaMax;

		// MultiMap que ira conter todos os nos ordenados por relevancia:
		std::multimap<float, Vector3Df > hzMultiMap;
		std::multimap<float, Vector3Df >::iterator hzMultiMapItr;
		hzMultiMap.insert( std::pair<float, Vector3Df >( max,  Vector3Df( (float)seed.x, (float)seed.y, (float)seed.z) ) );

		int contador = 0;
		while( hzMultiMap.size() > 0 )
		{
			// Pega o elemento de maior relevancia:
			hzMultiMapItr = hzMultiMap.end();
			--hzMultiMapItr;
			Vector3Df thisNode = (*hzMultiMapItr).second;
			dipNode* thisdipNode = _wdv_WeightedDipVolume[Rounded(thisNode.x)][Rounded(thisNode.y)][Rounded(thisNode.z)];

			// Caso o no nao seja valido, continua:
			if( thisdipNode->isValid() == false )
			{
				hzMultiMap.erase( hzMultiMapItr );
				continue;
			}

			// Pega as informacoes desse no:
			Vector4Df dirs =  thisdipNode->getDir();
			float conf = (*hzMultiMapItr).first;

			// Caso o elemento de maior relevancia tenha valor de continuidade mais alto que o threshold estabelecido,
			// termina a procura por novos voxels:
			if( msaThreshold > conf )
				break;

			if( contador > (_wdv_Dim.x * _wdv_Dim.y ) )
				break;

			// Insere esse no como sendo parte do horizonte:
			tmpHorizon( Rounded(thisNode.x), Rounded(thisNode.y), 0 ) = Vector3Df( thisNode.x, thisNode.y, thisNode.z );

			if( (thisNode.x >= _wdv_Dim.x-2) || (thisNode.x <= 1) )
			{
				hzMultiMap.erase( hzMultiMapItr );
				continue;
			}
			if( (thisNode.y >= _wdv_Dim.y-2) || (thisNode.y <= 1) )
			{
				hzMultiMap.erase( hzMultiMapItr );
				continue;
			}

			// Encontra as posicoes dos seus quatro vizinhos, e insere no multimap:
			Vector3Df xPos( thisNode.x + 1, thisNode.y		, thisNode.z + dirs.x );
			Vector3Df yPos( thisNode.x		, thisNode.y + 1	, thisNode.z + dirs.y );
			Vector3Df xNeg( thisNode.x - 1, thisNode.y			, thisNode.z + dirs.z );
			Vector3Df yNeg( thisNode.x		, thisNode.y - 1	, thisNode.z + dirs.w );

			Vector3Df xPos_tmpHorizon = tmpHorizon( Rounded(xPos.x), Rounded(xPos.y), 0 );
			if( xPos_tmpHorizon.z == -1 )
			{
				if( (Rounded(xPos.z) >= _wdv_Dim.z-2) || (Rounded(xPos.z) <= 1) )
					continue;
				dipNode* xPos_dipNode = _wdv_WeightedDipVolume[Rounded(xPos.x)][Rounded(xPos.y)][Rounded(xPos.z)];
				float val = max - xPos_dipNode->getConf();
				hzMultiMap.insert( std::pair<float, Vector3Df >( val,  xPos) );
			}


			Vector3Df yPos_tmpHorizon = tmpHorizon( Rounded(yPos.x), Rounded(yPos.y), 0 );
			if( yPos_tmpHorizon.z == -1 )
			{
				if( (Rounded(yPos.z) >= _wdv_Dim.z-2) || (Rounded(yPos.z) <= 1) )
					continue;
				dipNode* yPos_dipNode = _wdv_WeightedDipVolume[Rounded(yPos.x)][Rounded(yPos.y)][Rounded(yPos.z)];
				float val = max - yPos_dipNode->getConf();
				hzMultiMap.insert( std::pair<float, Vector3Df >( val,  yPos) );
			}


			Vector3Df xNeg_tmpHorizon = tmpHorizon( Rounded(xNeg.x), Rounded(xNeg.y), 0 );
			if( xNeg_tmpHorizon.z == -1 )
			{
				if( (Rounded(xNeg.z) >= _wdv_Dim.z-2) || (Rounded(xNeg.z) <= 1) )
					continue;
				dipNode* xNeg_dipNode = _wdv_WeightedDipVolume[Rounded(xNeg.x)][Rounded(xNeg.y)][Rounded(xNeg.z)];
				float val = max - xNeg_dipNode->getConf();
				hzMultiMap.insert( std::pair<float, Vector3Df >( val,  xNeg) );
			}

			Vector3Df yNeg_tmpHorizon = tmpHorizon( Rounded(yNeg.x), Rounded(yNeg.y), 0 );
			if( yNeg_tmpHorizon.z == -1 )
			{
				if( (Rounded(yNeg.z) >= _wdv_Dim.z-2) || (Rounded(yNeg.z) <= 1) )
					continue;
				dipNode* yNeg_dipNode = _wdv_WeightedDipVolume[Rounded(yNeg.x)][Rounded(yNeg.y)][Rounded(yNeg.z)];
				float val = max - yNeg_dipNode->getConf();
				hzMultiMap.insert( std::pair<float, Vector3Df >( val,  yNeg) );
			}

			contador++;
			hzMultiMap.erase( hzMultiMapItr );
		}

		// Retornando o vetor de pontos:
		for( int x = 0; x < _wdv_Dim.x; x++ )
			for( int y = 0; y < _wdv_Dim.y; y++ )
			{
				Vector3Df pos = tmpHorizon( x, y, 0 );
				if( pos.z != -1 )
					horizonRet.push_back( Vector3Df( pos.x, pos.y, pos.z ) );
			}

		return 0;
	}


};

}

#endif /* DIPVOLUME_H_ */
