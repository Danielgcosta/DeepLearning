/* 
 * File:   main.cpp
 * Author: dcosta
 */

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <vector>

using namespace std;


class Voxel{
public:
    int inLine; //inline
    int crossline;
    float x;
    float y;
    float depth;
};


class Seismic{
public:
    int year;
    vector<Voxel> voxels;
};
            
            
int main()
{
    //string fileNames[4] { "DadosNorne2001.csv", "DadosNorne2003.csv", "DadosNorne2004.csv", "DadosNorne2006.csv" };
    ifstream file( "DadosNorne2001.csv" );
    
    Seismic seismic;
    seismic.year = 2001;
    
    int i = 0;
    while ( ~ios::eofbit || file.good() )
    {
        Voxel voxel;
        string stringValue;
        
        getline ( file, stringValue, ',' );
        voxel.inLine = stoi(string( stringValue, 1, stringValue.length()-2 ));
                
        getline ( file, stringValue, ',' );
        voxel.crossline = stoi(string( stringValue, 1, stringValue.length()-2 ));
        
        getline ( file, stringValue, ',' );
        voxel.x = stof(string( stringValue, 1, stringValue.length()-2 ));
        
        getline ( file, stringValue, ',' );
        voxel.y = stof(string( stringValue, 1, stringValue.length()-2 ));
        
        getline ( file, stringValue, ',' );
        voxel.depth = stof(string( stringValue, 1, stringValue.length()-2 ));
        
        seismic.voxels.push_back( voxel );
        cout << i << endl;
        i++;
    }
    file.close();
    
    return 0;       
}