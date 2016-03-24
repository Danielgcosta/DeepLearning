/* 
 * File:   CUDAErrorMessage.h
 * Author: eaperz
 *
 * Created on December 11, 2015, 3:24 PM
 */

#ifndef CUDAERRORMESSAGE_H
#define	CUDAERRORMESSAGE_H

#include <iostream>
#include <sstream>
#include <string>
#include <curand.h>
#include "cublas_v2.h"


namespace gpu
{

class CUDAErrorMessage
{
    public:
        static void printErrorMessage( const cudaError_t& status, const std::string& file, int line )
        {
            std::cerr << cudaErrorMessage( status, file, line ) << "\n";
        }
        
        static void printErrorMessage( const cublasStatus_t& status, const std::string& file, int line )
        {
            std::cerr << cublasErrorMessage( status, file, line ) << "\n";
        }
        
        static void printErrorMessage( const curandStatus_t& status, const std::string& file, int line )
        {
            std::cerr << curandErrorMessage( status, file, line ) << "\n";
        }
        
    private:
        static std::string cudaErrorMessage( const cudaError_t& status, const std::string& file, int line )
        {
            std::stringstream message;
            message << "CUDA error: " << std::string(cudaGetErrorString(status)) << "\n";
            
            if ( !file.empty() )
            {
                message << "\t" << file << ":" << line << "\n";
            }
            
            return message.str();
        }

        static std::string cublasErrorMessage( const cublasStatus_t& status, const std::string& file, int line )
        {
            std::stringstream message;
            message << "cuBLAS error: ";
            switch ( status )
            {
                case CUBLAS_STATUS_SUCCESS:
                    message << "The operation completed successfully.";
                    break;

                case CUBLAS_STATUS_NOT_INITIALIZED:
                    message << "The cuBLAS library was not initialized.";
                    break;

                case CUBLAS_STATUS_ALLOC_FAILED:
                    message << "Resource allocation failed inside the cuBLAS library.";
                    break;

                case CUBLAS_STATUS_INVALID_VALUE:
                    message << "An unsupported value or parameter was passed to the function (a negative vector size, for example).";
                    break;

                case CUBLAS_STATUS_ARCH_MISMATCH:
                    message << "The function requires a feature absent from the device architecture.";
                    break;

                case CUBLAS_STATUS_MAPPING_ERROR:
                    message << "An access to GPU memory space failed.";
                    break;

                case CUBLAS_STATUS_EXECUTION_FAILED:
                    message << "The GPU program failed to execute.";
                    break;

                case CUBLAS_STATUS_INTERNAL_ERROR:
                    message << "An internal cuBLAS operation failed.";
                    break;

                case CUBLAS_STATUS_NOT_SUPPORTED:
                    message << "The functionnality requested is not supported.";
                    break;

//                case CUBLAS_STATUS_LICENSE_ERROR:
//                    message << "The functionnality requested requires some license and an error was detected when trying to check the current licensing.";
//                    break;
                    
                default:
                    message << "Undefined error.";
                    break;
            }
            
            message << "\n";
            if ( !file.empty() )
            {
                message << "\t" << file << ":" << line << "\n";
            }
            
            return message.str();
        }
        
        static std::string curandErrorMessage( const curandStatus_t& status, const std::string& file, int line )
        {
            std::stringstream message;
            message << "cuRAND error: ";
            switch ( status )
            {
                case CURAND_STATUS_SUCCESS:
                    message << "CURAND_STATUS_SUCCESS";
                    break;
                    
                case CURAND_STATUS_VERSION_MISMATCH:
                    message << "CURAND_STATUS_VERSION_MISMATCH";
                    break;
                    
                case CURAND_STATUS_NOT_INITIALIZED:
                    message << "CURAND_STATUS_NOT_INITIALIZED";
                    break;
                    
                case CURAND_STATUS_ALLOCATION_FAILED:
                    message << "CURAND_STATUS_ALLOCATION_FAILED";
                    break;
                    
                case CURAND_STATUS_TYPE_ERROR:
                    message << "CURAND_STATUS_TYPE_ERROR";
                    break;
                    
                case CURAND_STATUS_OUT_OF_RANGE:
                    message << "CURAND_STATUS_OUT_OF_RANGE";
                    break;
                    
                case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
                    message << "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
                    break;
                    
                case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
                    message << "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
                    break;
                    
                case CURAND_STATUS_LAUNCH_FAILURE:
                    message << "CURAND_STATUS_LAUNCH_FAILURE";
                    break;
                    
                case CURAND_STATUS_PREEXISTING_FAILURE:
                    message << "CURAND_STATUS_PREEXISTING_FAILURE";
                    break;
                    
                case CURAND_STATUS_INITIALIZATION_FAILED:
                    message << "CURAND_STATUS_INITIALIZATION_FAILED";
                    break;
                    
                case CURAND_STATUS_ARCH_MISMATCH:
                    message << "CURAND_STATUS_ARCH_MISMATCH";
                    break;
                    
                case CURAND_STATUS_INTERNAL_ERROR:
                    message << "CURAND_STATUS_INTERNAL_ERROR";
                    break;
                    
                default:
                    message << "Undefined error.";
                    break;
            }
            
            message << "\n";
            if ( !file.empty() )
            {
                message << "\t" << file << ":" << line << "\n";
            }
            
            return message.str();
        }
        
};

}

#endif	/* CUDAERRORMESSAGE_H */

