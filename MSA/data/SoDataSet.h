#ifndef SODATASET_H
#define SODATASET_H

namespace MSA
{

class SoDataSet
{
public:

    /**
     * Supported data type
     */
    enum DataType {
        /** unsigned byte */
        UNSIGNED_BYTE  = 0,
        /** unsigned short */
        UNSIGNED_SHORT = 1,
        /** unsigned int (32bits) */
        UNSIGNED_INT32 = 2,
        /** signed byte */
        SIGNED_BYTE  = 4,
        /** signed short */
        SIGNED_SHORT = 5,
        /** signed int (32bits) */
        SIGNED_INT32 = 6,
        /** float */
        FLOAT = 10,
        /** undefined */
        UNDEFINED

    };


    SoDataSet()
    {}


    ~SoDataSet()
    {}

};

}

#endif // SODATASET_H
