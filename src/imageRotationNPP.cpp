/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

//#include <Exceptions.h>
//#include <ImageIO.h>
//#include <ImagesCPU.h>
//#include <ImagesNPP.h>
#include <FreeImage.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <npp.h>
#include <cassert>
#include <iostream>
#include <stdexcept>

#include <helper_cuda.h>
#include <helper_string.h>


bool printfNPPinfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
           libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
           (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
           (runtimeVersion % 100) / 10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

FIBITMAP *LoadImg(const char *szFile)
{
    FREE_IMAGE_FORMAT nFif;

    if (szFile == NULL || *szFile == 0)
    {
        return NULL;
    }

    if ((nFif = FreeImage_GetFileType(szFile, 0)) == FIF_UNKNOWN)
    {
        if ((nFif = FreeImage_GetFIFFromFilename(szFile)) == FIF_UNKNOWN)
        {
            return NULL;
        }
    }

    if (!FreeImage_FIFSupportsReading(nFif))
    {
        return NULL;
    }

    return FreeImage_Load(nFif, szFile);
}

int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);
    double aBox[2][2];
    try
    {
        std::string sFilename;
        char *filePath;

        findCudaDevice(argc, (const char **)argv);

        if (printfNPPinfo(argc, argv) == false)
        {
            exit(EXIT_SUCCESS);
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "input"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
        }
        else
        {
            filePath = sdkFindFilePath("Lena.pgm", argv[0]);
        }

        if (filePath)
        {
            sFilename = filePath;
        }
        else
        {
            sFilename = "Lena.pgm";
        }

        // if we specify the filename at the command line, then we only test
        // sFilename[0].
        int file_errors = 0;
        std::ifstream infile(sFilename.data(), std::ifstream::in);

        if (infile.good())
        {
            std::cout << "nppiRotate opened: <" << sFilename.data()
                      << "> successfully!" << std::endl;
            file_errors = 0;
            infile.close();
        }
        else
        {
            std::cout << "nppiRotate unable to open: <" << sFilename.data() << ">"
                      << std::endl;
            file_errors++;
            infile.close();
        }

        if (file_errors > 0)
        {
            exit(EXIT_FAILURE);
        }

        std::string sResultFilename = sFilename;

        std::string::size_type dot = sResultFilename.rfind('.');

        if (dot != std::string::npos)
        {
            sResultFilename = sResultFilename.substr(0, dot);
        }

        sResultFilename += "_rotate.pgm";
        

        if (checkCmdLineFlag(argc, (const char **)argv, "output"))
        {
            char *outputFilePath;
            getCmdLineArgumentString(argc, (const char **)argv, "output",
                                     &outputFilePath);
            sResultFilename = outputFilePath;
        }

        // declare a host image object for an 8-bit grayscale image
        cudaError_t cuRet;
        NppStatus nppRet;
        BOOL fiRet = false;
        FIBITMAP *pSrcBmp = NULL;
        FIBITMAP *pDstBmp = NULL;
        unsigned char *pSrcData = NULL;
        unsigned char *pDstData = NULL;
        Npp8u *pSrcDataCUDA = NULL;
        Npp8u *pDstDataCUDA = NULL;
        NppiSize oSrcSize = {0};
        NppiSize oDstSize = {0};
        NppiRect oSrcROI = {0};
        NppiRect oDstROI = {0};
        int nImgBpp = 0;
        int nSrcPitch = 0;
        int nDstPitch = 0;
        int nSrcPitchCUDA = 0;
        int nDstPitchCUDA = 0;
        double aBoundingBox[2][2] = {0};
        double nAngle = 0;

        FreeImage_Initialise(0);

        
        pSrcBmp = LoadImg(sFilename.c_str());
        assert(pSrcBmp != NULL);

        
        nImgBpp = (FreeImage_GetBPP(pSrcBmp) >> 3);
        std::cout << "freeImg nImgBpp :" << nImgBpp << std::endl;
        
        pSrcData = FreeImage_GetBits(pSrcBmp);

        oSrcSize.width = (int)FreeImage_GetWidth(pSrcBmp);
        std::cout << "freeImg oSrcSize.width :" << oSrcSize.width << std::endl;
        oSrcSize.height = (int)FreeImage_GetHeight(pSrcBmp);
        std::cout << "freeImg oSrcSize.height :" << oSrcSize.height << std::endl;
        nSrcPitch = (int)FreeImage_GetPitch(pSrcBmp);
        std::cout << "freeImg nSrcPitch :" << nSrcPitch << std::endl;

        oSrcROI.x = oSrcROI.y = 0;
        oSrcROI.width = oSrcSize.width;
        oSrcROI.height = oSrcSize.height;

        nAngle = atof("90");

        
        cuRet = cudaSetDevice(0);
        assert(cuRet == cudaSuccess);

        
        int type = 0;
        switch (nImgBpp)
        {
        case 1:
            pSrcDataCUDA = nppiMalloc_8u_C1(oSrcSize.width, oSrcSize.height, &nSrcPitchCUDA);
            break;
        case 3:
            pSrcDataCUDA = nppiMalloc_8u_C3(oSrcSize.width, oSrcSize.height, &nSrcPitchCUDA);
            break;
        case 4:
            pSrcDataCUDA = nppiMalloc_8u_C4(oSrcSize.width, oSrcSize.height, &nSrcPitchCUDA);
            break;
        default:
            assert(0);
            break;
        }
        assert(pSrcDataCUDA != NULL);

        
        cudaMemcpy2D(pSrcDataCUDA, nSrcPitchCUDA, pSrcData, nSrcPitch, oSrcSize.width * nImgBpp, oSrcSize.height, cudaMemcpyHostToDevice);

        
        nppiGetRotateBound(oSrcROI, aBoundingBox, nAngle, 0, 0);
        oDstSize.width = (int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0]));
        oDstSize.height = (int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1]));

        
        pDstBmp = FreeImage_Allocate(oDstSize.width, oDstSize.height, nImgBpp << 3);
        assert(pDstBmp != NULL);

        pDstData = FreeImage_GetBits(pDstBmp);

        nDstPitch = (int)FreeImage_GetPitch(pDstBmp);
        std::cout << "freeImg nDstPitch :" << nDstPitch << std::endl;
        oDstROI.x = oDstROI.y = 0;
        oDstROI.width = oDstSize.width;
        std::cout << "freeImg  oDstSize.width :" << oDstSize.width << std::endl;
        oDstROI.height = oDstSize.height;
        std::cout << "freeImg oDstSize.height :" << oDstSize.height << std::endl;

        
        switch (nImgBpp)
        {
        case 1:
            pDstDataCUDA = nppiMalloc_8u_C1(oDstSize.width, oDstSize.height, &nDstPitchCUDA);
            break;
        case 3:
            pDstDataCUDA = nppiMalloc_8u_C3(oDstSize.width, oDstSize.height, &nDstPitchCUDA);
            break;
        case 4:
            pDstDataCUDA = nppiMalloc_8u_C4(oDstSize.width, oDstSize.height, &nDstPitchCUDA);
            break;
        }
        assert(pDstDataCUDA != NULL);
        cudaMemset2D(pDstDataCUDA, nDstPitchCUDA, 0, oDstSize.width * nImgBpp, oDstSize.height);

        
        switch (nImgBpp)
        {
        case 1:
            nppRet = nppiRotate_8u_C1R(pSrcDataCUDA, oSrcSize, nSrcPitchCUDA, oSrcROI,
                                       pDstDataCUDA, nDstPitchCUDA, oDstROI,
                                       nAngle, -aBoundingBox[0][0], -aBoundingBox[0][1], NPPI_INTER_CUBIC);
            break;
        case 3:
            nppRet = nppiRotate_8u_C3R(pSrcDataCUDA, oSrcSize, nSrcPitchCUDA, oSrcROI,
                                       pDstDataCUDA, nDstPitchCUDA, oDstROI,
                                       nAngle, -aBoundingBox[0][0], -aBoundingBox[0][1], NPPI_INTER_CUBIC);
            break;
        case 4:
            nppRet = nppiRotate_8u_C4R(pSrcDataCUDA, oSrcSize, nSrcPitchCUDA, oSrcROI,
                                       pDstDataCUDA, nDstPitchCUDA, oDstROI,
                                       nAngle, -aBoundingBox[0][0], -aBoundingBox[0][1], NPPI_INTER_CUBIC);
            break;
        }
        assert(nppRet == NPP_NO_ERROR);

        cudaMemcpy2D(pDstData, nDstPitch, pDstDataCUDA, nDstPitchCUDA, oDstSize.width * nImgBpp, oDstSize.height, cudaMemcpyDeviceToHost);

        fiRet = FreeImage_Save(FIF_BMP, pDstBmp, sResultFilename.c_str());
        assert(fiRet);

        nppiFree(pSrcDataCUDA);
        nppiFree(pDstDataCUDA);

        cudaDeviceReset();

        FreeImage_Unload(pSrcBmp);
        FreeImage_Unload(pDstBmp);
        exit(EXIT_SUCCESS);
    }
    /*catch (const exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }*/
    catch (...)
    {
        std::cerr << "Program error! An unknown type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        return -1;
    }

    return 0;
}

