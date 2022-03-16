#ifdef USE_ICPC
#include "mkl.h"
#endif
#include <chrono>
#include <iostream>

#include "common.hpp"

#define COMPUTE Output[y*Xoutput*F + x*F + f] += Input[(y*strideh+h)*C*(X+W-1) + (x*stridew+w) * C  + c] * K[h * C*F*W + w * C*F + c*F + f];

#define COMPUTE_FULL   for (int x=0; x<Xoutput; x++)\
                                   for (int y=0; y<Youtput; y++)\
                                   for (int h=0; h<H; h++){\
                                   for (int w=0; w<W; w++)\
                                   for (int c=0; c<C; c++)\
                                   for (int f=0; f<F; f++)\
                                       COMPUTE\
                                   }

#define COMPUTE_ZERO for (int f=0; f<F; f++)\
                                for (int y=0; y<Youtput; y++)\
                                for (int x=0; x<Xoutput; x++)\
                                Output[y*Xoutput*F + x*F + f] = 0.0;

typedef std::chrono::high_resolution_clock Clock;

template<int F,int C,int Y,int H, int strideh>
void conv_naive_impl(float const * const __restrict__ K,
        float const * const __restrict__ Input,
        float * const __restrict__ Output){
    int X = Y;
    int W = H;
    int stridew = strideh;

    int Xoutput = X /strideh;
    int Youtput = Y /stridew;
    COMPUTE_ZERO
        COMPUTE_FULL
        return;
}


template<int F,int C,int Y,int H, int strideh>
void wrapper(std::string name){
    int nbthreads = 1;
#ifdef USE_POLLY
    std::string compiler = "POLLY";
#endif
#ifdef USE_LLVM
    std::string compiler = "LLVM";
#endif
#ifdef USE_ICPC
    std::string compiler = "INTEL";
#endif
#ifdef USE_GNU
    std::string compiler = "GNU";
#endif
    std::string option = "";
#ifdef USE_COLDCACHE
    option += " COLD_CACHE";
#else
    option += " HOT_CACHE";
#endif
    // Fixed
    const int B = 1;
    int X = Y;
    int W = H;
    int stridew = strideh;

    int size_K = W * H * C * F;
    int size_Input = B * (X+W-1) * (Y+H-1) * C;
    int Xoutput = X /strideh;
    int Youtput = Y /stridew;
    int size_Output = Xoutput * Youtput * F;

    // Data preparation - build the input arrays
#ifdef USE_ICPC
    float* K = (float*) mkl_malloc(sizeof(float) * size_K, 64);
    float* Input = (float*) mkl_malloc(sizeof(float) * size_Input,64);
    float* Output = (float*) mkl_malloc(sizeof(float) * size_Output,64);
#else
    float* K = (float*) malloc(sizeof(float) * size_K);
    float* Input = (float*) malloc(sizeof(float) * size_Input);
    float* Output = (float*) malloc(sizeof(float) * size_Output);
#endif

    int64_t bound = 128;		// For bounded random generation
    // Filling the data with random vals
    for (int count=0; count < size_K; count++) {
        int64_t val = (rand() % bound) - 64;
        K[count] = (float) val;
    }
    for (int count=0; count<size_Input; count++) {
        int64_t val = (rand() % bound) - 64;
        Input[count] = (float) val;
    }

    auto t1=Clock::now();
    auto t2=Clock::now();
#ifdef USE_COLDCACHE
    t1 = Clock::now();
    conv_naive_impl<F,C,Y,H,strideh>(K, Input, Output);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
    //std::cerr << "COLD CACHE" << std::endl;
#else
    conv_naive_impl<F,C,Y,H,strideh>(K, Input, Output);
    conv_naive_impl<F,C,Y,H,strideh>(K, Input, Output);
    t1 = Clock::now();
    conv_naive_impl<F,C,Y,H,strideh>(K, Input, Output);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
    //std::cerr << "HOT CACHE" << std::endl;
#endif

#ifdef USE_ICPC
    mkl_free(K);
    mkl_free(Input);
    mkl_free(Output);
#else
    free(K);
    free(Input);
    free(Output);
#endif
}

std::string conv_names[12] = {"ResNet18_01",
    "ResNet18_02",
    "ResNet18_03",
    "ResNet18_04",
    "ResNet18_05",
    "ResNet18_06",
    "ResNet18_07",
    "ResNet18_08",
    "ResNet18_09",
    "ResNet18_10",
    "ResNet18_11",
    "ResNet18_12"};

// Cout, Cin, Y, X, h, w, sh, sw
// K,C,H,W,R,S,StrideH,StrideW
int conv_sizes[12][8] = {
    {64, 3,    224, 224, 7, 7, 2, 2},
    {64, 64,   56, 56,   3, 3, 1, 1},
    {64, 64,   56, 56,   1, 1, 1, 1},
    {128, 64,  56, 56,   3, 3, 2, 2},
    {128, 64,  56, 56,   1, 1, 2, 2},
    {128, 128, 28, 28,   3, 3, 1, 1},
    {256, 128, 28, 28,   3, 3, 2, 2},
    {256, 128, 28, 28,   3, 3, 1, 1},
    {256, 256, 14, 14,   3, 3, 1, 1},
    {512, 512, 14, 14,   3, 3, 2, 2},
    {512, 256, 14, 14,   1, 1, 2, 2},
    {512, 512, 7, 7,     3, 3, 1, 1}};

// --- Main function (for testing) ---
int main() {
    srand(0);


    std::string name = conv_names[0];
    {
        const int F = 64;
        const int C = 3;
        const int Y =224 ;
        const int H = 7;
        const int strideh = 2;
        wrapper<F,C,Y,H,strideh>(name);
    }

#ifdef MORE_CONV
    {
    name = conv_names[1];
    const int F = 64;
    const int C = 64;
    const int Y =56 ;
    const int H = 3;
    const int strideh = 1;
        wrapper<F,C,Y,H,strideh>(name);
    }

    {
    name = conv_names[2];
    const int F = 64;
    const int C = 64;
    const int Y =56 ;
    const int H = 1;
    const int strideh = 1;
        wrapper<F,C,Y,H,strideh>(name);
    }

    {
    name = conv_names[3];
    const int F = 128;
    const int C =  64;
    const int Y =56 ;
    const int H = 3;
    const int strideh = 2;
    wrapper<F,C,Y,H,strideh>(name);
    }

    {
    name = conv_names[4];
    const int F = 128;
    const int C =  64;
    const int Y =56 ;
    const int H = 1;
    const int strideh = 2;
    wrapper<F,C,Y,H,strideh>(name);
    }
    {
    name = conv_names[5];
    const int F = 128;
    const int C =  128;
    const int Y =28 ;
    const int H = 3;
    const int strideh = 1;
    wrapper<F,C,Y,H,strideh>(name);
    }

    {
    name = conv_names[6];
    const int F = 256;
    const int C =  128;
    const int Y =28 ;
    const int H = 3;
    const int strideh = 2;
    wrapper<F,C,Y,H,strideh>(name);
    }

    {
    name = conv_names[7];
    const int F = 256;
    const int C =  128;
    const int Y =28 ;
    const int H = 3;
    const int strideh = 1;
    wrapper<F,C,Y,H,strideh>(name);
    }

    {
    name = conv_names[8];
    const int F = 256;
    const int C =  256;
    const int Y =14 ;
    const int H = 3;
    const int strideh = 1;
    wrapper<F,C,Y,H,strideh>(name);
    }

    {
    name = conv_names[9];
    const int F = 512;
    const int C =  512;
    const int Y =14 ;
    const int H = 3;
    const int strideh = 2;
    wrapper<F,C,Y,H,strideh>(name);
    }

    {
    name = conv_names[10];
    const int F = 512;
    const int C =  256;
    const int Y =14 ;
    const int H = 1;
    const int strideh = 2;
    wrapper<F,C,Y,H,strideh>(name);
    }

    {
    name = conv_names[11];
    const int F = 512;
    const int C =  512;
    const int Y =7  ;
    const int H = 3;
    const int strideh = 1;
    wrapper<F,C,Y,H,strideh>(name);
    }
#endif



    return 0;
}
