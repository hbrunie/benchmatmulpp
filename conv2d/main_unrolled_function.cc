#ifdef USE_ICPC
#include "mkl.h"
#endif
#include <chrono>
#include <iostream>

#define COMPUTE Output[y*Xoutput*F + x*F + f] += Input[(y*strideh+h)*C*(X+W-1) + (x*stridew+w) * C  + c] * K[f * C*H*W + c * H*W + h*W + w];

#define COMPUTE_FULL    for (int f=0; f<F; f++)\
                            for (int c=0; c<C; c++)\
                                for (int x=0; x<Xoutput; x++)\
                                    for (int y=0; y<Youtput; y++)\
                                        for (int w=0; w<W; w++)\
                                            for (int h=0; h<H; h++){\
                                                COMPUTE\
                                            }

#define COMPUTE_ZERO for (int f=0; f<F; f++)\
                        for (int y=0; y<Youtput; y++)\
                            for (int x=0; x<Xoutput; x++)\
                                Output[f*Youtput*Xoutput + y*Xoutput + x] = 0.0;

void display_value(float * outbuf, int size){
    std::cout << outbuf[rand()%size] << "\n";
}

void display_time(std::chrono::time_point<std::chrono::system_clock> t1,  std::chrono::time_point<std::chrono::system_clock>t2, int size,
        int nbthreads, std::string compiler, std::string option, std::string name){

    std::chrono::duration< double > fs = t2 - t1;
    std::chrono::microseconds d = std::chrono::duration_cast< std::chrono::microseconds >( fs );
    //std::cout << fs.count() << "s\n";
    //std::cerr << nbthreads << ";" ;
    std::cerr << compiler << ";" ;
    std::cerr << option << ";" ;
    std::cerr << name  << ";" ;
    std::cerr << d.count() /1000.<< "\n";
}

#ifdef USE_ICPC
__forceinline
#else
inline
#endif
void conv_naive_impl(float const * const __restrict__ K,
        float const * const __restrict__ Input,
        float * const __restrict__ Output,
        const int W, const int H, const int C, const int F, const int X, const int Y,
        const int strideh, const int stridew) {

    int Xoutput = X /strideh;
    int Youtput = Y /stridew;
    COMPUTE_ZERO

    COMPUTE_FULL
    return;
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

typedef std::chrono::high_resolution_clock Clock;
// --- Main function (for testing) ---
int main() {
    srand(0);
    // Problem sizes (MobiNet-1 for the sizes)
    //std::cerr << "nbthreads;compiler;option;conv_name;time(ms);\n";
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

    std::string name = conv_names[0];
    int F = 64;
    int C = 3;
    int Y =224 ;
    int X = Y;
    int H = 7;
    int W = H;
    int strideh = 2;
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
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
#else
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
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
    name = conv_names[1];
    F = 64;
    C = 64;
    Y =56 ;
    X = Y;
    H = 3;
    W = H;
    strideh = 1;
    stridew = strideh;

    size_K = W * H * C * F;
    size_Input = B * (X+W-1) * (Y+H-1) * C;
    Xoutput = X /strideh;
    Youtput = Y /stridew;
    size_Output = Xoutput * Youtput * F;

    // Data preparation - build the input arrays
#ifdef USE_ICPC
    K = (float*) mkl_malloc(sizeof(float) * size_K, 64);
    Input = (float*) mkl_malloc(sizeof(float) * size_Input,64);
    Output = (float*) mkl_malloc(sizeof(float) * size_Output,64);
#else
    K = (float*) malloc(sizeof(float) * size_K);
    Input = (float*) malloc(sizeof(float) * size_Input);
    Output = (float*) malloc(sizeof(float) * size_Output);
#endif

    // Filling the data with random vals
    for (int count=0; count < size_K; count++) {
        int64_t val = (rand() % bound) - 64;
        K[count] = (float) val;
    }
    for (int count=0; count<size_Input; count++) {
        int64_t val = (rand() % bound) - 64;
        Input[count] = (float) val;
    }
#ifdef USE_COLDCACHE
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
#else
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
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

#ifdef MORE_CONV
    name = conv_names[2];
    F = 64;
    C = 64;
    Y =56 ;
    X = Y;
    H = 1;
    W = H;
    strideh = 1;
    stridew = strideh;
    size_K = W * H * C * F;
    size_Input = B * (X+W-1) * (Y+H-1) * C;
    Xoutput = X /strideh;
    Youtput = Y /stridew;
    size_Output = Xoutput * Youtput * F;

    // Data preparation - build the input arrays
#ifdef USE_ICPC
    K = (float*) mkl_malloc(sizeof(float) * size_K, 64);
    Input = (float*) mkl_malloc(sizeof(float) * size_Input,64);
    Output = (float*) mkl_malloc(sizeof(float) * size_Output,64);
#else
    K = (float*) malloc(sizeof(float) * size_K);
    Input = (float*) malloc(sizeof(float) * size_Input);
    Output = (float*) malloc(sizeof(float) * size_Output);
#endif

    // Filling the data with random vals
    for (int count=0; count < size_K; count++) {
        int64_t val = (rand() % bound) - 64;
        K[count] = (float) val;
    }
    for (int count=0; count<size_Input; count++) {
        int64_t val = (rand() % bound) - 64;
        Input[count] = (float) val;
    }
#ifdef USE_COLDCACHE
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
#else
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
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

    name = conv_names[3];
    F = 128;
    C =  64;
    Y =56 ;
    X = Y;
    H = 3;
    W = H;
    strideh = 2;
    stridew = strideh;
    size_K = W * H * C * F;
    size_Input = B * (X+W-1) * (Y+H-1) * C;
    Xoutput = X /strideh;
    Youtput = Y /stridew;
    size_Output = Xoutput * Youtput * F;

    // Data preparation - build the input arrays
#ifdef USE_ICPC
    K = (float*) mkl_malloc(sizeof(float) * size_K, 64);
    Input = (float*) mkl_malloc(sizeof(float) * size_Input,64);
    Output = (float*) mkl_malloc(sizeof(float) * size_Output,64);
#else
    K = (float*) malloc(sizeof(float) * size_K);
    Input = (float*) malloc(sizeof(float) * size_Input);
    Output = (float*) malloc(sizeof(float) * size_Output);
#endif

    // Filling the data with random vals
    for (int count=0; count < size_K; count++) {
        int64_t val = (rand() % bound) - 64;
        K[count] = (float) val;
    }
    for (int count=0; count<size_Input; count++) {
        int64_t val = (rand() % bound) - 64;
        Input[count] = (float) val;
    }
#ifdef USE_COLDCACHE
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
#else
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
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

    name = conv_names[4];
    F = 128;
    C =  64;
    Y =56 ;
    X = Y;
    H = 1;
    W = H;
    strideh = 2;
    stridew = strideh;
    size_K = W * H * C * F;
    size_Input = B * (X+W-1) * (Y+H-1) * C;
    Xoutput = X /strideh;
    Youtput = Y /stridew;
    size_Output = Xoutput * Youtput * F;

    // Data preparation - build the input arrays
#ifdef USE_ICPC
    K = (float*) mkl_malloc(sizeof(float) * size_K, 64);
    Input = (float*) mkl_malloc(sizeof(float) * size_Input,64);
    Output = (float*) mkl_malloc(sizeof(float) * size_Output,64);
#else
    K = (float*) malloc(sizeof(float) * size_K);
    Input = (float*) malloc(sizeof(float) * size_Input);
    Output = (float*) malloc(sizeof(float) * size_Output);
#endif

    // Filling the data with random vals
    for (int count=0; count < size_K; count++) {
        int64_t val = (rand() % bound) - 64;
        K[count] = (float) val;
    }
    for (int count=0; count<size_Input; count++) {
        int64_t val = (rand() % bound) - 64;
        Input[count] = (float) val;
    }
#ifdef USE_COLDCACHE
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
#else
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
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

    name = conv_names[5];
    F = 128;
    C =  128;
    Y =28 ;
    X = Y;
    H = 3;
    W = H;
    strideh = 1;
    stridew = strideh;
    size_K = W * H * C * F;
    size_Input = B * (X+W-1) * (Y+H-1) * C;
    Xoutput = X /strideh;
    Youtput = Y /stridew;
    size_Output = Xoutput * Youtput * F;

    // Data preparation - build the input arrays
#ifdef USE_ICPC
    K = (float*) mkl_malloc(sizeof(float) * size_K, 64);
    Input = (float*) mkl_malloc(sizeof(float) * size_Input,64);
    Output = (float*) mkl_malloc(sizeof(float) * size_Output,64);
#else
    K = (float*) malloc(sizeof(float) * size_K);
    Input = (float*) malloc(sizeof(float) * size_Input);
    Output = (float*) malloc(sizeof(float) * size_Output);
#endif

    // Filling the data with random vals
    for (int count=0; count < size_K; count++) {
        int64_t val = (rand() % bound) - 64;
        K[count] = (float) val;
    }
    for (int count=0; count<size_Input; count++) {
        int64_t val = (rand() % bound) - 64;
        Input[count] = (float) val;
    }
#ifdef USE_COLDCACHE
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
#else
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
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

    name = conv_names[6];
    F = 256;
    C =  128;
    Y =28 ;
    X = Y;
    H = 3;
    W = H;
    strideh = 2;
    stridew = strideh;
    size_K = W * H * C * F;
    size_Input = B * (X+W-1) * (Y+H-1) * C;
    Xoutput = X /strideh;
    Youtput = Y /stridew;
    size_Output = Xoutput * Youtput * F;

    // Data preparation - build the input arrays
#ifdef USE_ICPC
    K = (float*) mkl_malloc(sizeof(float) * size_K, 64);
    Input = (float*) mkl_malloc(sizeof(float) * size_Input,64);
    Output = (float*) mkl_malloc(sizeof(float) * size_Output,64);
#else
    K = (float*) malloc(sizeof(float) * size_K);
    Input = (float*) malloc(sizeof(float) * size_Input);
    Output = (float*) malloc(sizeof(float) * size_Output);
#endif

    // Filling the data with random vals
    for (int count=0; count < size_K; count++) {
        int64_t val = (rand() % bound) - 64;
        K[count] = (float) val;
    }
    for (int count=0; count<size_Input; count++) {
        int64_t val = (rand() % bound) - 64;
        Input[count] = (float) val;
    }
#ifdef USE_COLDCACHE
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
#else
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
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

    name = conv_names[7];
    F = 256;
    C =  128;
    Y =28 ;
    X = Y;
    H = 3;
    W = H;
    strideh = 1;
    stridew = strideh;
    size_K = W * H * C * F;
    size_Input = B * (X+W-1) * (Y+H-1) * C;
    Xoutput = X /strideh;
    Youtput = Y /stridew;
    size_Output = Xoutput * Youtput * F;

    // Data preparation - build the input arrays
#ifdef USE_ICPC
    K = (float*) mkl_malloc(sizeof(float) * size_K, 64);
    Input = (float*) mkl_malloc(sizeof(float) * size_Input,64);
    Output = (float*) mkl_malloc(sizeof(float) * size_Output,64);
#else
    K = (float*) malloc(sizeof(float) * size_K);
    Input = (float*) malloc(sizeof(float) * size_Input);
    Output = (float*) malloc(sizeof(float) * size_Output);
#endif

    // Filling the data with random vals
    for (int count=0; count < size_K; count++) {
        int64_t val = (rand() % bound) - 64;
        K[count] = (float) val;
    }
    for (int count=0; count<size_Input; count++) {
        int64_t val = (rand() % bound) - 64;
        Input[count] = (float) val;
    }
#ifdef USE_COLDCACHE
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
#else
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
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

    name = conv_names[8];
    F = 256;
    C =  256;
    Y =14 ;
    X = Y;
    H = 3;
    W = H;
    strideh = 1;
    stridew = strideh;
    size_K = W * H * C * F;
    size_Input = B * (X+W-1) * (Y+H-1) * C;
    Xoutput = X /strideh;
    Youtput = Y /stridew;
    size_Output = Xoutput * Youtput * F;

    // Data preparation - build the input arrays
#ifdef USE_ICPC
    K = (float*) mkl_malloc(sizeof(float) * size_K, 64);
    Input = (float*) mkl_malloc(sizeof(float) * size_Input,64);
    Output = (float*) mkl_malloc(sizeof(float) * size_Output,64);
#else
    K = (float*) malloc(sizeof(float) * size_K);
    Input = (float*) malloc(sizeof(float) * size_Input);
    Output = (float*) malloc(sizeof(float) * size_Output);
#endif

    // Filling the data with random vals
    for (int count=0; count < size_K; count++) {
        int64_t val = (rand() % bound) - 64;
        K[count] = (float) val;
    }
    for (int count=0; count<size_Input; count++) {
        int64_t val = (rand() % bound) - 64;
        Input[count] = (float) val;
    }
#ifdef USE_COLDCACHE
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
#else
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
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

    name = conv_names[9];
    F = 512;
    C =  512;
    Y =14 ;
    X = Y;
    H = 3;
    W = H;
    strideh = 2;
    stridew = strideh;
    size_K = W * H * C * F;
    size_Input = B * (X+W-1) * (Y+H-1) * C;
    Xoutput = X /strideh;
    Youtput = Y /stridew;
    size_Output = Xoutput * Youtput * F;

    // Data preparation - build the input arrays
#ifdef USE_ICPC
    K = (float*) mkl_malloc(sizeof(float) * size_K, 64);
    Input = (float*) mkl_malloc(sizeof(float) * size_Input,64);
    Output = (float*) mkl_malloc(sizeof(float) * size_Output,64);
#else
    K = (float*) malloc(sizeof(float) * size_K);
    Input = (float*) malloc(sizeof(float) * size_Input);
    Output = (float*) malloc(sizeof(float) * size_Output);
#endif

    // Filling the data with random vals
    for (int count=0; count < size_K; count++) {
        int64_t val = (rand() % bound) - 64;
        K[count] = (float) val;
    }
    for (int count=0; count<size_Input; count++) {
        int64_t val = (rand() % bound) - 64;
        Input[count] = (float) val;
    }
#ifdef USE_COLDCACHE
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
#else
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
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

    name = conv_names[10];
    F = 512;
    C =  256;
    Y =14 ;
    X = Y;
    H = 1;
    W = H;
    strideh = 2;
    stridew = strideh;
    size_K = W * H * C * F;
    size_Input = B * (X+W-1) * (Y+H-1) * C;
    Xoutput = X /strideh;
    Youtput = Y /stridew;
    size_Output = Xoutput * Youtput * F;

    // Data preparation - build the input arrays
#ifdef USE_ICPC
    K = (float*) mkl_malloc(sizeof(float) * size_K, 64);
    Input = (float*) mkl_malloc(sizeof(float) * size_Input,64);
    Output = (float*) mkl_malloc(sizeof(float) * size_Output,64);
#else
    K = (float*) malloc(sizeof(float) * size_K);
    Input = (float*) malloc(sizeof(float) * size_Input);
    Output = (float*) malloc(sizeof(float) * size_Output);
#endif

    // Filling the data with random vals
    for (int count=0; count < size_K; count++) {
        int64_t val = (rand() % bound) - 64;
        K[count] = (float) val;
    }
    for (int count=0; count<size_Input; count++) {
        int64_t val = (rand() % bound) - 64;
        Input[count] = (float) val;
    }
#ifdef USE_COLDCACHE
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
#else
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
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

    name = conv_names[11];
    F = 512;
    C =  512;
    Y =7  ;
    X = Y;
    H = 3;
    W = H;
    strideh = 1;
    stridew = strideh;
    size_K = W * H * C * F;
    size_Input = B * (X+W-1) * (Y+H-1) * C;
    Xoutput = X /strideh;
    Youtput = Y /stridew;
    size_Output = Xoutput * Youtput * F;

    // Data preparation - build the input arrays
#ifdef USE_ICPC
    K = (float*) mkl_malloc(sizeof(float) * size_K, 64);
    Input = (float*) mkl_malloc(sizeof(float) * size_Input,64);
    Output = (float*) mkl_malloc(sizeof(float) * size_Output,64);
#else
    K = (float*) malloc(sizeof(float) * size_K);
    Input = (float*) malloc(sizeof(float) * size_Input);
    Output = (float*) malloc(sizeof(float) * size_Output);
#endif

    // Filling the data with random vals
    for (int count=0; count < size_K; count++) {
        int64_t val = (rand() % bound) - 64;
        K[count] = (float) val;
    }
    for (int count=0; count<size_Input; count++) {
        int64_t val = (rand() % bound) - 64;
        Input[count] = (float) val;
    }
#ifdef USE_COLDCACHE
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
#else
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t1 = Clock::now();
    conv_naive_impl(K, Input, Output, W, H, C, F, X, Y, strideh, stridew);
    t2 = Clock::now();
    display_time(t1, t2, size_Output, nbthreads, compiler, option,name);
    display_value(Output, size_Output);
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
#endif



    return 0;
}
