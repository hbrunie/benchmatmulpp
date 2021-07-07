#include <iostream>
#include <chrono>
#include <mkl.h>

using prec = float;
#define CACHE_LINE_SIZE 64
#define HEAT_CACHE 10
#define ITERATION_COUNT 10000

double time_peak_perf(int m, int n, int k, double freq, int nbfma, int vec_size)
{
    double time = 0.;
    // Nb operations FMA (m*n*k) divided by nb FMA on CPU * freq
    // Assumes 1 FMA is one cycle.
    time =  ((double) m*n*k)  / (nbfma*freq*vec_size);

    return time;
}

template <typename T>
void init_matrix(T*A,T*B,T*C,int m,int n, int k)
{
    for (int i = 0; i < (m*k); i++) {
        A[i] = (T)(i+1);
    }

    for (int i = 0; i < (k*n); i++) {
        B[i] = (T)(-i-1);
    }

    for (int i = 0; i < (m*n); i++) {
        C[i] = 0.0;
    }
}

template <typename T>
int matmul(T * A, T* B, T*C,int m, int n, int k, T alpha, T beta){
    if (A == NULL || B == NULL || C == NULL) {
      printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      mkl_free(A);
      mkl_free(B);
      mkl_free(C);
      return 1;
    }
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    init_matrix(A,B,C,m,n,k);
    // Heating cache
    for (int j=0; j<HEAT_CACHE; j++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, k, B, n, beta, C, n);
    // Measuring time

    auto t1 = high_resolution_clock::now();
    for(int j=0; j<ITERATION_COUNT; j++ )
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, k, B, n, beta, C, n);
    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    //auto ms_int = duration_cast<milliseconds>(t2 - t1);
    duration<double, std::milli> ms_double = t2 - t1;
    int nbfma = 2;
    int vec_size = 16;
    double freq = 2.1E9 / 1000.; //in ms
    double time_pp = time_peak_perf(m,n,k,freq,nbfma,vec_size);//in ms
    double time_measured = ms_double.count() / ((double)ITERATION_COUNT);
    double peak_perf = time_pp / time_measured;
    std::cerr << m << "," << n << "," <<
       k << ","
       << time_measured << ","
       << time_pp << ","
       << peak_perf*100 << std::endl;
    /* Getting number of milliseconds as a double. */

    //std::cout << ms_double.count() << "ms";
    return 0;
}

int main(int ac, char * av[]){
    (void) ac;
    (void) av;
    prec *A, *B, *C;
    int n, k;
    prec alpha, beta;
    k = 128;
    alpha = 1.0; beta = 0.0;
    std::cerr << "m,n,k,time(ms),time_pp(ms),peakperf(%)\n" ;
    for(int n = 128 ; n < 129; n++)
    for (int m = 8 ; m < 52; m ++)
    {
        A = (prec *)mkl_malloc( m*k*sizeof( prec ), CACHE_LINE_SIZE );
        B = (prec *)mkl_malloc( k*n*sizeof( prec ), CACHE_LINE_SIZE );
        C = (prec *)mkl_malloc( m*n*sizeof( prec ), CACHE_LINE_SIZE );

        matmul(A,B,C,m,n,k, alpha, beta);
        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
    }
    return 0;
}
