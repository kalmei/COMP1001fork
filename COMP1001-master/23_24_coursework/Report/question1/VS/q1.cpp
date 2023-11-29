/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP1001 ------------------------------------------------------------------
------------------COMPUTER SYSTEMS MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/


#include <stdio.h>
#include <time.h>
#include <pmmintrin.h>
#include <process.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>
#include <omp.h>

#define M 1024*512
#define ARITHMETIC_OPERATIONS1 3*M
#define TIMES1 1

#define N 8192
#define ARITHMETIC_OPERATIONS2 4*N*N
#define TIMES2 1


//function declaration
void initialize();
void routine1(float alpha, float beta);
void routine2(float alpha, float beta);

__declspec(align(64)) float  y[M], z[M] ;
__declspec(align(64)) float A[N][N], x[N], w[N];

int main() {

    float alpha = 0.023f, beta = 0.045f;
    double run_time, start_time;
    unsigned int t;

    initialize();

    printf("\nRoutine1:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES1; t++)
        routine1(alpha, beta);
        //routine1_vec(alpha, beta);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS1) / ((double)run_time / TIMES1));

    printf("\nRoutine2:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES2; t++)
        routine2(alpha, beta);
        //routine2_vec(alpha, beta);
    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS2) / ((double)run_time / TIMES2));



    return 0;
}

void initialize() {

    unsigned int i, j;

    //initialize routine2 arrays
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = (i % 99) + (j % 14) + 0.013f;
        }

    //initialize routine1 arrays
    for (i = 0; i < N; i++) {
        x[i] = (i % 19) - 0.01f;
        w[i] = (i % 5) - 0.002f;
    }

    //initialize routine1 arrays
    for (i = 0; i < M; i++) {
        z[i] = (i % 9) - 0.08f;
        y[i] = (i % 19) + 0.07f;
    }


}




void routine1(float alpha, float beta) {

    unsigned int i;


    for (i = 0; i < M; i++)
        y[i] = alpha * y[i] + beta * z[i];

}

void routine2(float alpha, float beta) {

    unsigned int i, j;


    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            w[i] = w[i] - beta + alpha * A[i][j] * x[j];


}

routine1_vec(float alpha, float beta) {
    unsigned int i = 0;
    __m128 num1, num2, num3, num4,num5,num6,num7;
    num3 = _mm_setzero_ps();
    num6 = _mm_loadu_ps(&alpha);
    num7 = _mm_loadu_ps(&beta);
    for (i = 0; i < (M/4)*4; i+=4) {
        num1 = _mm_loadu_ps(&(y[i]));
        num2 = _mm_loadu_ps(&(z[i]));
        num3 = _mm_mul_ps(num1, num6);
        num4 = _mm_mul_ps(num2, num7);
        num5 = _mm_add_ps(num3, num4);
        _mm_storeu_ps(&y[i], num5);
    }
    for ( ; i < M; i++) {
    y[i] = alpha*y[i] + beta*z[i];
    }
}

routine2_vec(float alpha, float beta) {
    unsigned int i = 0; 
    unsigned int j = 0;
    __m256 num1, num2, num3 ,num4, num5, num6, num7, num8, num9;
    num4 = _mm256_setzero_ps();
    num8 = _mm256_loadu_ps(&alpha);
    num9 = _mm256_loadu_ps(&beta);
    for (i =0 ; i < N; i++) {
    for ( j= 0; j < (N/8)*8; j +=8) {
    num1 = _mm256_loadu_ps(&w[i]);
    num2 = _mm256_loadu_ps(&A[i][j]);
    num3 = _mm256_loadu_ps(&x[j]);
    num4 = _mm256_sub_ps(num1, num9);
    num5 = _mm256_mul_ps(num2, num3);
    num6 = _mm256_mul_ps(num5, num8);
    num7 = _mm256_add_ps(num4,num6);
    _mm256_storeu_ps(&(w[i]), num7);
    }
    for ( ; j < N; j++) {
    w[i] = w[i]*beta - alpha*A[i][j]*x[j];
    }
    }
}
