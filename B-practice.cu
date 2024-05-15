#include <cuda.h>
#include <stdlib.h>
#include "util.h"

// #define PRINT_INFO
using namespace cute;


template <typename T>
void gen_rand_data(T *data, int n)
{
  for (int i = 0; i < n; ++i)
  {
    float v = (rand() % 200 - 100) * 0.01;
    data[i] = v;
  }
}


template <typename T,int kTileM,int kTileN,int kTileK >
__global__ void gemm_simple(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k) {

  int tx = threadIdx.x;


  int A_offset = (blockIdx.x * k * kTileM);
  int B_offset = (blockIdx.y * k * kTileN);

  extern __shared__ float sram[];

  float* tile_A=sram;
  float* tile_B=&sram[kTileM * k];
  float* tile_C=&sram[kTileM * k + kTileN * k];

  //load tile_A,tile_B
  for(int x=0;x<k;x++){
    tile_A[(tx*k)+x] = Aptr[A_offset + (tx * k) + x];
    tile_B[(tx*k)+x] = Bptr[B_offset + (tx * k) + x];
  }
  __syncthreads();

  #pragma unroll
  for(int i=0;i<kTileN;i++){
    float sum =0;
     for(int j=0;j<k;j++){
      sum += tile_A[(tx * k)+j] * tile_B[(i * k)+j];
     }
     tile_C[(tx * kTileN) + i]=sum;
  }

    #pragma unroll
    for(int i=0;i<kTileN;i++){
        Cptr[(tx * kTileN)+ i ]= tile_C[(tx * kTileN) + i];
    }

  __syncthreads();
     printf("Thread %d:", tx);

}



int main(){
srand(1000);

  using T = cute::half_t;
  cudaEvent_t start, end;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  T *Cptr;
  T *Aptr;
  T *Bptr;

  int m = 1024*64;
  int n = 128;
  int k = 256;

  cudaMalloc(&Cptr, sizeof(T) * m * n);
  cudaMalloc(&Aptr, sizeof(T) * m * k);
  cudaMalloc(&Bptr, sizeof(T) * k * n);

  T *Aptr_host;
  T *Bptr_host;
  Aptr_host = (T *)malloc(sizeof(T) * m * k);
  Bptr_host = (T *)malloc(sizeof(T) * n * k);
  gen_rand_data(Aptr_host, m * k);
  gen_rand_data(Bptr_host, n * k);

  cudaMemcpy(Aptr, Aptr_host, sizeof(T) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(Bptr, Bptr_host, sizeof(T) * n * k, cudaMemcpyHostToDevice);


  constexpr int kTileM = 32;
  constexpr int kTileN = 32;
  constexpr int kTileK = 32;



  dim3 grid(n / kTileN, m / kTileM);
  dim3 block(kTileM);
  //const int sram
  int count = 100;
  const int sram_size =(kTileM * k + kTileN * k + kTileM*kTileN) * sizeof(T) ;
     int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);
  cudaEventRecord(start);
  for (int i = 0; i < count; ++i)
  {
    gemm_simple<T, kTileM, kTileN, kTileK><<<grid,block,sram_size>>>(Cptr, Aptr, Bptr, m, n, k);
  }
  auto err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsedTime, start, end);
  std::cout << "gemm-simple took " << elapsedTime / count << "ms." << std::endl;
}