#include <cuda.h>
#include <stdlib.h>
#include "util.h"

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


template <typename T,int kTileM,int kTileN,int ktileK,typename TiledMMA>
__global__ void gemm_simple_cute(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k) {

    //进行定义一个大的tensor 这个tensor的主要目的是为了后续的分块放入全局内存。
    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{})); 

    int ix = blockIdx.x;
    int iy = blockIdx.y;
    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
    Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
    //  gA(kTileM, kTileK, num_tile_k)
    //  gB(kTileN, kTileK, num_tile_k)
    //  gC(kTileM, kTileN) 

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tAgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K, num_tile_k)
    auto tBgB = thr_mma.partition_B(gB);  // (MMA, MMA_N, MMA_K, num_tile_k)
    auto tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)
    

    //移动到寄存器
    auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
    auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
    auto tCrC = thr_mma.partition_fragment_C(gC(_, _));     // (MMA, MMA_M, MMA_N)
    clear(tcrc);

    int num_tile_k=size<2>(gA);
    for(int itile = 0; itile < num_tile_k; ++itle) {
    cute::copy(tAgA(_, _, _, itile), tArA);
    cute::copy(tBgB(_, _, _, itile), tBrB);
    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }


    cute::copy(tCrC, tCgC); 


}


template <typename T,int kTileM,int kTileN,int ktileK >
__global__ void gemm_simple(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k) {

  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int by = blockIdx.y;


  int A_offset = (blockIdx.x * k * kTileM);
  int B_offset = (blockIdx.y * k * kTileN);

  extren__shared__float sram[];

  float* tile_A=sram;
  float* tile_B=&sram[kTileM * k];
  float* tile_C=&sram[kTileM * k + kTileN * k];


  for(int x=0;x<k;x++){
    tile_A[(tx*k)+x] = A[A_offset + (tx * k) + x];
    tile_B[(tx*k)+x] = B[B_offset + (tx * k) + x];
  }

  __syncthreads();

  #pragma unroll
  for(int i=0;i<kTileN;i++){
    float sum =0;
     for(int j=0;j<k;j++){
      sum += tile_A[(tx*k)+j] * tile_B[(i * k)+j];
     }
     tile_C[(tx * kTileN) + i]=sum;
  }

  __syncthreads();

}


template<typename T,int kTileM,int kTileN,int ktileK>
__global__ void gemm_simple(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k) {




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
  int k = 1024;

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

    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits=MMA_Traits<mma_op>;
    using mma_atom =MMA_Atom<mma_traits>;

auto MMA = decltype(make_tiled_mma(mma_atom{}, 
                      make_layout(Shape<_2, _2, _1>{}), 
                      make_layout(Shape<_1, _2, _1>{})));


  PRINT("mma",size(MMA{}));

  constexpr int kTileM = 128;
  constexpr int kTileN = 128;
  constexpr int kTileK = 32;

  // each thread block handle with (kTileM, kTileN) output
  dim3 grid(n / kTileN, m / kTileM);
  dim3 block(size(MMA{}));

  int count = 100;
  cudaEventRecord(start);
  for (int i = 0; i < count; ++i)
  {
    gemm_simple_cute<T, kTileM, kTileN, kTileK, MMA><<<grid, block>>>(Cptr, Aptr, Bptr, m, n, k);
  }
  auto err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsedTime, start, end);
  std::cout << "gemm-simple-cute took " << elapsedTime / count << "ms." << std::endl;


  dim3 grid(n / kTileN, m / kTileM);
  dim3 block(kTileM);
  //const int sram
  int count = 100;
  cudaEventRecord(start);
  for (int i = 0; i < count; ++i)
  {
    gemm_simple<T, kTileM, kTileN, kTileK><<<grid, block>>>(Cptr, Aptr, Bptr, m, n, k);
  }
  auto err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsedTime, start, end);
  std::cout << "gemm-simple took " << elapsedTime / count << "ms." << std::endl;

}