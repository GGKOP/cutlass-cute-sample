#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <util.h>
__global__
void forward_kernel(const float* Q, const float* K, const float* V, const float* Mask ,const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O){
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z;

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int q_offset = (by*gridDim.z*N*d + bz*N*d);
    int kv_offset = (by*gridDim.z*N*d + bz*N*d); 
    int L_offset = (by*gridDim.z*N + by*N ); 

    extern __shared__ float sram[];
    int tile_q_size = Bc * d;
    int tile_s_size = Bc * Br; 
    int tile_kv_size =Br * d; 
    float* Qi = sram;
    float* Kj = &sram[tile_q_size];
    float* Vj = &sram[tile_q_size + tile_kv_size ];
    float* Si = &sram[tile_q_size + (tile_kv_size *2)];
    float* Oi = &sram[tile_q_size + (tile_kv_size *2) + tile_s_size];
    float* Li = &sram[tile_q_size + (tile_kv_size *2) + tile_s_size + 2 * Bc * d];
    float* Mi = &sram[tile_q_size + (tile_kv_size *2) + tile_s_size + 2* Bc * d + Bc];

    for(int x=0;x<d;x++){
            Qi[(tx * d) + x] =Q[q_offset +tile_q_size * blockIdx.x + (tx * d) + x];
        }
     //float  row_l_max  =  0;
     Li[tx]=0;
     Mi[tx]=-INFINITY;
     __syncthreads();


    for(int j=0;j<Tr;j++){
        float  row_m = -INFINITY;

        //加载K,V到SRAM中,compute Si
        for(int mi = 0;mi<Br;mi++ ){
        float sum =0;
            for (int x = 0; x < d; x++) {
                Kj[(tx * d) + x] = K[kv_offset + (tile_kv_size * j) + (tx * d) + x];
                Vj[(tx * d) + x] = V[kv_offset + (tile_kv_size * j) + (tx * d) + x];
                __syncthreads();
                sum += Qi[(tx * d) + x] * Kj[(mi * d) + x];
            }
            sum *= softmax_scale;

            //mask compute
            if(Mask[blockIdx.x * Bc * N + tx * N  + j * Br+ mi] == 1){
                sum = -10000;
            }
                    
       	    Si[(Br * tx) + mi] = sum ;
            // find row max
            if(sum > row_m){
                row_m =sum;
            }
        }
        
        //compute new,m_row_new
        float sum_new = 0;
        float m_new = max(Mi[tx],row_m);
	    //compute P
        for(int x=0;x<Br;x++){
                Si[(tx * Br) + x] = __expf(Si[(tx * Br) + x] - m_new);
                sum_new += Si[(tx * Br) + x] ;
        }
        __syncthreads();
        float l_new = __expf(Mi[tx] - m_new) * Li[tx] + sum_new ;
        //compute O
        for(int n =0;n < d;n++){
            float pv=0; 
            for(int x = 0; x<Br; x++){
                    pv += Si[(tx * Br) + x] * Vj[(x * d) + n]; 
            }  
            
            float temp = __expf(Mi[tx] - m_new); 

            if(j==0){
                Oi[(tx *d)+n] =pv;
            }else{
                Oi[(tx * d)+ n ] =  (1/temp) * Oi[(tx *d) + n ]+pv;
            }


        }
        __syncthreads();
        Mi[tx] = m_new;
        Li[tx] = l_new;
        __syncthreads();
    } 

    //compute and store Oi
        for(int i =0 ;i <d;i++){
        
            Oi[(tx * d) + i] = Oi[(tx * d) + i] * (1/Li[tx]);
            O[q_offset + (tile_q_size * blockIdx.x)+(tx *d) + i] = Oi[(tx * d) + i]; 
        }
        L[L_offset + (Bc * blockIdx.x) + tx] = Mi[tx]+ logf(Li[tx]);  

        __syncthreads();
    }




torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {

    cudaEvent_t start,end;
    cudaEvenntCreate(&start);
    cudaEventCreate(&end);

    const int Bc = 32; 
    const int Br = 32;

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);


    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    auto L = torch::zeros({B,nh,N});
    torch::Device device(torch::kCUDA);
    l = l.to(device);
    m = m.to(device);
    L = L.to(device);

    const int sram_size = (3 * Bc * d * sizeof(float)) + (2 * Br * d * sizeof(float)) + (Bc * Br * sizeof(float)) + 2 * (Bc * sizeof(float));

    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    dim3 grid_dim(Tc , B, nh); 
    dim3 block_dim(Bc);  
    cudaEventRecord(start);
    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),Mask.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, start, end);
    std::cout << "flash-attention-cute took " << elapsedTime / count << "ms." << std::endl;

    std::pair<torch::Tensor, torch::Tensor> result(O, L);
    return result;

/*
    auto result = model.forward(Q, K, V, Mask);
    torch::Tensor O = result.first;
    torch::Tensor L = result.second;

*/
}

