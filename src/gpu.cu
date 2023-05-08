#include <iostream>
#include <cuda.h>
#include <vector>

using namespace std;

__global__ void BellmanFord(int *B, int *C, int *rowIndices, int* columnIndices, int *updatedVertices, int edgeCount) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = threadID; i < edgeCount; i += stride) {
        int source = rowIndices[i];
        int target = columnIndices[i];

        if (B[source] + 1 < C[target]) {
            atomicMin(&C[target], B[source] + 1);
            updatedVertices[target] = 1;
        }
    }
}

void runGPU(vector<int> &B, vector<int> &C, vector<int> &rowIndices, vector<int> &columnIndices, vector<int> &updatedVertices) {

    // arbitrary blocksize
    int edgeCount = rowIndices.size();
    int blockSize = 256;
    int numBlocks = (edgeCount + blockSize - 1) / blockSize;

    int *deviceB, *deviceC, *deviceRowIndices, *deviceColumnIndices, *deviceUpdatedVertices;

    cudaMalloc((void **) &deviceB, B.size() * sizeof(int));
    cudaMalloc((void **) &deviceC, C.size() * sizeof(int));
    cudaMalloc((void **) &deviceRowIndices, rowIndices.size() * sizeof(int));
    cudaMalloc((void **) &deviceColumnIndices, columnIndices.size() * sizeof(int));
    cudaMalloc((void **) &deviceUpdatedVertices, updatedVertices.size() * sizeof(int));

    cudaMemcpy(deviceB, B.data(), B.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceC, C.data(), C.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceRowIndices, rowIndices.data(), rowIndices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceColumnIndices, columnIndices.data(), columnIndices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceUpdatedVertices, updatedVertices.data(), updatedVertices.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    BellmanFord<<<numBlocks, blockSize>>>(deviceB, deviceC, deviceRowIndices, deviceColumnIndices, deviceUpdatedVertices, edgeCount);

    cudaMemcpy(B.data(), deviceB, B.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(C.data(), deviceC, C.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(updatedVertices.data(), deviceUpdatedVertices, updatedVertices.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(deviceB);
    cudaFree(deviceC);
    cudaFree(deviceRowIndices);
    cudaFree(deviceColumnIndices);
    cudaFree(deviceUpdatedVertices);
}
