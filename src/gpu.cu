#include <iostream>
#include <cuda.h>
#include <vector>

using namespace std;

__global__ void BellmanFord(int *B, int *C, int *rows, int* columns, int *updatedVertices, int edgeCount) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = threadID; i < edgeCount; i += stride) {
        int source = rows[i];
        int target = columns[i];

        if (B[source] + 1 < C[target]) {
            atomicMin(&C[target], B[source] + 1);
            updatedVertices[target] = 1;
        }
    }
}

void runGPU(vector<int> &B, vector<int> &C, vector<int> &rows, vector<int> &columns, vector<int> &updatedVertices) {
    // arbitrary blocksize
    int edgeCount = rows.size();
    int blockSize = 256;
    int numBlocks = (edgeCount + blockSize - 1) / blockSize;

    int *deviceB, *deviceC, *deviceRows, *deviceColumns, *deviceUpdatedVertices;

    cudaMalloc((void **) &deviceB, B.size() * sizeof(int));
    cudaMalloc((void **) &deviceC, C.size() * sizeof(int));
    cudaMalloc((void **) &deviceRows, rows.size() * sizeof(int));
    cudaMalloc((void **) &deviceColumns, columns.size() * sizeof(int));
    cudaMalloc((void **) &deviceUpdatedVertices, updatedVertices.size() * sizeof(int));

    cudaMemcpy(deviceB, B.data(), B.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceC, C.data(), C.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceRows, rows.data(), rows.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceColumns, columns.data(), columns.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceUpdatedVertices, updatedVertices.data(), updatedVertices.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    BellmanFord<<<numBlocks, blockSize>>>(deviceB, deviceC, deviceRows, deviceColumns, deviceUpdatedVertices, edgeCount);

    cudaMemcpy(B.data(), deviceB, B.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(C.data(), deviceC, C.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(rows.data(), deviceRows, rows.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(columns.data(), deviceColumns, columns.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(updatedVertices.data(), deviceUpdatedVertices, updatedVertices.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(deviceB);
    cudaFree(deviceC);
    cudaFree(deviceRows);
    cudaFree(deviceColumns);
    cudaFree(deviceUpdatedVertices);
}
