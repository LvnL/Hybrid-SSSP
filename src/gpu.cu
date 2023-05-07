#include <iostream>
#include <cuda.h>
#include <vector>

using namespace std;

__global__ void BellmanFord(int numVertices, int numEdges, int *rows, int *columns, float *vals, float *dists, float *sources) {
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // int update = FALSE;

    for (int i=t_id; i<numEdges; i+=stride) {
        int u = rows[i];
        int v = columns[i];
        float val = vals[i];

        if (dists[v] + val < sources[u]) {
            sources[u] = dists[v] + val;
            // update = TRUE; // non-functional for now
        }
    }

    // swap sources and dists here or in main loop
    float *tmp = dists;
    dists = sources;
    sources = tmp;

    /*
    long threadID = (long) tn;
	int blockSize = numberOfRows / numberOfThreads;

	for (int source = threadID * blockSize; source < (threadID + 1) * blockSize; source++) {
		for (int j = rowIndices[source]; j < (source == numberOfRows - 1 ? columnIndices.size() : rowIndices[source + 1]); j++) {
			int target = columnIndices[j];
			float value = values[j];
			if (B[source] + value < C[target]) {
				C[target] = B[source] + value;
				updated = true;
			}
		}
	}
    */
}

void runGPU(vector<float> &B, vector<float> &C, vector<float> &values, vector<int> &rowIndices, vector<int> &columnIndices, int numVertices) {

    cout << "Begin GPU runtime..." << endl;

    // arbitrary blocksize
    int numEdges = rowIndices.size();
    int blockSize = 256;
    int numBlocks = (numEdges + blockSize - 1) / blockSize;

    int *d_rows, *d_columns;
    float *d_vals, *d_dists, *d_sources;

    // intialize device array pointers
    cudaMalloc((void **) &d_rows, rowIndices.size() * sizeof(int));
    cudaMalloc((void **) &d_columns, columnIndices.size() * sizeof(int));
    cudaMalloc((void **) &d_vals, values.size() * sizeof(float));
    cudaMalloc((void **) &d_dists, B.size() * sizeof(float));
    cudaMalloc((void **) &d_sources, C.size() * sizeof(float));

    // load data into device 
    cudaMemcpy(d_rows, rowIndices.data(), rowIndices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, columnIndices.data(), columnIndices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dists, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sources, C.data(), C.size() * sizeof(float), cudaMemcpyHostToDevice);

    // begin iteration
    BellmanFord<<<numBlocks, blockSize>>>(numVertices, numEdges, d_rows, d_columns, d_vals, d_dists, d_sources);

    // update host memory
    cudaMemcpy(&B, d_dists, B.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&C, d_sources, C.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_rows);
    cudaFree(d_columns);
    cudaFree(d_vals);
    cudaFree(d_dists);
    cudaFree(d_sources);
}
