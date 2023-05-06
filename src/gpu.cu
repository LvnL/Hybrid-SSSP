#include <iostream>
#include <cuda.h>
#include <vector>

#define TRUE 1
#define FALSE 0
#define INF 999999

using namespace std;

__global__ void BellmanFord(int numVertices, int numEdges, int rows[], int columns[], float vals[], int dists[], int sources[]) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // int update = FALSE;

    for (int i=index; i<numEdges; i+=stride) {
        int u = rows[i];
        int v = columns[i];
        int val = vals[i];

        if (dists[v] + 1 < sources[u]) {
            sources[u] = dists[v] + 1;
            // update = TRUE; // non-functional for now
        }
    }

    // swap sources and dists here or in main loop

    return;
}

void runGPU(int src, int numVertices, vector<int> rows, vector<int> columns, vector<float> vals) {

    int numEdges = rows.size();
    
    // arbitrary blocksize
    int blockSize = 256;
    int numBlocks = (numEdges + blockSize - 1) / blockSize;

    // initialize source and dist vectors
    vector<int> dists;
    vector<int> sources;
    
    for (int i=0; i<numVertices; i++) {
        dists.push_back(INF);
        sources.push_back(INF);
    }
    dists[src] = 0;

    // begin iterations
    for (int i=0; i<numVertices; i++) {
        // always defers to GPU for now
        // BellmanFord<<<numBlocks, blockSize>>>(numVertices, numEdges, rows, columns, vals, dists, sources);
        cudaDeviceSynchronize();
    }
}
