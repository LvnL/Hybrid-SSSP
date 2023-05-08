#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <omp.h>
#include "../include/cpu.hpp"
#include "../include/gpu.hpp"

#define THREAD_COUNT 8
#define ALPHA 0.05 // proportion of active vertices required to use GPU

using namespace std;

// global variables
int rowCount, columnCount, vertexCount;
vector<int> cpuRows, cpuColumns, gpuRows, gpuColumns, values;

bool compareRows(const vector<int>& vectorA, const vector<int>& vectorB) {
	return vectorA[0] < vectorB[0];
}

int bellmanFord(bool useCPU, bool useGPU, float alpha, int threadCount) {
    vector<int> updatedVertices, updatedVertexIndices, B, C;

    // initialize source and destination vectors
    B.resize(vertexCount, 9999999);
    C.resize(vertexCount, 9999999);
    B[0] = 0;

    // initialize updates vector 
    updatedVertices.resize(vertexCount, 0);
    updatedVertices[0] = 1;
    updatedVertexIndices.push_back(0);
    
    auto start_time = chrono::high_resolution_clock::now();

    // run bellman-ford
    int i;
    for (i = 0; i < vertexCount; i++) {

        if ((useGPU && !useCPU) || (useGPU && (updatedVertexIndices.size() / (float) vertexCount) >= alpha)) {
            // Reset updates vector 
            fill(updatedVertices.begin(), updatedVertices.end(), 0);

            // run gpu iteration
            runGPU(B, C, gpuRows, gpuColumns, updatedVertices);

            // Synchronize updatedVertexIndices to match updatedVertices 
            updatedVertexIndices.clear();
            for (int i = 0; i < updatedVertices.size(); i++) {
                if (updatedVertices[i])
                    updatedVertexIndices.push_back(i);
            }
        } else {
            // run cpu iteration
            updatedVertexIndices = runCPU(B, C, values, cpuRows, cpuColumns, updatedVertexIndices, vertexCount, threadCount);
        }

        swap(B, C);

        if (!updatedVertexIndices.size()) {
            break;
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "CPU: " << useCPU << " GPU: " << useGPU << " Alpha: " << alpha << " Threads: " << threadCount << " Runtime: " << duration.count() << "ms Iterations: " << i << " C[100]: " << C[100] << endl;
}

int main(int argc, char* argv[]) {
	ifstream inputFile(argv[1]);
	cout << "Reading file \"" << argv[1] << "\"... " << flush;

    // parse input file
    vector<vector<int>> cooEdges;
    int row, column;
	while (inputFile >> row >> column) {
		vector<int> rowColumnIndex{row - 1, column - 1};
        gpuRows.push_back(row - 1);
        gpuColumns.push_back(column - 1);
		cooEdges.push_back(rowColumnIndex);
        values.push_back(1);

		if (row > rowCount) rowCount = row;
        if (column > columnCount) columnCount = column;
    }
    vertexCount = max(rowCount, columnCount);
    cout << "done" << endl;

    // sort edges
    cout << "Sorting edges by row... " << flush;
	sort(cooEdges.begin(), cooEdges.end(), compareRows);
    cout << "done" << endl;

    // initialize csr vector
    cout << "Storing edges as CSR... " << flush;
	cpuRows.push_back(0);
	cpuColumns.push_back(cooEdges[0][1]);

    // generate csr
    int offset = 0;
    #pragma omp parallel for num_threads(THREAD_COUNT)
	for (int i = 1; i < values.size(); i++) {
		if (cooEdges[i][0] != cooEdges[i - 1][0]) {
			for (int j = 0; j < cooEdges[i][0] - cooEdges[i - 1][0]; j++)
				cpuRows.push_back(cpuRows.back());

			if (cooEdges[i][0] == 1) offset += 1;

			cpuRows.back() = cpuRows.back() + offset;
			offset = 0;
		}
		cpuColumns.push_back(cooEdges[i][1]);
		offset++;
	}
    cout << "done" << endl;
    
    // cpu+gpu hybrid
    cout << endl << "CPU + GPU" << endl << "==========" << endl;
    for (float alpha : { 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5 }) {
        for (int threads : { 1, 2, 4, 8, 16 }) {
            bellmanFord(true, true, alpha, threads);
        }
    }
    
    // cpu only
    cout << endl << "CPU Only" << endl << "==========" << endl;
    for (int threads : { 1, 2, 4, 8, 16 }) {
        bellmanFord(true, false, 1.0, threads);
    }

    // gpu only
    cout << endl << "GPU Only" << endl << "==========" << endl;
    bellmanFord(false, true, 0.0, 1);

	return 0;
}
