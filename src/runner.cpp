#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <omp.h>
#include "../include/cpu.hpp"
#include "../include/gpu.hpp"

#define THREAD_COUNT 8

using namespace std;

int rowCount, columnCount, vertexCount;
vector<int> cpuRows, cpuColumns, gpuRows, gpuColumns, updatedVertices, updatedVertexIndices, B, C, values;

bool compareRows(const vector<int>& vectorA, const vector<int>& vectorB) {
	return vectorA[0] < vectorB[0];
}

int main(int argc, char* argv[]) {
    // parse input file
	ifstream inputFile(argv[1]);
    vector<vector<int>> cooEdges;

	cout << "Reading file \"" << argv[1] << "\"... " << flush;

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

    // create csr
    cout << "Storing edges as CSR... " << flush;
    
	cpuRows.push_back(0);
	cpuColumns.push_back(cooEdges[0][1]);
    
    int offset = 0;
    // #pragma omp parallel for num_threads(THREAD_COUNT)
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
    
    // initialize source and destination vectors
    cout << "Initializing paths... " << flush;

    B.resize(vertexCount, 9999999);
    C.resize(vertexCount, 9999999);
    B[0] = 0;

    cout << "done" << endl;

    // initialize updates vector 
    updatedVertices.resize(vertexCount, 0);
    updatedVertices[0] = 1;
    updatedVertexIndices.push_back(0);
    
    // Iterate Bellman-Ford
    for (int i = 0; i < vertexCount; i++) {
        cout << "Iteration: " << i << endl;

        //if (true) {
        if ((updatedVertexIndices.size() / (float) vertexCount * 100) > 5.) {
            cout << "    Starting GPU iteration... " << endl;

            auto start_time = std::chrono::high_resolution_clock::now();

            // Reset updates vector 
            fill(updatedVertices.begin(), updatedVertices.end(), 0);

            runGPU(B, C, gpuRows, gpuColumns, updatedVertices);

            // Synchronize updatedVertexIndices to match updatedVertices 
            updatedVertexIndices.clear();
            for (int i = 0; i < updatedVertices.size(); i++) {
                if (updatedVertices[i])
                    updatedVertexIndices.push_back(i);
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

            cout << "    GPU time taken: " << duration.count() << " microseconds" << endl;
        } else {
            cout << "    Starting CPU iteration... " << endl;

            auto start_time = std::chrono::high_resolution_clock::now();

            updatedVertexIndices = runCPU(B, C, values, cpuRows, cpuColumns, updatedVertexIndices, vertexCount, 1);

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
   
            std::cout << "    CPU time taken: " << duration.count() << " microseconds" << endl;
        }
        
        cout << "done" << endl;

        swap(B, C);
        
        if (!updatedVertexIndices.size()) {
            break;
        }
    }
    
    cout << "Shortest path to C[100]: " << C[100] << endl;

	return 0;
}
