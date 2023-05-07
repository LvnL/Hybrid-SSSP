#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include "../include/cpu.hpp"
#include "../include/gpu.hpp"

#define THREAD_COUNT 8

using namespace std;

int numberOfRows, numberOfColumns;
vector<int> rowIndices, columnIndices, updatedVertices, updatedVertexIndices;
vector<float> B, C, values;

bool compareRows(const vector<int>& vectorA, const vector<int>& vectorB) {
	return vectorA[0] < vectorB[0];
}

int main(int argc, char* argv[]) {
	ifstream inputFile(argv[1]);
    vector<vector<int>> rowColumnIndices;

	cout << "Reading file \"" << argv[1] << "\"... " << flush;

    int row, column;
	while (inputFile >> row >> column) {
		vector<int> rowColumnIndex{row - 1, column - 1};
		rowColumnIndices.push_back(rowColumnIndex);
        values.push_back(1);

		if (row > numberOfRows) numberOfRows = row;
        if (column > numberOfColumns) numberOfColumns = column;
    }	

    cout << "done" << endl;

    cout << "Sorting edges by row... " << flush;

	sort(rowColumnIndices.begin(), rowColumnIndices.end(), compareRows);

    cout << "done" << endl;

    cout << "Storing edges as CSR... " << flush;
    
	rowIndices.push_back(0);
	columnIndices.push_back(rowColumnIndices[0][1]);
    
    int offset = 0;
    #pragma omp parallel for num_threads(THREAD_COUNT)
	for (int i = 1; i < values.size(); i++) {
		if (rowColumnIndices[i][0] != rowColumnIndices[i - 1][0]) {
			for (int j = 0; j < rowColumnIndices[i][0] - rowColumnIndices[i - 1][0]; j++)
				rowIndices.push_back(rowIndices.back());

			if (rowColumnIndices[i][0] == 1) offset += 1;

			rowIndices.back() = rowIndices.back() + offset;
			offset = 0;
		}
		columnIndices.push_back(rowColumnIndices[i][1]);
		offset++;
	}

    cout << "done" << endl;
    
    cout << "Initializing paths... " << flush;

    B.resize(numberOfColumns, 9999999);
    C.resize(numberOfColumns, 9999999);

    // debug arrays
    vector<float> D, E;
    D.resize(numberOfColumns, 9999999);
    E.resize(numberOfColumns, 9999999);
    D[0] = 0;

    cout << "done" << endl;

    B[0] = 0;
    updatedVertices.resize(numberOfColumns, 0);
    updatedVertices[0] = 1;
    updatedVertexIndices.push_back(0);
    
    // Iterate Bellman-Ford
    for (int i = 0; i < numberOfRows; i++) {
        cout << "Iteration: " << i << endl;
        cout << "    Vertices being processed: " << updatedVertexIndices.size() / (float) B.size() * 100 << "%" << endl;

        if (updatedVertices.size() < 0) { // Placeholder to test CPU code, change as needed
            cout << "    Starting GPU iteration... " << flush;

            // reset updates vector 
            fill(updatedVertices.begin(), updatedVertices.end(), 0);

            runGPU(D, E, values, rowIndices, columnIndices, numberOfRows, updatedVertices);
            cout << "done" << endl; // debug

            // Synchronize updatedVertexIndices to match updatedVertices 
            updatedVertexIndices.clear();
            for (int i = 0; i < updatedVertices.size(); i++)
                updatedVertexIndices[i] ? updatedVertices.push_back(i) : void();
        } else {
            cout << "    Starting CPU iteration... " << flush;

            updatedVertexIndices = runCPU(B, C, values, rowIndices, columnIndices, updatedVertexIndices, numberOfRows, 1);
        }

        cout << "done" << endl;

        swap(B, C);
        swap(D, E);
        
        if (!updatedVertexIndices.size())
            break;
    }
    
    cout << "Shortest path to C[100]: " << C[100] << endl;
    cout << "Shortest path to E[100]: " << E[100] << endl;

	return 0;
}
