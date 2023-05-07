#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include "../include/cpu.hpp"
#include "../include/gpu.hpp"

using namespace std;

int numberOfRows, numberOfColumns;
vector<int> rowIndices, columnIndices, updatedVertex, updatedVertexIndices;
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

    for (int i = 0; i < numberOfColumns; i++)
        B.push_back(9999999);
    for (int i = 0; i < numberOfRows; i++)
        C.push_back(9999999);
    
    cout << "done" << endl;

    B[0] = 0;
    updatedVertex.resize(numberOfColumns, 0);
    updatedVertex[0] = 1;
    updatedVertexIndices.push_back(0);
    
    // Iterate Bellman-Ford
    for (int i = 0; i < numberOfRows; i++) {
        cout << "Iteration: " << i << endl;
        cout << "    Vertices being processed: " << updatedVertexIndices.size() / (float) B.size() * 100 << "%" << endl;

        if (updatedVertexIndices.size() < 0) { // Placeholder to test CPU code, change as needed
            cout << "    Starting GPU iteration... " << flush;

            for (int i = 0; i < updatedVertex.size(); i++)
                updatedVertex[i] = 0;

            runGPU(B, C, values, rowIndices, columnIndices, numberOfRows, updatedVertex);

            // Synchronize updatedVertexIndices to match updatedVertex
            updatedVertexIndices.clear();
            for (int i = 0; i < updatedVertex.size(); i++)
                updatedVertexIndices[i] ? updatedVertex.push_back(i) : void();
        } else {
            cout << "    Starting CPU iteration... " << flush;

            updatedVertexIndices = runCPU(B, C, values, rowIndices, columnIndices, updatedVertexIndices, numberOfRows, 1);
        }

        cout << "done" << endl;

        swap(B, C);
        
        if (!updatedVertexIndices.size())
            break;
    }
    
    cout << "Shortest path to C[100]: " << C[100] << endl;

	return 0;
}
