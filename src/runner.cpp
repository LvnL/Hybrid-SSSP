#include <fstream>
#include <iostream>
#include <vector>

#include "../include/cpu.hpp"
#include "../include/gpu.hpp"

using namespace std;

int numberOfRows, numberOfColumns;
vector<int> rowIndices, columnIndices, updatedVertex, updatedVertexIndices;
vector<float> B, C, values;

int main(int argc, char* argv[]) {
	ifstream inputFile(argv[1]);

	cout << "Reading file \"" << argv[1] << "\"... " << flush;

	int row, column;
    while (inputFile >> row >> column) {
        rowIndices.push_back(row - 1);
        columnIndices.push_back(column - 1);
        values.push_back(1);
        
        if (row > numberOfRows) numberOfRows = row;
        if (column > numberOfColumns) numberOfColumns = column;
    }
    
    cout << "done" << endl;
    
    cout << "Initializing matrices... " << flush;

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
        cout << "    Vertices being processed: " << updatedVertexIndices.size() / (float) B.size() * 100 << "%" << endl << "    ";

        if (updatedVertexIndices.size() < 0) { // Placeholder to test CPU code, change as needed
            runGPU(B, C, values, rowIndices, columnIndices, numberOfRows);

            // Synchronize updatedVertexIndices to match updatedVertex
            updatedVertexIndices.clear();
            for (int i = 0; i < updatedVertex.size(); i++)
                updatedVertexIndices[i] ? updatedVertex.push_back(i) : void();
        } else {
            updatedVertexIndices = runCPU(B, C, values, rowIndices, columnIndices, updatedVertexIndices, 1);

            // Synchronize updatedVertex to match updatedVertexIndices
            for (int i = 0; i < updatedVertex.size(); i++)
                updatedVertex[i] = 0;

            for (int i = 0; i < updatedVertexIndices.size(); i++)
                updatedVertex[updatedVertexIndices[i]] = 1;
        }

        swap(B, C);
    }
    
	return 0;
}
