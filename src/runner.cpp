#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

#include "../include/cpu.hpp"
#include "../include/gpu.hpp"

using namespace std;

int numberOfRows, numberOfColumns;
vector<int> rowIndices, columnIndices;
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

    B.clear();
    C.clear();

    for (int i = 0; i < numberOfColumns; i++)
        B.push_back(9999999);
    for (int i = 0; i < numberOfRows; i++)
        C.push_back(9999999);

    B[0] = 0;

    cout << "done" << endl;

    runCPU(B, C, values, rowIndices, columnIndices, numberOfRows, 1);
    runGPU(B, C, values, rowIndices, columnIndices, numberOfRows);

	return 0;
}
