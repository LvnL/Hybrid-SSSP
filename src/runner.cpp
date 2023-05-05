#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

#include "../include/cpu.h"
#include "../include/gpu.h"

using namespace std;

int numberOfRows, numberOfColumns;
vector<int> rowIndices, columnIndices;
vector<float> values;

int main(int argc, char* argv[])
{
	ifstream inputFile(argv[1]);

	cout << "Reading file... " << flush;

	int row, column;
    while (inputFile >> row >> column) {
        rowIndices.push_back(row - 1);
        columnIndices.push_back(column - 1);
        values.push_back(1);
        
        if (row > numberOfRows) numberOfRows = row;
        if (column > numberOfColumns) numberOfColumns = column;
    }
    
    cout << "done" << endl;

    runCPU();
    runGPU();

	return 0;
}
