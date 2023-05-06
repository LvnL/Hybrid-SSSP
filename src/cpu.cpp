#include <iostream>
#include <vector>

using namespace std;

bool updated;

void runCPU(vector<float> &B, vector<float> &C, vector<float> &values, vector<int> &rowIndices, vector<int> &columnIndices, int numberOfRows, int threadCount) {
    cout << "Running iteration on CPU... " << flush;

    for (int i = 0; i < numberOfRows; i++) {
        updated = false;

        for (int j = 0; j < numberOfRows; j++){
                int row = rowIndices[j];
                int column = columnIndices[j];
                float value = values[j];
                if (B[column] + value < C[row]) {
                    C[row] = B[column] + value;
                    updated = true;
                }
        }

        if (!updated) break;
        swap(B, C);
    }

    cout << "done" << endl;

    return;
}
