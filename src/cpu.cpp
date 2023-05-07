#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

vector<int> runCPU(vector<float> &B, vector<float> &C, vector<float> &values, vector<int> &rowIndices, vector<int> &columnIndices, vector<int> &updatedVertexIndices, int threadCount) {
    cout << "Running iteration on CPU... " << flush;

    vector<int> newUpdatedVertexIndices;

    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < updatedVertexIndices.size(); i++) {
        int source = updatedVertexIndices[i];
        for (int j = 0; j < values.size(); j++) { // Inefficiency caused by using COO instead of CSR
            if (rowIndices[j] == source) {
                int target = columnIndices[j];
                float value = values[j];
                if (B[source] + value < C[target]) {
                    C[target] = B[source] + value;
                    newUpdatedVertexIndices.push_back(target);
                }
            }
        }
    }

    cout << "done" << endl;

    return newUpdatedVertexIndices;
}
