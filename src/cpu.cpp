#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

vector<int> runCPU(vector<int> &B, vector<int> &C, vector<int> &values, vector<int> &rowIndices, vector<int> &columnIndices, vector<int> &updatedVertexIndices, int numberOfRows, int threadCount) {
    vector<int> newUpdatedVertexIndices;

    #pragma omp parallel for num_threads(threadCount)
    for (int i = 0; i < updatedVertexIndices.size(); i++) {
        int source = updatedVertexIndices[i];
		for (int j = rowIndices[source]; j < (source == numberOfRows - 1 ? columnIndices.size() : rowIndices[source + 1]); j++) {
			int target = columnIndices[j];
			int value = values[j];
			if (B[source] + value < C[target]) {
				C[target] = B[source] + value;
				newUpdatedVertexIndices.push_back(target);
			}
		}
	}

    return newUpdatedVertexIndices;
}
