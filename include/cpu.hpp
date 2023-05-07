#ifndef CPU
#define CPU

#include <vector>

using namespace std;

vector<int> runCPU(vector<float> &B, vector<float> &C, vector<float> &values, vector<int> &rowIndices, vector<int> &columnIndices, vector<int> &updatedVertexIndices, int numberOfRows, int threadCount);

#endif
