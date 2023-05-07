#ifndef CPU
#define CPU

#include <vector>

using namespace std;

vector<int> runCPU(vector<int> &B, vector<int> &C, vector<int> &values, vector<int> &rowIndices, vector<int> &columnIndices, vector<int> &updatedVertexIndices, int numberOfRows, int threadCount);

#endif
