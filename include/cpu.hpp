#ifndef CPU
#define CPU

#include <vector>

using namespace std;

void runCPU(vector<float> &B, vector<float> &C, vector<float> &values, vector<int> &rowIndices, vector<int> &columnIndices, int numberOfRows, int threadCount);

#endif
