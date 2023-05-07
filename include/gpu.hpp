#ifndef GPU
#define GPU

#include <vector>

using namespace std;

void runGPU(vector<float> &B, vector<float> &C, vector<float> &values, vector<int> &rows, vector<int> &columns, int numVertices, vector<int> &updates);

#endif
