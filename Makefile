runner: src/runner.cpp src/cpu.cpp src/gpu.cu
	nvcc -o runner -std=c++11 src/runner.cpp src/cpu.cpp src/gpu.cu

clean:
	rm runner
