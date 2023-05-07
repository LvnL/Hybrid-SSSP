runner: src/runner.cpp src/cpu.cpp src/gpu.cu
	nvcc -o runner src/runner.cpp src/cpu.cpp src/gpu.cu

clean:
	rm runner
