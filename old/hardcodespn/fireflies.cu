#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
// Helper function to check CUDA errors
#define checkCudaErrors(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                 << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}
struct Transition{
	int type;
	int from;
	int to;
	bool done{ false };
	float guard[2];
	void(*function)(float*, float*);
};
struct SPN{
    int timesFired{ 0 };
	int timesCalled{ 0 };
	float places[2]{ 0 };
	int tokens[2]{ 1, 0 };
	bool success{ false };
	Transition transitions[20];
};

__global__ void initThread(){

}

int main(int argc, char* argv[]) {
	auto start = std::chrono::high_resolution_clock::now();
	int gridSize = argc > 1 ? std::stoi(argv[1]) : 1000;
	int blockSize = 1024;
	int numSimulations = gridSize * blockSize;
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "time run: " << duration.count() << std::endl;
	return 0;
}
