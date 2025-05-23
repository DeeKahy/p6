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

struct Transition
{
	int type;
	int from;
	int to;
	bool done{ false };
	float guard[2];
	void(*function)(float*, float*);
};

struct Euler
{
	int timesFired{ 0 };
	int timesCalled{ 0 };
	float places[2]{ 0 };
	int tokens[2]{ 1, 0 };
	bool success{ false };
	Transition transitions[2];
};

__device__ void reset(float* from, float* to) {
	*from = 0.0f;
	*to = 0.0f;
}

__device__ void increment(float* from, float* to) {
	curandState state;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(clock64() + tid, tid, 0, &state);
	float randomValue = curand_uniform(&state);
	*to = *from + randomValue;
};

__device__ void simulateThread(Euler* euler) {
	bool shouldBreak = false;
	euler->timesCalled += 1;
	while (!shouldBreak) {
		Transition youngest = euler->transitions[0];
		for (size_t i = 0; i < 2; i++)
		{
			if (euler->places[0] >= euler->transitions[i].guard[0] && youngest.guard[0] < euler->transitions[i].guard[0] && euler->tokens[euler->transitions[i].from] > 0)
			{
				youngest = euler->transitions[i];
			}
		}
		youngest.function(&euler->places[youngest.from], &euler->places[youngest.to]);


		if (youngest.done == true) {
			shouldBreak = true;
		}
		else {
			euler->timesFired++;
		}
	}
}

__global__ void initThread(float* results) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	Euler euler;

	Transition timeIncrease;
	timeIncrease.from = 0;
	timeIncrease.to = 0;
	timeIncrease.guard[0] = 0;
	timeIncrease.guard[1] = 1.0f;
	timeIncrease.function = &increment;
	euler.transitions[0] = timeIncrease;

	Transition timeOut;
	timeOut.from = 0;
	timeOut.to = 1;
	timeOut.guard[0] = 1.0f;
	timeOut.guard[1] = 100000.0f;
	timeOut.function = &reset;
	timeOut.done = true;
	euler.transitions[1] = timeOut;
	simulateThread(&euler);

	results[tid] = euler.timesFired / euler.timesCalled;
}
__global__ void initCurandStates(curandState* states, unsigned long seed)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed + tid, tid, 0, &states[tid]);
}
__global__ void sum(float* array, int numSimulations) {
	float total = 0.0f;
	for (int i = 0; i < 1024; i++) {
		total += array[i];
	}
	//printf("euler value is %.11f\n", (double)total / numSimulations);
}
__global__ void summage(float* array, int numSimulations) {
	int tid = threadIdx.x;
	float sum = 0.0f;

	for (int i = 0; i < numSimulations / 1024; i++) {
		sum += array[tid + i * 1024];
	}

	array[tid] = sum;
}

int main(int argc, char* argv[]) {
	auto start = std::chrono::high_resolution_clock::now();

	int gridSize = argc > 1 ? std::stoi(argv[1]) / 1024 < 1 ? 1 : std::stoi(argv[1]) / 1024 : 1000;
	int blockSize = 1024;
	int numSimulations = gridSize * blockSize;
	float* d_results;
	cudaMalloc((void**)&d_results, numSimulations * sizeof(float));
	initThread << <gridSize, blockSize >> > (d_results);
	summage << <1, blockSize >> > (d_results, numSimulations);
	cudaDeviceSynchronize();
	sum << <1, 1 >> > (d_results, numSimulations);
	cudaDeviceSynchronize();
	std::cout << "True value of e: 2.71828... \n";
	cudaFree(d_results);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "time run: " << duration.count() << std::endl;
	return 0;
}