#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <iostream>
#include <vector>

// Helper function to check CUDA errors
#define checkCudaErrors(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                 << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

enum FunctionType
{
    TIME_INCREASE,
    TIME_OUT
};

struct Transition
{
    int type;
    int from;
    int to;
    float guard[2];
    FunctionType functionType;
};

struct Euler
{
    int places[2]{ 0 };
    Transition transitions[2];
};

__device__ void timeIncreaseFunction(float* value, curandState* state)
{
    float randomValue = curand_uniform(state);
    *value += randomValue;
}

__device__ void timeOutFunction(float* value, curandState* state)
{
    // No operation needed for timeOut
}

__device__ void callFunction(FunctionType functionType, float* value, curandState* state)
{
    switch (functionType)
    {
    case TIME_INCREASE:
        timeIncreaseFunction(value, state);
        break;
    case TIME_OUT:
        timeOutFunction(value, state);
        break;
    }
}

__global__ void simulate(Euler* euler, int* counts, float* values, curandState* states)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState* state = &states[tid];

    float test = 0.0f;
    int timesFired = 0;
    bool shouldBreak = false;

    while (!shouldBreak) {
        // Always select the TIME_INCREASE transition until test > 1.0f
        Transition youngest = euler->transitions[0];
        for (size_t i = 0; i < 2; i++)
        {
            if (test >= euler->transitions[i].guard[0] && youngest.guard[0] < euler->transitions[i].guard[0])
            {
                youngest = euler->transitions[i];
            }
        }
        callFunction(youngest.functionType, &test, state);
        timesFired++;
        if (test >= 1.0f) {
            shouldBreak = true;
        }
    }

    // Store the count (this approximates e)
    counts[tid] += 1;
    values[tid] += timesFired;
}

__global__ void initCurandStates(curandState* states, unsigned long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed + tid, tid, 0, &states[tid]);
}

int main()
{
    // Create Euler struct with transitions
    Euler euler;

    Transition timeIncrease;
    timeIncrease.from = 0;
    timeIncrease.to = 0;
    timeIncrease.guard[0] = 0;
    timeIncrease.guard[1] = 1.0f;
    timeIncrease.functionType = TIME_INCREASE;
    euler.transitions[0] = timeIncrease;

    Transition timeOut;
    timeOut.from = 0;
    timeOut.to = 1;
    timeOut.guard[0] = 1.0f;
    timeOut.guard[1] = std::numeric_limits<float>::max();
    timeOut.functionType = TIME_OUT;
    euler.transitions[1] = timeOut;

    // Allocate memory on the device for Euler struct
    Euler* d_euler;
    checkCudaErrors(cudaMalloc((void**)&d_euler, sizeof(Euler)));
    checkCudaErrors(cudaMemcpy(d_euler, &euler, sizeof(Euler), cudaMemcpyHostToDevice));

    // Number of threads and blocks
    const int numThreads = 1024;
    const int numBlocks = 3000;
    const int numSimulations = numThreads * numBlocks;

    // Allocate arrays for return values
    int* d_counts;
    float* d_values;
    checkCudaErrors(cudaMalloc((void**)&d_counts, numSimulations * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_values, numSimulations * sizeof(float)));

    // Initialize arrays to zero
    checkCudaErrors(cudaMemset(d_counts, 0, numSimulations * sizeof(int)));
    checkCudaErrors(cudaMemset(d_values, 0, numSimulations * sizeof(float)));

    // Allocate and initialize curand states
    curandState* d_states;
    checkCudaErrors(cudaMalloc((void**)&d_states, numSimulations * sizeof(curandState)));

    std::cout << "Initializing CURAND states..." << std::endl;
    initCurandStates << <numBlocks, numThreads >> > (d_states, time(0));
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "Running simulation..." << std::endl;
    // Launch the kernel with Euler struct

    // for (size_t i = 0; i < 10000; i++)
    // {
        simulate << <numBlocks, numThreads >> > (d_euler, d_counts, d_values, d_states);
        checkCudaErrors(cudaDeviceSynchronize());

    // }


    // Check for errors after kernel execution
    checkCudaErrors(cudaGetLastError());

    std::cout << "Copying results back to host..." << std::endl;
    // Allocate arrays on the host
    std::vector<int> h_counts(numSimulations);
    std::vector<float> h_values(numSimulations);

    // Copy data from device to host
    checkCudaErrors(cudaMemcpy(h_counts.data(), d_counts, numSimulations * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_values.data(), d_values, numSimulations * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_euler);
    cudaFree(d_counts);
    cudaFree(d_values);
    cudaFree(d_states);

    // Process results on the host
    double totalCounts = 0;
    float totalValues = 0;
    for (int i = 0; i < numSimulations; ++i)
    {
        // std::cout << h_counts[i] << " " << h_values[i] << "\n";
        totalCounts += h_counts[i];
        totalValues += h_values[i];
    }

    if (totalCounts > 0)
    {
        float approxE = totalValues / totalCounts;
        std::cout << "Approximation of e: " << approxE << std::endl;
        std::cout << "True value of e: 2.71828..." << std::endl;
        std::cout << numSimulations << std::endl;
    }
    else
    {
        std::cout << "No valid results found." << std::endl;
    }

    return 0;
}