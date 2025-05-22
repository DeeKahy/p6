#include "euler.h"
#define checkCudaErrors(call)                                                            \
    {                                                                                    \
        cudaError_t err = call;                                                          \
        if (err != cudaSuccess)                                                          \
        {                                                                                \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                           \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    }
__global__ void wood(float *results)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    Tapn net;
    Transition transitions[2];
    net.currentTime = 0.0f;
    net.steps = 0;
    Place places[20];
    net.placesCount = 20;
    net.transitions = transitions;
    net.transitionsCount = 2;

    transitions[0].distribution.type = EXPONENTIAL;
    transitions[0].distribution.a = 0.06f;
    transitions[0].outputArcs[0].output = 0;
    transitions[0].outputArcsCount++;

    transitions[1].distribution.type = EXPONENTIAL;
    transitions[1].distribution.a = 0.0828f;
    transitions[1].outputArcs[0].output = 0;
    transitions[1].outputArcsCount++;

    transitions[2].distribution.type = EXPONENTIAL;
    transitions[2].distribution.a = 0.000718f;
    transitions[2].outputArcs[0].output = 0;
    transitions[2].outputArcsCount++;

    transitions[3].distribution.type = CONSTANT;
    transitions[3].distribution.a = 0.0f;
    transitions[3].urgent = true;
    transitions[3].inputArcs[0].place = 0;
    transitions[3].inputArcs[0].type = INPUT;
    transitions[3].inputArcs[0].timings[0] = 0.0f;
    transitions[3].inputArcs[0].timings[1] = FLT_MAX;
    transitions[3].inputArcsCount++;
    transitions[3].outputArcs[0].output = 1;
    transitions[3].outputArcsCount++;

    transitions[4].distribution.type = CONSTANT;
    transitions[4].distribution.a = 0.0556f;
    transitions[4].inputArcs[0].place = 1;
    transitions[4].inputArcs[0].type = INPUT;
    transitions[4].inputArcs[0].timings[0] = 0.0f;
    transitions[4].inputArcs[0].timings[1] = FLT_MAX;
    transitions[4].inputArcsCount++;
    transitions[4].inputArcs[1].place = 5;
    transitions[4].inputArcs[1].type = INHIBITOR;
    transitions[4].inputArcs[1].timings[0] = 0.0f;
    transitions[4].inputArcs[1].timings[1] = FLT_MAX;
    transitions[4].inputArcsCount++;
    transitions[4].outputArcs[0].output = 2;
    transitions[4].outputArcsCount++;


    
    net.run(places);

    results[tid] += net.steps;
}

__global__ void sum(float *array, unsigned long long numSimulations, unsigned long long totalThreads)
{
    double total = 0.0f;

    for (int i = 0; i < totalThreads; i++)
    {
        total += array[i];
    }
    printf("euler value is %.11f\n", total / numSimulations);
    printf("real euler is 2.71828\n");
}
__global__ void summage(float *array, unsigned long long numSimulations, unsigned long long totalThreads)
{
    int tid = threadIdx.x;
    double sum = 0.0f;

    for (int i = 0; i < numSimulations / totalThreads; i++)
    {
        sum += array[tid + i * totalThreads];
    }

    array[tid] = sum;
}
int main(int argc, char *argv[])
{
    auto start = std::chrono::high_resolution_clock::now();
    float confidence;
    float error;
    unsigned long long threads = 1024;
    unsigned long long blockCount = 2048;
    if (argc < 3)
    {
        confidence = 0.95f;
        error = 0.005f;
    }
    else
    {
        confidence = std::stof(argv[1]);
        error = std::stof(argv[2]);
    }
    std::cout << "confidence: " << confidence << " error: " << error << "\n";
    float number = ceil((log(2 / (1 - confidence))) / (2 * error * error));
    std::cout << "execution calculated: " << number << "\n";
    unsigned long long loopCount = ceil(number / (blockCount * threads));
    std::cout << "loop count: " << loopCount << "\n";
    unsigned long long N{blockCount * threads};
    std::cout << "number of executions run: " << N * loopCount << "\n";
    float *d_results;
    cudaMalloc((void **)&d_results, N * sizeof(float));
    cudaMemset(d_results, 0, N * sizeof(float));
    for (size_t i = 0; i < loopCount; i++)
    {
        wood<<<blockCount, threads>>>(d_results);
        cudaDeviceSynchronize();
    }

    thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(d_results);
    double tot = thrust::reduce(d_ptr, d_ptr + N);
    std::cout << "Success rate: " << tot / (N * loopCount) << "\n";
    cudaError_t errSync = cudaDeviceSynchronize();
    cudaError_t errAsync = cudaGetLastError();

    if (errSync != cudaSuccess)
    {
        printf("Sync error: %s\n", cudaGetErrorString(errSync));
    }
    if (errAsync != cudaSuccess)
    {
        printf("Launch error: %s\n", cudaGetErrorString(errAsync));
    }
    cudaFree(d_results);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "time run: " << duration.count() << "\n";
    return 0;
}