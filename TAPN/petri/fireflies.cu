#include "fireflies.h"

// Helper function to check CUDA errors
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

__global__ void fireflies(float *results)
{
    // printf("start of fireflies");
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Tapn net;

    float token = 0.0f;
    float tokens[1]{token};
    // starting transitions transitions
    // printf("waiting and tokens");

    Place places[14];
    places[0].addTokens(tokens, 1);
    places[1].addTokens(tokens, 1);
    places[2].addTokens(tokens, 1);
    places[3].addTokens(tokens, 1);
    Transition transitions[17];
    net.transitions = transitions;
    net.transitionsCount = 17;
    net.places = places;
    net.placesCount = 14;

    // dis1.init();

    transitions[0].distribution.a = 0.0f;
    transitions[0].distribution.b = 10.0f;
    transitions[0].distribution.type = UNIFORM;

    transitions[0].inputArcs[0].place = &places[0];
    transitions[0].inputArcs[0].type = INPUT;
    transitions[0].inputArcs[0].timings[0] = 0.0f;
    transitions[0].inputArcs[0].timings[1] = FLT_MAX;
    transitions[0].inputArcsCount++;

    transitions[0].outputArcs[0].output = &places[4];
    transitions[0].outputArcsCount++;

    transitions[1].distribution.a = 0.0f;
    transitions[1].distribution.b = 10.0f;
    transitions[1].distribution.type = UNIFORM;
    transitions[1].inputArcs[0].place = &places[1];
    transitions[1].inputArcs[0].type = INPUT;
    transitions[1].inputArcs[0].timings[0] = 0.0f;
    transitions[1].inputArcs[0].timings[1] = FLT_MAX;
    transitions[1].inputArcsCount++;
    transitions[1].outputArcs[0].output = &places[5];
    transitions[1].outputArcsCount++;

    transitions[2].distribution.a = 0.0f;
    transitions[2].distribution.b = 10.0f;
    transitions[2].distribution.type = UNIFORM;
    transitions[2].inputArcs[0].place = &places[2];
    transitions[2].inputArcs[0].type = INPUT;
    transitions[2].inputArcs[0].timings[0] = 0.0f;
    transitions[2].inputArcs[0].timings[1] = FLT_MAX;
    transitions[2].inputArcsCount++;
    transitions[2].outputArcs[0].output = &places[6];
    transitions[2].outputArcsCount++;

    transitions[3].distribution.a = 0.0f;
    transitions[3].distribution.b = 10.0f;
    transitions[3].distribution.type = UNIFORM;
    transitions[3].inputArcs[0].place = &places[3];
    transitions[3].inputArcs[0].type = INPUT;
    transitions[3].inputArcs[0].timings[0] = 0.0f;
    transitions[3].inputArcs[0].timings[1] = FLT_MAX;
    transitions[3].inputArcsCount++;
    transitions[3].outputArcs[0].output = &places[7];
    transitions[3].outputArcsCount++;


    transitions[4].distribution.type = NORMAL;
    transitions[4].distribution.a = 2.0f;
    transitions[4].distribution.b = 0.5f;
    transitions[4].inputArcs[0].place = &places[4];
    transitions[4].inputArcs[0].type = INPUT;
    transitions[4].inputArcs[0].timings[0] = 1.0f;
    transitions[4].inputArcs[0].timings[1] = FLT_MAX;
    transitions[4].inputArcsCount++;
    transitions[4].outputArcs[0].output = &places[8];
    transitions[4].outputArcsCount++;
    transitions[4].outputArcs[1].output = &places[12];
    transitions[4].outputArcsCount++;

    transitions[5].distribution.type = NORMAL;
    transitions[5].distribution.a = 2.0f;
    transitions[5].distribution.b = 0.5f;
    transitions[5].inputArcs[0].place = &places[5];
    transitions[5].inputArcs[0].type = INPUT;
    transitions[5].inputArcs[0].timings[0] = 1.0f;
    transitions[5].inputArcs[0].timings[1] = FLT_MAX;
    transitions[5].inputArcsCount++;
    transitions[5].outputArcs[0].output = &places[9];
    transitions[5].outputArcsCount++;
    transitions[5].outputArcs[1].output = &places[12];
    transitions[5].outputArcsCount++;

    transitions[6].distribution.type = NORMAL;
    transitions[6].distribution.a = 2.0f;
    transitions[6].distribution.b = 0.5f;
    transitions[6].inputArcs[0].place = &places[6];
    transitions[6].inputArcs[0].type = INPUT;
    transitions[6].inputArcs[0].timings[0] = 1.0f;
    transitions[6].inputArcs[0].timings[1] = FLT_MAX;
    transitions[6].inputArcsCount++;
    transitions[6].outputArcs[0].output = &places[10];
    transitions[6].outputArcsCount++;
    transitions[6].outputArcs[1].output = &places[12];
    transitions[6].outputArcsCount++;

    transitions[7].distribution.type = NORMAL;
    transitions[7].distribution.a = 2.0f;
    transitions[7].distribution.b = 0.5f;
    transitions[7].inputArcs[0].place = &places[7];
    transitions[7].inputArcs[0].type = INPUT;
    transitions[7].inputArcs[0].timings[0] = 1.0f;
    transitions[7].inputArcs[0].timings[1] = FLT_MAX;
    transitions[7].inputArcsCount++;
    transitions[7].outputArcs[0].output = &places[11];
    transitions[7].outputArcsCount++;
    transitions[7].outputArcs[1].output = &places[12];
    transitions[7].outputArcsCount++;

    transitions[8].distribution.type = EXPONENTIAL;
    transitions[8].distribution.a = 0.1f;
    transitions[8].inputArcs[0].place = &places[13];
    transitions[8].inputArcs[0].type = INHIBITOR;
    transitions[8].inputArcs[0].constraint = 1;
    transitions[8].inputArcs[0].timings[0] = 0.0f;
    transitions[8].inputArcs[0].timings[1] = FLT_MAX;
    transitions[8].inputArcsCount++;
    transitions[8].inputArcs[1].place = &places[8];
    transitions[8].inputArcs[1].type = INPUT;
    transitions[8].inputArcs[1].timings[0] = 0.0f;
    transitions[8].inputArcs[1].timings[1] = FLT_MAX;
    transitions[8].inputArcsCount++;
    transitions[8].inputArcs[2].place = &places[12];
    transitions[8].inputArcs[2].type = INPUT;
    transitions[8].inputArcs[2].timings[0] = 0.0f;
    transitions[8].inputArcs[2].timings[1] = FLT_MAX;
    transitions[8].inputArcsCount++;
    transitions[8].outputArcs[0].output = &places[13];
    transitions[8].outputArcsCount++;
    transitions[8].outputArcs[1].output = &places[4];
    transitions[8].outputArcsCount++;

    transitions[9].distribution.type = EXPONENTIAL;
    transitions[9].distribution.a = 0.1f;
    transitions[9].inputArcs[0].place = &places[13];
    transitions[9].inputArcs[0].type = INHIBITOR;
    transitions[9].inputArcs[0].constraint = 1;
    transitions[9].inputArcs[0].timings[0] = 0.0f;
    transitions[9].inputArcs[0].timings[1] = FLT_MAX;
    transitions[9].inputArcsCount++;
    transitions[9].inputArcs[1].place = &places[9];
    transitions[9].inputArcs[1].type = INPUT;
    transitions[9].inputArcs[1].timings[0] = 0.0f;
    transitions[9].inputArcs[1].timings[1] = FLT_MAX;
    transitions[9].inputArcsCount++;
    transitions[9].inputArcs[2].place = &places[12];
    transitions[9].inputArcs[2].type = INPUT;
    transitions[9].inputArcs[2].timings[0] = 0.0f;
    transitions[9].inputArcs[2].timings[1] = FLT_MAX;
    transitions[9].inputArcsCount++;
    transitions[9].outputArcs[0].output = &places[13];
    transitions[9].outputArcsCount++;
    transitions[9].outputArcs[1].output = &places[5];
    transitions[9].outputArcsCount++;

    transitions[10].distribution.type = EXPONENTIAL;
    transitions[10].distribution.a = 0.1f;
    transitions[10].inputArcs[0].place = &places[13];
    transitions[10].inputArcs[0].type = INHIBITOR;
    transitions[10].inputArcs[0].constraint = 1;
    transitions[10].inputArcs[0].timings[0] = 0.0f;
    transitions[10].inputArcs[0].timings[1] = FLT_MAX;
    transitions[10].inputArcsCount++;
    transitions[10].inputArcs[1].place = &places[10];
    transitions[10].inputArcs[1].type = INPUT;
    transitions[10].inputArcs[1].timings[0] = 0.0f;
    transitions[10].inputArcs[1].timings[1] = FLT_MAX;
    transitions[10].inputArcsCount++;
    transitions[10].inputArcs[2].place = &places[12];
    transitions[10].inputArcs[2].type = INPUT;
    transitions[10].inputArcs[2].timings[0] = 0.0f;
    transitions[10].inputArcs[2].timings[1] = FLT_MAX;
    transitions[10].inputArcsCount++;
    transitions[10].outputArcs[0].output = &places[13];
    transitions[10].outputArcsCount++;
    transitions[10].outputArcs[1].output = &places[6];
    transitions[10].outputArcsCount++;

    transitions[11].distribution.type = EXPONENTIAL;
    transitions[11].distribution.a = 0.1f;
    transitions[11].inputArcs[0].place = &places[13];
    transitions[11].inputArcs[0].type = INHIBITOR;
    transitions[11].inputArcs[0].constraint = 1;
    transitions[11].inputArcs[0].timings[0] = 0.0f;
    transitions[11].inputArcs[0].timings[1] = FLT_MAX;
    transitions[11].inputArcsCount++;
    transitions[11].inputArcs[1].place = &places[11];
    transitions[11].inputArcs[1].type = INPUT;
    transitions[11].inputArcs[1].timings[0] = 0.0f;
    transitions[11].inputArcs[1].timings[1] = FLT_MAX;
    transitions[11].inputArcsCount++;
    transitions[11].inputArcs[2].place = &places[12];
    transitions[11].inputArcs[2].type = INPUT;
    transitions[11].inputArcs[2].timings[0] = 0.0f;
    transitions[11].inputArcs[2].timings[1] = FLT_MAX;
    transitions[11].inputArcsCount++;
    transitions[11].outputArcs[0].output = &places[13];
    transitions[11].outputArcsCount++;
    transitions[11].outputArcs[1].output = &places[7];
    transitions[11].outputArcsCount++;


    transitions[16].urgent = true;
    transitions[16].distribution.type = CONSTANT;
    transitions[16].distribution.a = 0.0f;
    transitions[16].inputArcs[0].place = &places[12];
    transitions[16].inputArcs[0].type = INHIBITOR;
    transitions[16].inputArcs[0].constraint = 1;
    transitions[16].inputArcs[0].timings[0] = 0.0f;
    transitions[16].inputArcs[0].timings[1] = FLT_MAX;
    transitions[16].inputArcsCount++;
    transitions[16].inputArcs[1].place = &places[13];
    transitions[16].inputArcs[1].type = INPUT;
    transitions[16].inputArcs[1].constraint = 1;
    transitions[16].inputArcs[1].timings[0] = 0.0f;
    transitions[16].inputArcs[1].timings[1] = FLT_MAX;
    transitions[16].inputArcsCount++;
    // tAllDone.outputArcs[0] = &oNoWhere;
    // tAllDone.outputArcsCount++;


    transitions[12].urgent = true;
    transitions[12].distribution.type = CONSTANT;
    transitions[12].distribution.a = 0.0f;
    transitions[12].inputArcs[0].place = &places[8];
    transitions[12].inputArcs[0].type = INPUT;
    transitions[12].inputArcs[0].timings[0] = 0.0f;
    transitions[12].inputArcs[0].timings[1] = FLT_MAX;
    transitions[12].inputArcsCount++;
    transitions[12].inputArcs[1].place = &places[12];
    transitions[12].inputArcs[1].type = INPUT;
    transitions[12].inputArcs[1].timings[0] = 0.0f;
    transitions[12].inputArcs[1].timings[1] = FLT_MAX;
    transitions[12].inputArcsCount++;
    transitions[12].inputArcs[2].place = &places[13];
    transitions[12].inputArcs[2].type = INPUT;
    transitions[12].inputArcs[2].timings[0] = 0.0f;
    transitions[12].inputArcs[2].timings[1] = FLT_MAX;
    transitions[12].inputArcsCount++;
    transitions[12].outputArcs[0].output = &places[13];
    transitions[12].outputArcsCount++;
    transitions[12].outputArcs[1].output = &places[4];
    transitions[12].outputArcsCount++;

    transitions[13].urgent = true;
    transitions[13].distribution.type = CONSTANT;
    transitions[13].distribution.a = 0.0f;
    transitions[13].inputArcs[0].place = &places[9];
    transitions[13].inputArcs[0].type = INPUT;
    transitions[13].inputArcs[0].timings[0] = 0.0f;
    transitions[13].inputArcs[0].timings[1] = FLT_MAX;
    transitions[13].inputArcsCount++;
    transitions[13].inputArcs[1].place = &places[12];
    transitions[13].inputArcs[1].type = INPUT;
    transitions[13].inputArcs[1].timings[0] = 0.0f;
    transitions[13].inputArcs[1].timings[1] = FLT_MAX;
    transitions[13].inputArcsCount++;
    transitions[13].inputArcs[2].place = &places[13];
    transitions[13].inputArcs[2].type = INPUT;
    transitions[13].inputArcs[2].timings[0] = 0.0f;
    transitions[13].inputArcs[2].timings[1] = FLT_MAX;
    transitions[13].inputArcsCount++;
    transitions[13].outputArcs[0].output = &places[13];
    transitions[13].outputArcsCount++;
    transitions[13].outputArcs[1].output = &places[5];
    transitions[13].outputArcsCount++;

    transitions[14].urgent = true;
    transitions[14].distribution.type = CONSTANT;
    transitions[14].distribution.a = 0.0f;
    transitions[14].inputArcs[0].place = &places[10];
    transitions[14].inputArcs[0].type = INPUT;
    transitions[14].inputArcs[0].timings[0] = 0.0f;
    transitions[14].inputArcs[0].timings[1] = FLT_MAX;
    transitions[14].inputArcsCount++;
    transitions[14].inputArcs[1].place = &places[12];
    transitions[14].inputArcs[1].type = INPUT;
    transitions[14].inputArcs[1].timings[0] = 0.0f;
    transitions[14].inputArcs[1].timings[1] = FLT_MAX;
    transitions[14].inputArcsCount++;
    transitions[14].inputArcs[2].place = &places[13];
    transitions[14].inputArcs[2].type = INPUT;
    transitions[14].inputArcs[2].timings[0] = 0.0f;
    transitions[14].inputArcs[2].timings[1] = FLT_MAX;
    transitions[14].inputArcsCount++;
    transitions[14].outputArcs[0].output = &places[13];
    transitions[14].outputArcsCount++;
    transitions[14].outputArcs[1].output = &places[6];
    transitions[14].outputArcsCount++;

    transitions[15].urgent = true;
    transitions[15].distribution.type = CONSTANT;
    transitions[15].distribution.a = 0.0f;
    transitions[15].inputArcs[0].place = &places[11];
    transitions[15].inputArcs[0].type = INPUT;
    transitions[15].inputArcs[0].timings[0] = 0.0f;
    transitions[15].inputArcs[0].timings[1] = FLT_MAX;
    transitions[15].inputArcsCount++;
    transitions[15].inputArcs[1].place = &places[12];
    transitions[15].inputArcs[1].type = INPUT;
    transitions[15].inputArcs[1].timings[0] = 0.0f;
    transitions[15].inputArcs[1].timings[1] = FLT_MAX;
    transitions[15].inputArcsCount++;
    transitions[15].inputArcs[2].place = &places[13];
    transitions[15].inputArcs[2].type = INPUT;
    transitions[15].inputArcs[2].timings[0] = 0.0f;
    transitions[15].inputArcs[2].timings[1] = FLT_MAX;
    transitions[15].inputArcsCount++;
    transitions[15].outputArcs[0].output = &places[13];
    transitions[15].outputArcsCount++;
    transitions[15].outputArcs[1].output = &places[7];
    transitions[15].outputArcsCount++;

    net.places = places;
    net.placesCount = 14;

    // Transition transitions[17]{aArrive0, aArrive1, aArrive2, aArrive3, aReady0, aReady1, aReady2, aReady3, aFlashing0, aFlashing1, aFlashing2, aFlashing3, tFlashJointly0, tFlashJointly1, tFlashJointly2, tFlashJointly3, tAllDone};
    net.transitions = transitions;
    net.transitionsCount = 17;
    // TokenCountObserver observer;
    // net.addObserver(&observer);
    bool success{false};
    net.run2(&success);
    results[tid] += success;
    // lenghts[tid] += net.steps;
}
__global__ void sum(float *array, unsigned long long numSimulations, unsigned long long totalThreads)
{
    double total = 0;

    for (int i = 0; i < totalThreads; i++)
    {
        total += array[i];
    }
    printf("success rate is %.11f\n", total / numSimulations);
}
__global__ void summage(float *array, unsigned long long numSimulations, unsigned long long totalThreads)
{
    int tid = threadIdx.x;
    double sum = 0;

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
    checkCudaErrors(cudaMalloc((void **)&d_results, N * sizeof(float)));
    checkCudaErrors(cudaMemset(d_results, 0, N * sizeof(float)));
    for (size_t i = 0; i < loopCount; i++)
    {
        fireflies<<<blockCount, threads>>>(d_results);
        checkCudaErrors(cudaDeviceSynchronize());
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