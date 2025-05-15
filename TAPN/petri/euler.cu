#include "main.h"

__global__ void euler(float *results)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Tapn net;
    Place place1;
    float token = 0.0f;
    float tokens[1]{token};
    place1.addTokens(tokens, 1);
    Place place2;

    Arc arc1;
    arc1.place = &place1;
    arc1.type = TRANSPORT;
    arc1.timings[0] = 0.0f;
    arc1.timings[1] = FLT_MAX;

    OutputArc oArc1;
    oArc1.isTransport = true;
    oArc1.output = &place1;

    Distribution dis1;
    dis1.type = UNIFORM;
    dis1.a = 0.0f;
    dis1.b = 1.0f;
    dis1.init();

    Transition trans1;
    trans1.distribution = &dis1;
    trans1.inputArcs[0] = &arc1;
    trans1.inputArcsCount++;
    trans1.outputArcs[0] = &oArc1;
    trans1.outputArcsCount++;

    Distribution dis2;
    dis2.type = CONSTANT;
    dis2.a = 0.0f;

    Arc arc2;
    arc2.place = &place1;
    arc2.type = INPUT;
    arc2.timings[0] = 1.0f;
    arc2.timings[1] = FLT_MAX;

    OutputArc oArc2;
    oArc2.isTransport = false;
    oArc2.output = &place2;

    Transition trans2;
    trans2.distribution = &dis2;
    trans2.inputArcs[0] = &arc2;
    trans2.inputArcsCount++;
    trans2.outputArcs[0] = &oArc2;
    trans2.outputArcsCount++;

    Place *places[2]{&place1, &place2};

    net.places = places;

    net.placesCount = 2;

    Transition *transitions[2]{&trans1, &trans2};
    net.transitions = transitions;
    net.transitionsCount = 2;
    // TokenAgeObserver tokenAgeObs(MAXFLOAT);
    // net.addObserver(&tokenAgeObs);
    // TokenCountObserver tokenCountObs;
    // net.addObserver(&tokenCountObs);
    net.run();
    results[tid] += net.steps - 1;
    // net.step(&test);
    // //printf("\n place 0 %f\n", place1.tokens[0]);
    // net.step(&test);
    // net.step(&test);
}

__global__ void sum(float *array, int numSimulations, int totalThreads)
{
    float total = 0.0f;

    for (int i = 0; i < totalThreads; i++)
    {
        total += array[i];
    }
    printf("euler value is %.11f\n", (double)total / numSimulations);
    printf("real euler is 2.71828");
}
__global__ void summage(float *array, int numSimulations, int totalThreads)
{
    int tid = threadIdx.x;
    float sum = 0.0f;

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
    int threads = 512;
    int blockCount = 2048;
    if (argc < 3)
    {
        confidence = 0.95f;
        error = 0.0005f;
    }
    else
    {
        confidence = std::stof(argv[1]);
        error = std::stof(argv[2]);
    }
    std::cout << "confidence: " << confidence << " error: " << error << std::endl;
    float number = ceil((log(2 / (1 - confidence))) / (2 * error * error));
    std::cout << "number of executions: " << number << std::endl;
    int loopCount = ceil(number / (blockCount * threads));
    std::cout << "loop count: " << loopCount << std::endl;
    std::cout << "number of executions: " << loopCount * blockCount * threads << std::endl;
    float *d_results;

    cudaMalloc((void **)&d_results, blockCount * threads * sizeof(float));
    cudaMemset((void **)&d_results, 0, blockCount * threads * sizeof(float));
    for (size_t i = 0; i < loopCount; i++)
    {
        euler<<<blockCount, threads>>>(d_results);
        cudaDeviceSynchronize();
    }

    summage<<<1, threads>>>(d_results, blockCount * threads, threads);
    cudaDeviceSynchronize();
    sum<<<1, 1>>>(d_results, loopCount * blockCount * threads, threads);
    cudaDeviceSynchronize();
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
    std::cout << "time run: " << duration.count() << std::endl;
    return 0;
}