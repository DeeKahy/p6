#include "fireflies.h"

// Helper function to check CUDA errors
#define checkCudaErrors(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                 << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}


__global__ void fireflies(float *results)
{
    // printf("start of fireflies");
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Tapn net;

    float token = 0.0f;
    float tokens[1] {token};
    // starting transitions transitions
    // printf("waiting and tokens");
    Place waiting0;
    waiting0.addTokens(tokens, 1);
    Place waiting1;
    waiting1.addTokens(tokens, 1);
    Place waiting2;
    waiting2.addTokens(tokens, 1);
    Place waiting3;
    waiting3.addTokens(tokens, 1);

    Distribution dis1;
    dis1.type = UNIFORM;
    dis1.a = 0.0f;
    dis1.b = 10.0f;
    dis1.init();
    Arc arrive0;
    arrive0.place = &waiting0;
    arrive0.type = INPUT;
    arrive0.timings[0] = 0.0f;
    arrive0.timings[1] = FLT_MAX;
    Arc arrive1;
    arrive1.place = &waiting1;
    arrive1.type = INPUT;
    arrive1.timings[0] = 0.0f;
    arrive1.timings[1] = FLT_MAX;
    Arc arrive2;
    arrive2.place = &waiting2;
    arrive2.type = INPUT;
    arrive2.timings[0] = 0.0f;
    arrive2.timings[1] = FLT_MAX;
    Arc arrive3;
    arrive3.place = &waiting3;
    arrive3.type = INPUT;
    arrive3.timings[0] = 0.0f;
    arrive3.timings[1] = FLT_MAX;

    Place charging0;
    Place charging1;
    Place charging2;
    Place charging3;

    OutputArc oArrive0;
    oArrive0.isTransport = false;
    oArrive0.output = &charging0;
    OutputArc oArrive1;
    oArrive1.isTransport = false;
    oArrive1.output = &charging1;
    OutputArc oArrive2;
    oArrive2.isTransport = false;
    oArrive2.output = &charging2;
    OutputArc oArrive3;
    oArrive3.isTransport = false;
    oArrive3.output = &charging3;

    Transition aArrive0;
    aArrive0.distribution = &dis1;
    aArrive0.inputArcs[0] = &arrive0;
    aArrive0.inputArcsCount++;
    aArrive0.outputArcs[0] = &oArrive0;
    aArrive0.outputArcsCount++;
    Transition aArrive1;
    aArrive1.distribution = &dis1;
    aArrive1.inputArcs[0] = &arrive1;
    aArrive1.inputArcsCount++;
    aArrive1.outputArcs[0] = &oArrive1;
    aArrive1.outputArcsCount++;
    Transition aArrive2;
    aArrive2.distribution = &dis1;
    aArrive2.inputArcs[0] = &arrive2;
    aArrive2.inputArcsCount++;
    aArrive2.outputArcs[0] = &oArrive2;
    aArrive2.outputArcsCount++;
    Transition aArrive3;
    aArrive3.distribution = &dis1;
    aArrive3.inputArcs[0] = &arrive3;
    aArrive3.inputArcsCount++;
    aArrive3.outputArcs[0] = &oArrive3;
    aArrive3.outputArcsCount++;

    Distribution dis2;
    dis2.type = NORMAL;
    dis2.a = 2.0f;
    dis2.b = 0.5f;
    dis2.init();

    Arc ready0;
    ready0.place = &charging0;
    ready0.type = INPUT;
    ready0.timings[0] = 1.0f;
    ready0.timings[1] = FLT_MAX;
    Arc ready1;
    ready1.place = &charging1;
    ready1.type = INPUT;
    ready1.timings[0] = 1.0f;
    ready1.timings[1] = FLT_MAX;
    Arc ready2;
    ready2.place = &charging2;
    ready2.type = INPUT;
    ready2.timings[0] = 1.0f;
    ready2.timings[1] = FLT_MAX;
    Arc ready3;
    ready3.place = &charging3;
    ready3.type = INPUT;
    ready3.timings[0] = 1.0f;
    ready3.timings[1] = FLT_MAX;

    Place charged0;
    Place charged1;
    Place charged2;
    Place charged3;
    Place chargedSum;

    OutputArc oReady0;
    oReady0.isTransport = false;
    oReady0.output = &charged0;
    OutputArc oReady1;
    oReady1.isTransport = false;
    oReady1.output = &charged1;
    OutputArc oReady2;
    oReady2.isTransport = false;
    oReady2.output = &charged2;
    OutputArc oReady3;
    oReady3.isTransport = false;
    oReady3.output = &charged3;

    OutputArc oChargedSum;
    oChargedSum.isTransport = false;
    oChargedSum.output = &chargedSum;

    Transition aReady0;
    aReady0.distribution = &dis2;
    aReady0.inputArcs[0] = &ready0;
    aReady0.inputArcsCount++;
    aReady0.outputArcs[0] = &oReady0;
    aReady0.outputArcsCount++;
    aReady0.outputArcs[1] = &oChargedSum;
    aReady0.outputArcsCount++;

    Transition aReady1;
    aReady1.distribution = &dis2;
    aReady1.inputArcs[0] = &ready1;
    aReady1.inputArcsCount++;
    aReady1.outputArcs[0] = &oReady1;
    aReady1.outputArcsCount++;
    aReady1.outputArcs[1] = &oChargedSum;
    aReady1.outputArcsCount++;

    Transition aReady2;
    aReady2.distribution = &dis2;
    aReady2.inputArcs[0] = &ready2;
    aReady2.inputArcsCount++;
    aReady2.outputArcs[0] = &oReady2;
    aReady2.outputArcsCount++;
    aReady2.outputArcs[1] = &oChargedSum;
    aReady2.outputArcsCount++;

    Transition aReady3;
    aReady3.distribution = &dis2;
    aReady3.inputArcs[0] = &ready3;
    aReady3.inputArcsCount++;
    aReady3.outputArcs[0] = &oReady3;
    aReady3.outputArcsCount++;
    aReady3.outputArcs[1] = &oChargedSum;
    aReady3.outputArcsCount++;
    Place flashing;
    Arc aChargedSum;
    aChargedSum.place = &chargedSum;
    aChargedSum.type = INPUT;
    aChargedSum.timings[0] = 0.0f;
    aChargedSum.timings[1] = FLT_MAX;

    Arc flashAlone0;
    flashAlone0.place = &flashing;
    flashAlone0.type = INHIBITOR;
    flashAlone0.constraint = 1;
    flashAlone0.timings[0] = 0.0f;
    flashAlone0.timings[1] = FLT_MAX;
    Arc flashAlone1;
    flashAlone1.place = &flashing;
    flashAlone1.type = INHIBITOR;
    flashAlone1.constraint = 1;
    flashAlone1.timings[0] = 0.0f;
    flashAlone1.timings[1] = FLT_MAX;
    Arc flashAlone2;
    flashAlone2.place = &flashing;
    flashAlone2.type = INHIBITOR;
    flashAlone2.constraint = 1;
    flashAlone2.timings[0] = 0.0f;
    flashAlone2.timings[1] = FLT_MAX;
    Arc flashAlone3;
    flashAlone3.place = &flashing;
    flashAlone3.type = INHIBITOR;
    flashAlone3.constraint = 1;
    flashAlone3.timings[0] = 0.0f;
    flashAlone3.timings[1] = FLT_MAX;

    Arc flashAlone01;
    flashAlone01.place = &charged0;
    flashAlone01.type = INPUT;
    flashAlone01.timings[0] = 0.0f;
    flashAlone01.timings[1] = FLT_MAX;
    Arc flashAlone11;
    flashAlone11.place = &charged1;
    flashAlone11.type = INPUT;
    flashAlone11.timings[0] = 0.0f;
    flashAlone11.timings[1] = FLT_MAX;
    Arc flashAlone21;
    flashAlone21.place = &charged2;
    flashAlone21.type = INPUT;
    flashAlone21.timings[0] = 0.0f;
    flashAlone21.timings[1] = FLT_MAX;
    Arc flashAlone31;
    flashAlone31.place = &charged3;
    flashAlone31.type = INPUT;
    flashAlone31.timings[0] = 0.0f;
    flashAlone31.timings[1] = FLT_MAX;

    OutputArc oFlashing;
    oFlashing.isTransport = false;
    oFlashing.output = &flashing;

    Distribution dis3;
    dis3.type = EXPONENTIAL;
    dis3.a = 0.5f;

    Transition aFlashing0;
    aFlashing0.distribution = &dis3;
    aFlashing0.inputArcs[0] = &flashAlone0;
    aFlashing0.inputArcsCount++;
    aFlashing0.inputArcs[1] = &flashAlone01;
    aFlashing0.inputArcsCount++;
    aFlashing0.inputArcs[2] = &aChargedSum;
    aFlashing0.inputArcsCount++;
    aFlashing0.outputArcs[0] = &oFlashing;
    aFlashing0.outputArcsCount++;
    aFlashing0.outputArcs[1] = &oArrive0;
    aFlashing0.outputArcsCount++;

    Transition aFlashing1;
    aFlashing1.distribution = &dis3;
    aFlashing1.inputArcs[0] = &flashAlone1;
    aFlashing1.inputArcsCount++;
    aFlashing1.inputArcs[1] = &flashAlone11;
    aFlashing1.inputArcsCount++;
    aFlashing1.inputArcs[2] = &aChargedSum;
    aFlashing1.inputArcsCount++;
    aFlashing1.outputArcs[0] = &oFlashing;
    aFlashing1.outputArcsCount++;
    aFlashing1.outputArcs[1] = &oArrive1;
    aFlashing1.outputArcsCount++;

    Transition aFlashing2;
    aFlashing2.distribution = &dis3;
    aFlashing2.inputArcs[0] = &flashAlone2;
    aFlashing2.inputArcsCount++;
    aFlashing2.inputArcs[1] = &flashAlone21;
    aFlashing2.inputArcsCount++;
    aFlashing2.inputArcs[2] = &aChargedSum;
    aFlashing2.inputArcsCount++;
    aFlashing2.outputArcs[0] = &oFlashing;
    aFlashing2.outputArcsCount++;
    aFlashing2.outputArcs[1] = &oArrive2;
    aFlashing2.outputArcsCount++;

    Transition aFlashing3;
    aFlashing3.distribution = &dis3;
    aFlashing3.inputArcs[0] = &flashAlone3;
    aFlashing3.inputArcsCount++;
    aFlashing3.inputArcs[1] = &flashAlone31;
    aFlashing3.inputArcsCount++;
    aFlashing3.inputArcs[2] = &aChargedSum;
    aFlashing3.inputArcsCount++;
    aFlashing3.outputArcs[0] = &oFlashing;
    aFlashing3.outputArcsCount++;
    aFlashing3.outputArcs[1] = &oArrive3;
    aFlashing3.outputArcsCount++;

    Arc aAllDone0;
    aAllDone0.place = &chargedSum;
    aAllDone0.type = INHIBITOR;
    aAllDone0.constraint = 1;
    aAllDone0.timings[0] = 0.0f;
    aAllDone0.timings[1] = FLT_MAX;

    Arc aAllDone1;
    aAllDone1.place = &flashing;
    aAllDone1.type = INPUT;
    aAllDone1.constraint = 1;
    aAllDone1.timings[0] = 0.0f;
    aAllDone1.timings[1] = FLT_MAX;

    Distribution dis4;
    dis4.type = CONSTANT;
    dis4.a = 0.0f;


    Place noWhere;
    OutputArc oNoWhere;
    oNoWhere.isTransport = false;
    oNoWhere.output = &noWhere;


    Transition tAllDone;
    tAllDone.distribution = &dis4;
    tAllDone.inputArcs[0] = &aAllDone0;
    tAllDone.inputArcsCount++;
    tAllDone.inputArcs[1] = &aAllDone1;
    tAllDone.inputArcsCount++;
    tAllDone.outputArcs[0] = &oNoWhere;
    tAllDone.outputArcsCount++;
    Arc flashJointly0;
    flashJointly0.place = &charged0;
    flashJointly0.type = INPUT;
    flashJointly0.timings[0] = 0.0f;
    flashJointly0.timings[1] = FLT_MAX;
    Arc flashJointly1;
    flashJointly1.place = &charged1;
    flashJointly1.type = INPUT;
    flashJointly1.timings[0] = 0.0f;
    flashJointly1.timings[1] = FLT_MAX;
    Arc flashJointly2;
    flashJointly2.place = &charged2;
    flashJointly2.type = INPUT;
    flashJointly2.timings[0] = 0.0f;
    flashJointly2.timings[1] = FLT_MAX;
    Arc flashJointly3;
    flashJointly3.place = &charged3;
    flashJointly3.type = INPUT;
    flashJointly3.timings[0] = 0.0f;
    flashJointly3.timings[1] = FLT_MAX;

    Transition tFlashJointly0;
    tFlashJointly0.urgent = true;
    tFlashJointly0.distribution = &dis4;
    tFlashJointly0.inputArcs[0] = &flashJointly0;
    tFlashJointly0.inputArcsCount++;
    tFlashJointly0.inputArcs[1] = &aChargedSum;
    tFlashJointly0.inputArcsCount++;
    tFlashJointly0.outputArcs[0] = &oFlashing;
    tFlashJointly0.outputArcsCount++;
    tFlashJointly0.outputArcs[1] = &oArrive0;
    tFlashJointly0.outputArcsCount++;

    Transition tFlashJointly1;
    tFlashJointly1.urgent = true;
    tFlashJointly1.distribution = &dis4;
    tFlashJointly1.inputArcs[0] = &flashJointly1;
    tFlashJointly1.inputArcsCount++;
    tFlashJointly1.inputArcs[1] = &aChargedSum;
    tFlashJointly1.inputArcsCount++;
    tFlashJointly1.outputArcs[0] = &oFlashing;
    tFlashJointly1.outputArcsCount++;
    tFlashJointly1.outputArcs[1] = &oArrive1;
    tFlashJointly1.outputArcsCount++;

    Transition tFlashJointly2;
    tFlashJointly2.urgent = true;
    tFlashJointly2.distribution = &dis4;
    tFlashJointly2.inputArcs[0] = &flashJointly2;
    tFlashJointly2.inputArcsCount++;
    tFlashJointly2.inputArcs[1] = &aChargedSum;
    tFlashJointly2.inputArcsCount++;
    tFlashJointly2.outputArcs[0] = &oFlashing;
    tFlashJointly2.outputArcsCount++;
    tFlashJointly2.outputArcs[1] = &oArrive2;
    tFlashJointly2.outputArcsCount++;

    Transition tFlashJointly3;
    tFlashJointly3.urgent = true;
    tFlashJointly3.distribution = &dis4;
    tFlashJointly3.inputArcs[0] = &flashJointly3;
    tFlashJointly3.inputArcsCount++;
    tFlashJointly3.inputArcs[1] = &aChargedSum;
    tFlashJointly3.inputArcsCount++;
    tFlashJointly3.outputArcs[0] = &oFlashing;
    tFlashJointly3.outputArcsCount++;
    tFlashJointly3.outputArcs[1] = &oArrive3;
    tFlashJointly3.outputArcsCount++;

    Place *places[14]{&waiting0, &waiting1, &waiting2, &waiting3, &charging0, &charging1, &charging2, &charging3, &charged0,&charged1,&charged2,&charged3, &chargedSum, &flashing};
    net.places = places;
    net.placesCount = 14;

    Transition *transitions[17]{&aArrive0, &aArrive1, &aArrive2, &aArrive3, &aReady0, &aReady1, &aReady2, &aReady3, &aFlashing0, &aFlashing1, &aFlashing2, &aFlashing3, &tAllDone, &tFlashJointly0, &tFlashJointly1, &tFlashJointly2, &tFlashJointly3};
    net.transitions = transitions;
    net.transitionsCount = 17;

    // TokenCountObserver observer;
    // net.addObserver(&observer);
    bool success {false};
    net.run(&success);
    results[tid] += success;
}
__global__ void sum(float *array, int numSimulations)
{
    float total = 0.0f;
    for (int i = 0; i < 512; i++)
    {
        total += array[i];
    }
    printf("euler value is %.11f\n", (double)total / numSimulations);
}
__global__ void summage(float *array, int numSimulations)
{
    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int i = 0; i < numSimulations / 512; i++)
    {
        sum += array[tid + i * 512];
    }

    array[tid] = sum;
}

int main(int argc, char *argv[])
{
    auto start = std::chrono::high_resolution_clock::now();
    float confidence;
    float error;
    int threads = 64;
    int blockCount = 2048;
    if (argc < 3)
    {
        confidence = 0.95f;
        error = 0.001f;
    }
    else
    {
        confidence = std::stof(argv[1]);
        error = std::stof(argv[2]);
    }
    std::cout << "confidence: " << confidence << " error: " << error << std::endl;
    float number = ceil((log(2 / (1 - confidence))) / (2 * error * error));
    std::cout << "number of executions: " << number << std::endl;
    int executionCount = ceil(number / threads);
    int loopCount = ceil(executionCount / blockCount);
    std::cout << "loop count: " << loopCount << std::endl;
    std::cout << "number of executions: " << loopCount * blockCount * threads << std::endl;
    float *d_results;

    checkCudaErrors(cudaMalloc((void **)&d_results, blockCount * threads * sizeof(float)));
    checkCudaErrors(cudaMemset(d_results, 0, blockCount * threads * sizeof(float)));

    for (size_t i = 0; i < loopCount; i++)
    {
        fireflies<<<blockCount, threads>>>(d_results);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    summage<<<1, threads>>>(d_results, blockCount * threads);
    cudaDeviceSynchronize();
    sum<<<1, 1>>>(d_results, loopCount * blockCount * threads);
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