#include "main.h"

__global__ void euler(float *results)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Tapn net;
    Place place1;
    float token = 0.0f;
    place1.addTokens(&token, 1);
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
    bool test;
    net.run();
    results[tid] = net.steps - 1;
    // net.step(&test);
    // //printf("\n place 0 %f\n", place1.tokens[0]);
    // net.step(&test);
    // net.step(&test);
}
__global__ void fireflies(float *results)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Tapn net;

    float token = 0.0f;
    // starting transitions transitions
    Place waiting0;
    waiting0.addTokens(&token, 1);
    Place waiting1;
    waiting1.addTokens(&token, 1);
    Place waiting2;
    waiting2.addTokens(&token, 1);
    Place waiting3;
    waiting3.addTokens(&token, 1);

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
    Place chargedSum;
    
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
    ready0.place = &waiting0;
    ready0.type = INPUT;
    ready0.timings[0] = 1.0f;
    ready0.timings[1] = FLT_MAX;
    Arc ready1;
    ready1.place = &waiting1;
    ready1.type = INPUT;
    ready1.timings[0] = 1.0f;
    ready1.timings[1] = FLT_MAX;
    Arc ready2;
    ready2.place = &waiting2;
    ready2.type = INPUT;
    ready2.timings[0] = 1.0f;
    ready2.timings[1] = FLT_MAX;
    Arc ready3;
    ready3.place = &waiting3;
    ready3.type = INPUT;
    ready3.timings[0] = 1.0f;
    ready3.timings[1] = FLT_MAX;

    Place charged0;
    Place charged1;
    Place charged2;
    Place charged3;

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

    Transition aReady0;
    aReady0.distribution = &dis2;
    aReady0.inputArcs[0] = &ready0;
    aReady0.inputArcsCount++;
    aReady0.outputArcs[0] = &oReady0;
    aReady0.outputArcsCount++;
    Transition aReady1;
    aReady1.distribution = &dis2;
    aReady1.inputArcs[0] = &ready1;
    aReady1.inputArcsCount++;
    aReady1.outputArcs[0] = &oReady1;
    aReady1.outputArcsCount++;
    Transition aReady2;
    aReady2.distribution = &dis2;
    aReady2.inputArcs[0] = &ready2;
    aReady2.inputArcsCount++;
    aReady2.outputArcs[0] = &oReady2;
    aReady2.outputArcsCount++;
    Transition aReady3;
    aReady3.distribution = &dis2;
    aReady3.inputArcs[0] = &ready3;
    aReady3.inputArcsCount++;
    aReady3.outputArcs[0] = &oReady3;
    aReady3.outputArcsCount++;

    

    Place flashing;
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
    int threads = 512;
    if (argc < 3)
    {
        confidence = 0.95f;
        error = 0.0008f;
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
    std::cout << "number of executions: " << executionCount << std::endl;
    std::cout << "number of executions: " << executionCount * threads << std::endl;
    float *d_results;
    cudaMalloc((void **)&d_results, executionCount * threads * sizeof(float));
    euler<<<executionCount, threads>>>(d_results);
    cudaDeviceSynchronize();
    summage<<<1, threads>>>(d_results, executionCount * threads);
    cudaDeviceSynchronize();
    sum<<<1, 1>>>(d_results, executionCount * threads);
    cudaDeviceSynchronize();
    cudaError_t errSync = cudaDeviceSynchronize();
    cudaError_t errAsync = cudaGetLastError();

    if (errSync != cudaSuccess)
    {
        // printf("Sync error: %s\n", cudaGetErrorString(errSync));
    }
    if (errAsync != cudaSuccess)
    {
        // printf("Launch error: %s\n", cudaGetErrorString(errAsync));
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    cudaFree(d_results);
    std::cout << "time run: " << duration.count() << std::endl;
    return 0;
}