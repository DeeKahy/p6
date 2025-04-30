#include "fireflies.h"
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