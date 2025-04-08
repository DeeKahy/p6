#include "main.h"

__global__ void simulate()
{
    Place place1;
    place1.id = 0;
    place1.tokens[0] = 1.0f;
    place1.tokenCount = 1;

    Place place2;
    place2.id = 1;
    place2.tokenCount = 0;

    Arc input;
    input.type = INPUT;
    input.place = &place1;
    input.weight = 1;
    input.timings[0] = {1.0};
    input.timings[1] = MAXFLOAT;

    OutputArc output;
    output.output = &place2;
    output.weight = 1;
    output.isTransport = true;

    Distribution dist;
    dist.type = CONSTANT;
    dist.a = 1.0f;

    Transition trans;
    trans.inputArcs[0] = {input};
    trans.inputArcsCount = 1;
    trans.outputArcs[0] = {output};
    trans.outputArcsCount = 1;
    trans.distribution = dist;
    trans.urgent = false;
    trans.id = 0;

    float consumed[2];
    int consumedCount = 2;
    int consumedAmount;
    printf("place1.tokens: %f \n",place1.tokens[0]);
    printf("place2.tokens: %f \n",place2.tokens[0]);
    trans.fire(consumed, consumedCount, &consumedAmount);
    printf("place1.tokens: %f \n",place1.tokens[0]);
    printf("place2.tokens: %f \n",place2.tokens[0]);
}

int main()
{
    simulate<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}