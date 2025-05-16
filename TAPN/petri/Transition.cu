#include "Transition.h"

/**
 * @brief Checks if a transition is ready to be fired
 *
 * @param result a pointer to a bool to indicate if it is ready
 * @return bool
 */
__device__ void Transition::isReady(bool *result, float *missing)
{
    // //printf("%d",inputArcsCount);
    for (size_t i = 0; i < inputArcsCount; i++)
    {

        // Check if transition can fire
        bool transitionCanFire = false;
        inputArcs[i]->canFire(&transitionCanFire, missing);

        if (!transitionCanFire)
        {
            *result = false;
            // printf("\nfailed\n");
            return;
        }
    }

    // if (!urgent)
    // {
    //     // //printf("got here");
    //     float test;
    //     distribution->sample(&test);
    //     firingTime = test;
    // }
    // else
    // {
    //     firingTime = 0.0f;
    // }

    *result = true;
}

/**
 * @brief Function for firing a transition
 *
 * @param consumed an array for the consumed tokens
 * @param consumedCount the maximum length of the consumed array
 * @param consumedAmout a pointer to tell how many tokens were consumed
 * @return The filled array and the amount consumed
 */
__device__ void Transition::fire(float *consumed, int consumedCount, int *consumedAmout)
{
    for (size_t i = 0; i < inputArcsCount; i++)
    {
        switch (inputArcs[i]->type)
        {
        case INPUT:
            inputArcs[i]->fire(consumed, consumedAmout);
            break;
        case TRANSPORT:
            inputArcs[i]->fire(consumed, consumedAmout);
            break;
        case INHIBITOR:
            // Inhibitor arcs don't consume tokens
            inputArcs[i]->fire(consumed, consumedAmout);
            break;
        // default:
        //     // //printf("could not find type");
        //     break;
        }
    }

    if (outputArcsCount == 0)
    {
        // //printf("No output arcs \n");
        return;
    }

    for (size_t i = 0; i < outputArcsCount; i++)
    {
        bool success;
        // //printf("Firing outputs \n");
        if (outputArcs[i]->isTransport)
        {
            for (size_t i = 0; i < *consumedAmout; i++)
            {
                consumed[i] += firingTime;
            }
            
            outputArcs[i]->fire(consumed, *consumedAmout, &success);
        }
        else
        {

            outputArcs[i]->fire(consumed, *consumedAmout, &success);
        }
        // //printf("Firing outputs \n");
    }
}

/**
 * @brief samples distribution function based on the type
 *
 * @param result a pointer to a float to return to
 * @return float
 */
__device__ void Distribution::sample(float *result)
{

    switch (type)
    {
    case CONSTANT:
        // For constant values a is used as the value returned
        *result = a;
        break;
    case UNIFORM:
        // a is used for the minimum value created by the uniform distribution
        // b is used for the maximum value created by the uniform distribution
        *result = (a + curand_uniform(&state) * (b - a));
        break;
    case NORMAL:
        // a is used for the for the mean by the normal distribution
        // b is used for the maximum value created by the uniform distribution
        *result = a + b * curand_normal(&state);
        ;
        break;
    case EXPONENTIAL:
        *result = -logf(curand_uniform(&state)) / a;
        break; // more distributions to come
    }
}
__device__ void Distribution::init()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(clock64() + tid, tid, 0, &state);
}