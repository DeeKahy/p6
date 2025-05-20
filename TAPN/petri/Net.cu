#include "Net.h"
__device__ void Tapn::addObserver(SimulationObserver *observer)
{

    cudaMalloc((void **)&observers, sizeof(*observers) + sizeof(*observer));
    observers[observersCount] = observer;
    observersCount++;
}

__device__ void Tapn::notify_observers(const SimulationEvent *event)
{
    // for (size_t i = 0; i < observersCount; i++)
    // {
    //     observers[i].onStep(event);
    // }
}

__device__ void Tapn::step(bool *result)
{
    int fire{-1};
    float missing{FLT_MAX};
    // printf("\nstart currentTime : %f\n", currentTime);
    for (int i = 0; i < transitionsCount; i++)
    {
        bool isReady = false;
        transitions[i].isReady(&isReady, &missing);
        // printf("%d %d %d\n",i,fire,isReady);
        if (isReady)
        {
            if (transitions[i].firingTime == FLT_MAX)
            {
                if (transitions[i].urgent)
                {
                    transitions[i].firingTime = 0.0f;
                }
                else
                {
                    float test;
                    transitions[i].distribution.sample(&state, &test);
                    transitions[i].firingTime = currentTime + test;
                }
            }
            if (fire == -1 ||
                (transitions[i].urgent && !transitions[fire].urgent) ||
                (!transitions[i].urgent && !transitions[fire].urgent && transitions[i].firingTime < transitions[fire].firingTime))
            {
                fire = i;
            }
        }
        else
        {
            transitions[i].firingTime = FLT_MAX;
        }
        // printf("transition %d can fire? %d firing time %f \n", i, isReady, transitions[i].firingTime);
    }
    if (fire != -1)
    {
        if (missing != FLT_MAX)
        {

            if (transitions[fire].firingTime - currentTime <= missing)
            {
                fireTransition(fire, result);
                transitions[fire].firingTime = FLT_MAX;
            }
            else
            {
                updateTokenAges(&missing);
            }
        }
        else
        {
            fireTransition(fire, result);
            transitions[fire].firingTime = FLT_MAX;
        }
    }
    else if (missing != FLT_MAX)
    {
        // printf("missis %f\n", missing);
        updateTokenAges(&missing);
    }
    else
    {
        *result = false;
    }
    // printf("\n end currentTime : %f\n", currentTime);
    // int urgentTransitionIndex = -1;
    // for (size_t i = 0; i < enabledCount; i++)
    // {
    //     int transitionIndex = enabled[i].index;
    //     if (transitions[transitionIndex].urgent)
    //     {
    //         urgentTransitionIndex = transitionIndex;
    //         break;
    //     }
    // }

    // if (urgentTransitionIndex != -1)
    // {
    //     fireTransition(urgentTransitionIndex, result);

    //     delete[] enabled;
    //     return;
    // }

    // EnabledTransition lowestFiringTime = {-1, FLT_MAX};
    // for (size_t i = 0; i < enabledCount; i++)
    // {
    //     int transitionIndex = enabled[i].index;
    //     if (transitions[transitionIndex]->firingTime < lowestFiringTime.firingTime)
    //     {
    //         lowestFiringTime = enabled[i];
    //     }
    // }
    // bool success = false;
    // fireTransition(lowestFiringTime.index, &success);
    // delete[] enabled;
}

__device__ void Tapn::fireTransition(size_t index, bool *result)
{

    float firingTime = transitions[index].firingTime;
    if (firingTime > timeLimit)
    {
        *result = false;
        return;
    }
    steps++;
    if (currentTime > firingTime)
    {
        updateTokenAges(&firingTime);
    }
    else
    {
        firingTime = firingTime - currentTime;
        updateTokenAges(&firingTime);
    }

    // Only advance time if the transition fires in the future

    // else{
    //     steps++;
    // }
    // // steps++;
    float consumed[8]{FLT_MAX};
    int consumedCount{8};
    int consumedAmount;
    // updateTokenAges()
    transitions[index].fire(consumed, consumedCount, &consumedAmount);

    transitionFirings[index]++;
    // steps++;

    *result = true;
}

/**
 * @brief Query function for seeing how many times a transition has fired
 *
 * @param index the transition to query
 * @param result a pointer to where the result should go
 * @return result
 */
__device__ void Tapn::firingCount(int index, int *result)
{
    *result = transitionFirings[index];
}

__device__ void Tapn::run()
{
    init();
    bool result{true};
    while (result)
    {
        step(&result);
    }
}
__device__ void Tapn::run2(bool *success)
{
    init();
    bool result{true};
    while (result)
    {
        // for (size_t i = 0; i < placesCount; i++)
        // {
        //     switch (i)
        //     {
        //     case 0:
        //         printf("waiting 0:");
        //         break;
        //     case 1:
        //         printf("waiting 1:");
        //         break;
        //     case 2:
        //         printf("waiting 2:");
        //         break;
        //     case 3:
        //         printf("waiting 3:");
        //         break;
        //     case 4:
        //         printf("charging 0:");
        //         break;
        //     case 5:
        //         printf("charging 1:");
        //         break;
        //     case 6:
        //         printf("charging 2:");
        //         break;
        //     case 7:
        //         printf("charging 3:");
        //         break;
        //     case 8:
        //         printf("charged 0:");
        //         break;
        //     case 9:
        //         printf("charged 1:");
        //         break;
        //     case 10:
        //         printf("charged 2:");
        //         break;
        //     case 11:
        //         printf("charged 3:");
        //         break;
        //     case 12:
        //         printf("charged Sum:");
        //         break;
        //     case 13:
        //         printf("flashing:");
        //         break;
        //     default:
        //         printf("unknown state:");
        //         break;
        //     }
        //     for (size_t j = 0; j < places[i].tokenCount; j++)
        //     {

        //         printf(" token number :%d value: %f", j, places[i].tokens[j]);
        //     }
        //     printf("\n");
        // }

        // if ((places[4]->tokenCount + places[5]->tokenCount +
        //          places[6]->tokenCount + places[7]->tokenCount >=
        //      1) &&
        //     places[13]->tokenCount == 1 &&
        //     places[0]->tokenCount == 0 && places[1]->tokenCount == 0 &&
        //     places[2]->tokenCount == 0 && places[3]->tokenCount == 0)

        if ((places[4].tokenCount + places[5].tokenCount +
                 places[6].tokenCount + places[7].tokenCount ==
             1) &&
            places[13].tokenCount == 1 &&
            places[0].tokenCount == 0 && places[1].tokenCount == 0 &&
            places[2].tokenCount == 0 && places[3].tokenCount == 0)
        {
            *success = true;
            return;
        }
        step(&result);
    }
}

__device__ void Tapn::shouldContinue(bool *result)
{
    // for (size_t i = 0; i < observersCount; i++)
    // {
    //     observers[i].getShouldStop(result);
    //     if (*result)
    //     {
    //         return;
    //     }
    // }
    // *result = true;
}

__device__ void Tapn::delay()
{
}

__device__ void Tapn::updateTokenAges(float *delay)
{

    currentTime += *delay;
    for (size_t i = 0; i < placesCount; i++)
    {
        for (size_t j = 0; j < places[i].tokenCount; j++)
        {
            places[i].tokens[j] += *delay;
        }
    }
}

__device__ void Tapn::updateEnabledTransitions()
{
    float missing{0};
    bool ready{true};
    for (size_t i = 0; i < transitionsCount; i++)
    {
        transitions[i].isReady(&ready, &missing);
    }
}
__device__ void Tapn::init()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(clock64() + tid, tid, 0, &state);
}