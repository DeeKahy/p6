#include "Net.h"
__device__ void Tapn::addObserver(SimulationObserver *observer)
{

    cudaMalloc((void **)&observers, sizeof(*observers) + sizeof(*observer));
    observers[observersCount] = observer;
    observersCount++;
}

__device__ void Tapn::notify_observers(const SimulationEvent *event)
{
    for (size_t i = 0; i < observersCount; i++)
    {
        observers[i]->onStep(event);
    }
}

__device__ void Tapn::step(bool *result)
{
    updateEnabledTransitions();

    EnabledTransition *enabled = new EnabledTransition[transitionsCount];
    int enabledCount = 0;

    for (int i = 0; i < transitionsCount; i++)
    {
        bool isReady = true;
        transitions[i]->isReady(&isReady);
        if (isReady)
        {
            enabled[enabledCount] = {i, transitions[i]->firingTime};
            enabledCount++;
        }
    }
    // //printf("%d\n", enabledCount);

    // No enabled transitions
    if (enabledCount == 0)
    {
        *result = false;
        delete[] enabled;
        return;
    }

    int urgentTransitionIndex = -1;
    for (size_t i = 0; i < enabledCount; i++)
    {
        int transitionIndex = enabled[i].index;
        if (transitions[transitionIndex]->urgent)
        {
            urgentTransitionIndex = transitionIndex; // Store actual transition index
            break;                                   // Stop at first urgent transition found
        }
    }

    if (urgentTransitionIndex != -1)
    {
        fireTransition(urgentTransitionIndex, result);
        delete[] enabled;
        return;
    }

    EnabledTransition lowestFiringTime = {-1, FLT_MAX};
    for (size_t i = 0; i < enabledCount; i++)
    {
        int transitionIndex = enabled[i].index;
        if (transitions[transitionIndex]->firingTime < lowestFiringTime.firingTime)
        {
            lowestFiringTime = enabled[i];
        }
    }

    // If total delayed time is usefull at some point we can implement delay and start using it :shrugs:
    // delay();
    bool success = false;
    fireTransition(lowestFiringTime.index, &success);
    delete[] enabled;
}

__device__ void Tapn::fireTransition(size_t index, bool *result)
{
    float firingTime = transitions[index]->firingTime;

    SimulationEvent event;
    event.type = TRANSITION_FIRING;
    event.firing = {(int)index, firingTime};
    SimulationEvent preEvent = event;
    notify_observers(&preEvent);

    updateTokenAges(&firingTime);

    float consumed[8]{FLT_MAX};
    int consumedCount{8};
    int consumedAmount;
    transitions[index]->fire(consumed, consumedCount, &consumedAmount);

    transitionFirings[index]++;
    steps++;
    // //printf("%d",steps);
    // more observer stuff here

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

__device__ void Tapn::run(bool *success)
{
    bool result;
    shouldContinue(&result);
    // for (size_t i = 0; i < count; i++)
    // {
    //     /* code */
    // }
    // for (size_t i = 0; i < 5; i++)
    // {
    //     step(&result);
    // }
    // printf("First sanity check - place pointers:\n");
    // for (size_t i = 0; i < placesCount; i++) {
    //     printf("Place %zu pointer: %p, token count: %d \n",
    //            i, places[i], places[i]->tokenCount);
    // }
    while (result)
    {
        if (places[4]->tokenCount = 1 && places[5]->tokenCount == 1 && places[6]->tokenCount == 1 && places[7]->tokenCount == 1 &&
            places[13]->tokenCount == 1 &&
            places[0]->tokenCount == 0 && places[1]->tokenCount == 0 && places[2]->tokenCount == 0 && places[3]->tokenCount == 0)
        {
            *success = true;
            // printf("_______________________________________________SUCCESS__________________________");
            return;
        }
        // for (size_t i = 0; i < placesCount; i++)
        // {
        //     printf("\nplace %i ", i);
        //     for (size_t j = 0; j < places[i]->tokenCount; j++)
        //     {
        //         printf("token number%d %f ", j, places[i]->tokens[j]);
        //     }
        // }

        step(&result);
        if (steps >= 30)
        {
            *success = false;
            return;
        }
    }
    // for (size_t i = 0; i < observersCount; i++)
    // {
    //     observers[i].onCompletion();
    // }
}

__device__ void Tapn::shouldContinue(bool *result)
{
    for (size_t i = 0; i < observersCount; i++)
    {
        observers[i]->getShouldStop(result);
        if (*result)
        {
            return;
        }
    }
    *result = true;
}

__device__ void Tapn::delay()
{
}

__device__ void Tapn::updateTokenAges(float *delay)
{

    if (*delay <= 0.0f)
    {
        return;
    }
    currentTime += *delay;
    for (size_t i = 0; i < placesCount; i++)
    {
        for (size_t j = 0; j < places[i]->tokenCount; j++)
        {
            // printf("\ntoken before%f\n",places[i]->tokens[j]);
            places[i]->tokens[j] += *delay;
            // printf("\ntoken after%f\n",places[i]->tokens[j]);
        }
    }
}

__device__ void Tapn::updateEnabledTransitions()
{
    for (size_t i = 0; i < transitionsCount; i++)
    {
        bool ready;
        transitions[i]->isReady(&ready);
    }
}