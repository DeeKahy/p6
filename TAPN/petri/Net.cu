#include "Net.h"
 __device__ void Tapn::addObserver(SimulationObserver *observer)
 {

     //observersCount++;
     //observers[observersCount - 2] = observer;
 }

__device__ void Tapn::notify_observers(const SimulationEvent *event)
{
    for (size_t i = 0; i < observersCount; i++)
    {
        observers[i].onStep(event);
    }
}

__device__ void Tapn::step(bool *result)
{
    updateEnabledTransitions();

    EnabledTransition* enabled = new EnabledTransition[transitionsCount];
    int enabledCount = 0;

    for (int i = 0; i < transitionsCount; i++)
    {
        bool isReady = false;
        transitions[i].isReady(&isReady);
        if (isReady)
        {
            enabled[enabledCount] = {i, transitions[i].firingTime};
            enabledCount++;
        }
    }

    // No enabled transitions
    if (enabledCount == 0)
    {
        *result = false;
        return;
    }

    int urgentTransitionIndex = -1;
    for (size_t i = 0; i < enabledCount; i++)
    {
        int transitionIndex = enabled[i].index;
        if (transitions[transitionIndex].urgent)
        {
            urgentTransitionIndex = i;
        }
    }

    if (urgentTransitionIndex != -1)
    {
        fireTransition(urgentTransitionIndex, result);
        return;
    }

    EnabledTransition lowestFiringTime = {-1, FLT_MAX };
    for (size_t i = 0; i < enabledCount; i++)
    {
        int transitionIndex = enabled[i].index;
        if (transitions[transitionIndex].firingTime < lowestFiringTime.firingTime)
        {
            lowestFiringTime = enabled[i];
        }
    }

    // If total delayed time is usefull at some point we can implement delay and start using it :shrugs:
    // delay();
    bool success = false;
    fireTransition(lowestFiringTime.index, &success);
}

__device__ void Tapn::fireTransition(size_t index, bool *result)
{
    float firingTime = transitions[index].firingTime;

    SimulationEvent event;
    event.type = TRANSITION_FIRING;
    event.firing = {(int)index, firingTime};
    SimulationEvent preEvent = event;
    notify_observers(&preEvent);

    updateTokenAges(&firingTime);

    float consumed[10];
    int consumedCount{10};
    int consumedAmount;
    transitions[index].fire(consumed, consumedCount, &consumedAmount);

    transitionFirings[index]++;
    steps++;

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

__device__ void Tapn::run()
{
    bool result;
    shouldContinue(&result);
    while (!result)
    {
        bool result;
        step(&result);
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
        observers[i].getShouldStop(result);
        if (*result)
        {
            return;
        }
    }
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
        for (size_t j = 0; j < places[i].tokenCount; j++)
        {
            places[i].tokens[j] += *delay;
        }
    }
}

__device__ void Tapn::updateEnabledTransitions()
{
    for (size_t i = 0; i < transitionsCount; i++)
    {
        bool ready;
        transitions[i].isReady(&ready);
    }
}