#include "Net.h"
struct Tapn
{
    Place *places;
    int placesCount{0};
    Transition *transitions;
    int transitionsCount{0};
    SimulationObserver *observers;
    int observersCount{0};
    uint steps{0};
    float currentTime{0.0f};
    int transitionFirings[20]{0};

    __device__ void addObserver(SimulationObserver *observer);
    __device__ void notify_observers(const SimulationEvent *event);
    __device__ void step(bool *result);
    __device__ void fireTransition(size_t index, bool *result);
    __device__ void firingCount(int index, int *result);
    __device__ void run();
    __device__ void shouldContinue(bool *result);
    __device__ void delay();
    __device__ void updateTokenAges(float *delay);
    __device__ void updateEnabledTransitions();
};

struct EnabledTransition
{
    int index;
    float firingTime;
};

__device__ void Tapn::addObserver(SimulationObserver *observer)
{
    observersCount++;
    realloc(observers, observersCount * sizeof(SimulationObserver));
    observers[observersCount - 2] = *observer;
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

    EnabledTransition enabled[transitionsCount];
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

    EnabledTransition lowestFiringTime = {-1, MAXFLOAT};
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
    while (!shouldContinue)
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