#pragma once
#include "Place.h"
#include "Arcs.h"
#include "Invariant.h"
#include "Transition.h"
#include "../observer/Observer.h"

struct Tapn
{
    Place **places;
    int placesCount{0};
    Transition **transitions;
    int transitionsCount{0};
    SimulationObserver **observers;
    int observersCount{0};
    int steps{0};
    float currentTime{0.0f};
    int transitionFirings[20]{0};
    int failedAttempt{0};

    __device__ void addObserver(SimulationObserver *observer);
    __device__ void notify_observers(const SimulationEvent *event);
    __device__ void step(bool *result);
    __device__ void fireTransition(size_t index, bool *result);
    __device__ void firingCount(int index, int *result);
    __device__ void run();
    __device__ void run2(bool *success);
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