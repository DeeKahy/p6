#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <iostream>

struct Transition
{
    int* in;
    int in_count;
    int* out;
    int out_count;
    float firing_time;
    bool ready = false;
};

struct SPN
{
    Transition* transition;
    int transition_count;
    int* places;
    int places_count;
};

__global__ void simulate_spn(SPN* SPN, int steps) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 0; i < SPN->transition_count; i++) {
        bool is_ready = true;

        for (int j = 0; j < SPN->transition[i].in_count; j++)
        {
            int place_index = SPN->transition[i].in[j];
            printf("Checking place[%d]: %d\n", place_index, SPN->places[place_index]);

            if (SPN->places[place_index] <= 0) {
                is_ready = false;
                break;
            }
        }

        SPN->transition[i].ready = is_ready;

        // Initialize random state
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        float u = curand_uniform(&state);
        SPN->transition[i].firing_time = -logf(u) / 2.0f;
    }

    // Find transition with the lowest firing time
    Transition* next_transition = nullptr;
    for (int i = 0; i < SPN->transition_count; i++) {
        if (SPN->transition[i].ready &&
            (next_transition == nullptr || SPN->transition[i].firing_time < next_transition->firing_time)) {
            next_transition = &SPN->transition[i];
        }
    }

    // Ensure next_transition is valid
    if (!next_transition) {
        printf("No transition is ready.\n");
        return;
    }

    // Fire the transition
    for (int i = 0; i < next_transition->in_count; i++) {
        int index = next_transition->in[i];
        SPN->places[index] -= 1;
    }

    for (int i = 0; i < next_transition->out_count; i++) {
        int index = next_transition->out[i];
        SPN->places[index] += 1;
    }

    printf("Fired transition with firing time: %f\n", next_transition->firing_time);
    next_transition->firing_time = 0;
    next_transition->ready = false;
}

int main() {
    int h_places[3] = { 1,1,0 };
    int h_in1[] = { 0 }, h_out1[] = { 1 };
    int h_in2[] = { 1 }, h_out2[] = { 2 };
    int h_in3[] = { 2 }, h_out3[] = { 0 };

    Transition h_transitions[3];

    // Device memory allocation
    SPN* d_spn;
    Transition* d_transitions;
    int* d_places;
    int* d_in1, * d_out1, * d_in2, * d_out2, * d_in3, * d_out3;

    cudaMalloc(&d_spn, sizeof(SPN));
    cudaMalloc(&d_transitions, sizeof(h_transitions));
    cudaMalloc(&d_places, sizeof(h_places));

    cudaMalloc(&d_in1, sizeof(h_in1));
    cudaMalloc(&d_out1, sizeof(h_out1));
    cudaMalloc(&d_in2, sizeof(h_in2));
    cudaMalloc(&d_out2, sizeof(h_out2));
    cudaMalloc(&d_in3, sizeof(h_in3));
    cudaMalloc(&d_out3, sizeof(h_out3));

    // Copy data to device
    cudaMemcpy(d_places, h_places, sizeof(h_places), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in1, h_in1, sizeof(h_in1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out1, h_out1, sizeof(h_out1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, h_in2, sizeof(h_in2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out2, h_out2, sizeof(h_out2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in3, h_in3, sizeof(h_in3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out3, h_out3, sizeof(h_out3), cudaMemcpyHostToDevice);

    // Initialize transitions with device pointers
    h_transitions[0] = { d_in1, 1, d_out1, 1, 0, false };
    h_transitions[1] = { d_in2, 1, d_out2, 1, 0, false };
    h_transitions[2] = { d_in3, 1, d_out3, 1, 0, false };

    cudaMemcpy(d_transitions, h_transitions, sizeof(h_transitions), cudaMemcpyHostToDevice);

    // Initialize SPN struct
    SPN h_spn;
    h_spn.transition = d_transitions;
    h_spn.transition_count = 3;
    h_spn.places = d_places;
    h_spn.places_count = 3;

    cudaMemcpy(d_spn, &h_spn, sizeof(SPN), cudaMemcpyHostToDevice);

    // Launch kernel
    simulate_spn << <1, 1 >> > (d_spn, 20);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(h_places, d_places, sizeof(h_places), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Final places: ";
    for (int i = 0; i < 3; i++) {
        std::cout << h_places[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    cudaFree(d_transitions);
    cudaFree(d_places);
    cudaFree(d_spn);
    cudaFree(d_in1);
    cudaFree(d_out1);
    cudaFree(d_in2);
    cudaFree(d_out2);
    cudaFree(d_in3);
    cudaFree(d_out3);

    return 0;
}
