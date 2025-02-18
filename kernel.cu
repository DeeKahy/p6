#include <cuda.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#define THREADS_PER_BLOCK 256

// Check GPU Memory
void check_cuda_memory() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "GPU Memory Usage: "
        << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB / "
        << total_mem / (1024.0 * 1024.0) << " MB\n";
}

// CUDA Kernel: Initialize Random States
__global__ void init_random(curandState* states, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &states[id]);
}

// CUDA Kernel: Petri Net Simulation
__global__ void petri_net_kernel(int* places, int* transitions_input, int* transitions_output,
    int steps, int* results, curandState* random_states) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < gridDim.x * blockDim.x) {
        int current_state[3];
        for (int i = 0; i < 3; i++) {
            current_state[i] = places[i];
        }

        curandState localState = random_states[thread_id];

        for (int step = 0; step < steps; step++) {
            for (int i = 0; i < 3; i++) {
                results[thread_id * steps * 3 + step * 3 + i] = current_state[i];
            }

            // Generate random number using curand
            if (curand(&localState) % 2 == 0) {
                // T1 then T2
                if (current_state[0] >= transitions_input[0]) {
                    for (int i = 0; i < 3; i++) {
                        current_state[i] -= transitions_input[i];
                        current_state[i] += transitions_output[i];
                    }
                }
                if (current_state[1] >= transitions_input[3]) {
                    for (int i = 0; i < 3; i++) {
                        current_state[i] -= transitions_input[3 + i];
                        current_state[i] += transitions_output[3 + i];
                    }
                }
            }
            else {
                // T2 then T1
                if (current_state[1] >= transitions_input[3]) {
                    for (int i = 0; i < 3; i++) {
                        current_state[i] -= transitions_input[3 + i];
                        current_state[i] += transitions_output[3 + i];
                    }
                }
                if (current_state[0] >= transitions_input[0]) {
                    for (int i = 0; i < 3; i++) {
                        current_state[i] -= transitions_input[i];
                        current_state[i] += transitions_output[i];
                    }
                }
            }
        }

        random_states[thread_id] = localState; // Save updated state
    }
}

// CUDA Kernel: Compute Statistics
__global__ void stats_kernel(int* results, float* stats_output, long long num_simulations, int steps) {
    int step = blockIdx.x * blockDim.x + threadIdx.x;
    if (step < steps) {
        for (int place = 0; place < 3; place++) {
            float sum_val = 0.0f, sum_sq = 0.0f;
            float min_val = INT_MAX, max_val = INT_MIN;

            for (int sim = 0; sim < num_simulations; sim++) {
                float val = results[sim * steps * 3 + step * 3 + place];
                sum_val += val;
                sum_sq += val * val;
                min_val = fmin(min_val, val);
                max_val = fmax(max_val, val);
            }

            float mean = sum_val / num_simulations;
            float variance = (sum_sq / num_simulations) - (mean * mean);

            stats_output[step * 3 * 4 + place * 4 + 0] = mean;
            stats_output[step * 3 * 4 + place * 4 + 1] = variance;
            stats_output[step * 3 * 4 + place * 4 + 2] = min_val;
            stats_output[step * 3 * 4 + place * 4 + 3] = max_val;
        }
    }
}

// GPUPetriNet class
class GPUPetriNet {
public:
    long long num_simulations;
    int places[3] = { 2, 0, 0 };
    int transitions_input[6] = { 1, 0, 0, 0, 1, 0 };
    int transitions_output[6] = { 0, 1, 0, 0, 0, 1 };

    GPUPetriNet(long long num_simulations) : num_simulations(num_simulations) {}

    void run_simulation_and_analyze(int steps, std::vector<float>& stats, std::vector<int>& sample_results) {
        check_cuda_memory(); // Before allocation

        int* d_places, * d_transitions_input, * d_transitions_output, * d_results;
        float* d_stats;
        curandState* d_random_states;
        size_t results_size = num_simulations * steps * 3 * sizeof(int);
        size_t stats_size = steps * 3 * 4 * sizeof(float);

        cudaMalloc(&d_places, 3 * sizeof(int));
        cudaMalloc(&d_transitions_input, 6 * sizeof(int));
        cudaMalloc(&d_transitions_output, 6 * sizeof(int));
        cudaMalloc(&d_results, results_size);
        cudaMalloc(&d_stats, stats_size);
        cudaMalloc(&d_random_states, num_simulations * sizeof(curandState));

        check_cuda_memory(); // After allocation

        // Copy data
        cudaMemcpy(d_places, places, 3 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_transitions_input, transitions_input, 6 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_transitions_output, transitions_output, 6 * sizeof(int), cudaMemcpyHostToDevice);

        // Initialize random states
        int blocks_per_grid = (num_simulations + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        init_random << <blocks_per_grid, THREADS_PER_BLOCK >> > (d_random_states, time(NULL));

        // Run Simulation
        petri_net_kernel << <blocks_per_grid, THREADS_PER_BLOCK >> > (d_places, d_transitions_input, d_transitions_output,
            steps, d_results, d_random_states);
        cudaDeviceSynchronize();

        check_cuda_memory(); // After kernel execution

        int stats_blocks = (steps + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        stats_kernel << <stats_blocks, THREADS_PER_BLOCK >> > (d_results, d_stats, num_simulations, steps);
        cudaDeviceSynchronize();

        check_cuda_memory(); // After stats computation

        // Copy results back to CPU
        stats.resize(steps * 3 * 4);
        sample_results.resize(num_simulations * steps * 3);
        cudaMemcpy(sample_results.data(), d_results, results_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(stats.data(), d_stats, stats_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(sample_results.data(), d_results, 3 * steps * 3 * sizeof(int), cudaMemcpyDeviceToHost);

        // Free memory
        cudaFree(d_places);
        cudaFree(d_transitions_input);
        cudaFree(d_transitions_output);
        cudaFree(d_results);
        cudaFree(d_stats);
        cudaFree(d_random_states);

        check_cuda_memory(); // After freeing memory
    }
};
void print_sample_results(const std::vector<int>& sample_results, int steps) {
    std::cout << "\nSample Simulation Results:\n";
    const char* place_names[] = { "P1", "P2", "P3" };

    for (int step = 0; step < steps; step++) {
        std::cout << "Step " << step + 1 << ": ";
        for (int place = 0; place < 3; place++) {
            std::cout << place_names[place] << "=" << sample_results[step * 3 + place] << " ";
        }
        std::cout << "\n";
    }
}
void print_analysis(const std::vector<float>& stats, double execution_time, int steps) {
    std::cout << "\nExecution time: " << execution_time << " seconds\n";

    std::cout << "\nFinal State Statistics:\n";
    const char* place_names[] = { "P1", "P2", "P3" };

    for (int place = 0; place < 3; place++) {
        std::cout << "\n" << place_names[place] << ":\n";
        std::cout << "  Mean tokens: " << stats[(steps - 1) * 3 * 4 + place * 4] << "\n";
        std::cout << "  Std Dev: " << sqrt(stats[(steps - 1) * 3 * 4 + place * 4 + 1]) << "\n";
    }
}

int main() {
    std::cout << "Running 10000 Petri net simulations...\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    GPUPetriNet net(1000000);
    std::vector<float> stats;
    std::vector<int> sample_results;
    net.run_simulation_and_analyze(20, stats, sample_results);

    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "Execution time: " << execution_time << " seconds\n";
    print_analysis(stats, execution_time, 20);
    // Print a sample of the results

    return 0;
}
