import sys
import subprocess
import matplotlib.pyplot as plt
import re
import numpy as np

def run_program(num_runs):
    step = num_runs // 10
    args = []
    times = []
    precisions = []  # To store the precision values

    # True value of Euler's number to high precision
    TRUE_E = 2.718281828459045

    for i in range(1, 101):
        argument = i * step
        result = subprocess.run(["./spn.out", str(argument)], 
                              capture_output=True, text=True)

        try:
            # Extract the time from the output
            time_match = re.search(r'time run: (\d+)', result.stdout)

            # Extract the calculated Euler value
            euler_match = re.search(r'euler value is (\d+\.\d+)', result.stdout)

            if time_match and euler_match:
                time_ms = int(time_match.group(1))
                euler_value = float(euler_match.group(1))

                # Calculate precision (absolute error)
                precision = abs(euler_value - TRUE_E)

                args.append(argument)
                times.append(time_ms)
                precisions.append(precision)

                print(f"Run {i}: Argument {argument} - Time {time_ms}ms - Value {euler_value} - Error {precision:.9f}")
            else:
                print(f"Could not find required information in output for argument {argument}")
                print(f"Output: {result.stdout}")
        except Exception as e:
            print(f"Error processing output for argument {argument}: {e}")

    return args, times, precisions

def plot_performance(args, times, precisions):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot execution time
    ax1.plot(args, times, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Argument Value', fontsize=12)
    ax1.set_ylabel('Execution Time (ms)', fontsize=12)
    ax1.set_title('Program Execution Time', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(args)

    # Plot precision (error)
    ax2.plot(args, precisions, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Argument Value', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title('Calculation Precision', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xticks(args)

    # Use scientific notation for small error values if needed
    if min(precisions) < 0.0001:
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    plt.tight_layout()
    plt.savefig('performance_graph.png')
    print("Graph saved as performance_graph.png")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python runner.py <total_number>")
        sys.exit(1)

    try:
        total = int(sys.argv[1])
        arguments, timings, precisions = run_program(total)
        plot_performance(arguments, timings, precisions)
    except ValueError:
        print("Please provide a valid integer")
