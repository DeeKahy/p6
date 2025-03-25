import sys
import subprocess
import matplotlib.pyplot as plt
import re

def run_program(num_runs):
    step = num_runs // 10
    args = []
    times = []

    for i in range(1, 11):
        argument = i * step
        result = subprocess.run(["./spn.out", str(argument)], 
                              capture_output=True, text=True)

        try:
            # Extract the time from the output using regex to find "time run: X"
            time_match = re.search(r'time run: (\d+)', result.stdout)

            if time_match:
                time_ms = int(time_match.group(1))
                args.append(argument)
                times.append(time_ms)
                print(f"Run {i}: Argument {argument} - Time {time_ms}ms")
            else:
                print(f"Could not find time information in output for argument {argument}")
                print(f"Output: {result.stdout}")
        except Exception as e:
            print(f"Error processing output for argument {argument}: {e}")

    return args, times

def plot_performance(args, times):
    plt.figure(figsize=(10, 6))
    plt.plot(args, times, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Argument Value', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title('Program Performance Analysis', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(args)
    plt.savefig('performance_graph.png')
    print("Graph saved as performance_graph.png")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python runner.py <total_number>")
        sys.exit(1)

    try:
        total = int(sys.argv[1])
        arguments, timings = run_program(total)
        plot_performance(arguments, timings)
    except ValueError:
        print("Please provide a valid integer")
