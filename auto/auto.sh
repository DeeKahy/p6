#!/bin/bash

number_of_files=$(find ./fireflies/queries -type f | wc -l)
output_file="experiment_results.csv"

# Write CSV header
echo "query_name,runs_executed,execution_time_ms" > "$output_file"

for ((i = 0; i < number_of_files; i++)) do
    current_query="./fireflies/queries/fireflies-queries-$i.xml"
    binary="./verifydtapn-linux64"
    model="./fireflies/fireflies.xml"

    start_time=$(date +%s%3N)  # Start time in milliseconds

    # Run the command and capture output (stdout + stderr)
    output=$("$binary" \
      --k-bound 8 --trace 0 --smc-parallel \
      --smc-obs-scale 500 --smc-print-cumulative-stats 4 \
      --smc-numeric-precision 0 \
      "$model" "$current_query" 2>&1)

    end_time=$(date +%s%3N)  # End time in milliseconds
    elapsed_ms=$((end_time - start_time))

    # Extract the line containing "runs executed:"
    runs_line=$(echo "$output" | grep -i "runs executed:" | xargs)
    
    # Extract runs executed count (just the number)
    runs_executed=$(echo "$output" | grep -i "runs executed:" | awk '{print $3}')

    # Echo result
    echo "fireflies-queries-$i.xml,$runs_executed,$elapsed_ms"

    # Append data to CSV
    echo "fireflies-queries-$i.xml,$runs_executed,$elapsed_ms" >> "$output_file"
done

echo "Results saved to $output_file"

