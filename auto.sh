
output_file="experiment_results.txt"

# Clear the output file if it exists
> "$output_file"

# Hardcoded parameters
parameters=( 0.01 0.005 0.003 0.002 0.001 0.0009 0.0008 0.0007 0.0006 0.0005 0.0004 0.0003 0.0002 0.0001 0.00009 0.00008 0.00007 0.00006 0.00005 0.00004 0.00003 0.00002 0.00001 0.000009 0.000008 0.000007 0.000006 0.000005 0.000004 0.000003 0.000002 0.000001 )
fixed_arg="0.95"

for param in "${parameters[@]}"; do
    # Add a header and separator before each run
    echo "RUNNING: ./fireflies $fixed_arg $param" | tee -a "$output_file"
    echo "------------------------------------------------" | tee -a "$output_file"

    # Run command and capture all output directly to file and terminal
    steam-run ./fireflies "$fixed_arg" "$param" 2>&1 | tee -a "$output_file"

    # Add time information and separator after the run
    echo "" | tee -a "$output_file"
    echo "COMPLETED AT: $(date)" | tee -a "$output_file"
    echo "================================================" | tee -a "$output_file"
    echo "" | tee -a "$output_file"
done

echo "All results saved to $output_file"
