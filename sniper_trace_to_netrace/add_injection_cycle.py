import random

def process_log(input_file, output_file):
    current_number = 0
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            increment = random.randint(1, 10)
            current_number += increment
            outfile.write(f"{current_number} {line}")
    
    print(f"Processing complete. Output saved to '{output_file}'.")

input_file = 'workloads/fft_trace.log'
output_file = 'workloads/fft_trace_2.log'

process_log(input_file, output_file)
