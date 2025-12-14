def add_column_to_log(input_file, output_file, new_column_value):
    with open(input_file, 'r') as infile:
        with open(output_file, 'w') as outfile:
            for line in infile:
                parts = line.strip().split()
                parts.append(str(new_column_value))
                outfile.write(' '.join(parts) + '\n')

input_file = '../sniper_network_trace_fft.log'
output_file = 'workloads/fft_trace.log'
new_column_value = 0

add_column_to_log(input_file, output_file, new_column_value)
