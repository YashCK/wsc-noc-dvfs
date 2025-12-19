import random


def add_column_to_log(input_file, output_file):
    with open(input_file, 'r') as infile:
        with open(output_file, 'w') as outfile:
            for line in infile:
                parts = line.strip().split()
                # Pick a random class per packet (0 or 1 with 50/50 chance)
                class_label = random.randint(0, 1)
                parts.append(str(class_label))
                outfile.write(' '.join(parts) + '\n')


input_file = '../sniper_network_trace_fft.log'
output_file = 'workloads/fft_trace.log'

add_column_to_log(input_file, output_file)
