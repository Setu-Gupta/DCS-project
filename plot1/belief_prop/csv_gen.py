# The first argument to this script is the number of threads, and the second argument is the output csv file name

import sys
sys.path.append('../../utils')

from ldpc import create_parity_matrix, encode, belief_propagation_decode
from qpsk import transmit, apply_channel_noise, receive, get_prob_single_bit_flip
import numpy as np
import random
from math import log10, sqrt
import multiprocessing as mp
from queue import Empty

sim_iters = 1000000000              # Number of iterations to run to compute the probability

def get_error_prob(H, snr, snr_db, max_iters, mtype, num_message_bits, sim_iters):
    total_bits = 0
    bit_errors = 0
    for sim_iter in range(sim_iters):
        for _ in range(max_iters):
        
            # Step 1: Create the message block of length num_message_bits
            message_block = [0 for x in range(num_message_bits)]

            # Step 2: Calculate the number of S:I:F messages which need to transmitted
            num_bits = 1 + mtype[0] + mtype[1]      # Number of bits in a single message
            num_messages = num_message_bits // num_bits # Number of messages in a message block
            useful_bits = num_message_bits - (num_message_bits % num_bits) # Tracks the number of useful bits in the message
            
            # Step 3: Populate the block accordingly
            block_idx = 0
            for i in range(num_messages):
                message = [random.randint(0, 1) for i in range(num_bits)]
                for bit in message:
                    message_block[block_idx] = bit
                    block_idx += 1

            # Step 4: Encode the message and pass it via the channel
            encoded_message = encode(H, message_block)

            # Step 5: Transmit the encoded message via the QPSK transmitter
            signal, extended = transmit(encoded_message, Eb) 
            
            # Step 6: Pass the message via the noisy channel
            noisy_signal = apply_channel_noise(signal, snr, Eb)

            # Step 7: Receive the noisy signal
            received_encoded_message_block = receive(noisy_signal, extended)

            # Step 8: Pass the received message through the LDPC decoder
            prob_bit_flip = get_prob_single_bit_flip(snr)
            decoded_message, success = belief_propagation_decode(H, received_encoded_message_block, prob_bit_flip, max_iters)
            
            # Step 9: Find the number of bit errors 
            for sent, received in zip(message_block, decoded_message):
                total_bits += 1
                if sent != received:
                    bit_errors += 1

    error_prob = bit_errors / total_bits
    label = str(mtype[0]) + ':' + str(mtype[1]) + " message (Belief Propagation Decoder)"
    output = str(snr_db) + ', ' + str(error_prob) + ',' + label
    return output    

def worker_wrapper(jobs, shared_list, shared_list_lock, H, max_iters, num_message_bits, sim_iters):
    while True:
        try:
            job = jobs.get(timeout=1)
            snr = job[0]
            snr_db = job[1]
            mtype = job[2]
            output = get_error_prob(H, snr, snr_db, max_iters, mtype, num_message_bits, sim_iters)
            with shared_list_lock:
                shared_list.append(output)
        except Empty:
            break

# Set up the parameters for simulation
n = 1536                            # Codeword/Block Length
r = 0.5                             # Code rate
num_message_bits = int(n * r)       # Number of message bits in the block
max_iters = 1000                    # Number of iterations to run the decoder for

# Create the encoding parity matrix
H = create_parity_matrix(n, num_message_bits)   # Generate the encoding and decoding matrices

# Compute the SNR values
Eb = 1                                      # Assuming unit average bit energy energy
snr_db = [1 + 0.2*x for x in range(11)]     # Range of SNR values in decibels
snr_linear = [10**(x/10) for x in snr_db]   # Range of SNR values in linear scale

# Set up the sweep parameters
message_types = [[1,4], [2,2], [2,3], [2,4]]

# Create jobs for parallel workers
jobs = mp.Queue() # Create a job queue for workers
for snr, snr_db in zip(snr_linear, snr_db):
    for mtype in message_types:
        jobs.put((snr, snr_db, mtype))

# Create a manager for the shared list
manager = mp.Manager()
shared_list = manager.list()
shared_list_lock = manager.Lock()

# Get the number of workers
if len(sys.argv) < 3:
    print("Error: The first argument to this script is the number of threads, and the second argument is the output csv file name")
    exit()
num_processes = int(sys.argv[1])

# Start parallel execution
workers = []
for worker in range(num_processes):
    process = mp.Process(target = worker_wrapper, args = (jobs, shared_list, shared_list_lock, H, max_iters, num_message_bits, sim_iters))
    process.start()
    workers.append(process)

# Join and cleanup the workers
for w in workers:
    w.join()
workers.clear()

# Sort the output for plotting
sorted_list = []
with shared_list_lock:
    for val in shared_list:
        sorted_list.append(val)
sorted_list.sort()

# Print the output on stdout
with open(sys.argv[2], 'w') as csv_file:
    for val in sorted_list:
        csv_file.write(val + '\n')
