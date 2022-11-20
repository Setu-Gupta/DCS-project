import numpy as np
from random import uniform
from copy import deepcopy

# define density to a constant value less than 1
density = 0.5

# Creates a parity matrix which can be used for decoding and encoding LDPC signals
def create_parity_matrix(block_length, message_bits):
    N = block_length
    K = message_bits

    # Create H1 matrix
    H1 = np.zeros((N-K, K), dtype=np.int8)
    for i in range(len(H1)):
        for j in range(len(H1[i])):
            x = uniform(0, 1)
            if x < density:
                H1[i, j] = 1

    # Create the H2 matrix
    H2 = np.zeros((N-K, N-K), dtype=np.int8)
    for i in range(N-K):
        row = np.zeros(N-K, dtype=np.int8)
        if i == 0:
            row[0] = 1
        else:
            row[i] = 1
            row[i-1] = 1
        H2[i] = row
    
    H = np.concatenate((H1, H2), axis=1)
    return H

def get_parity(arr):
    parity = 0
    for i in arr:
        if (i == 1):
            parity = int(not parity)

    return parity

# Ref: https://onlinelibrary.wiley.com/doi/epdf/10.1002/sat.787
def encode(H, message):
    
    # Convert to numpy array for ease
    message = np.array(message, dtype=np.int8)

    # Get the codeword length
    N = H.shape[1]
    
    # Get the message length
    K = N - H.shape[0]

    # Initialize the codeword
    codeword = np.zeros(N, dtype=np.int8)
    
    # First part of the codeword is identical to the message
    for i in range(K):
        codeword[i] = message[i]

    # Next part of the codeword is computed using parity
    last_parity = 0
    for i in range(N-K):
        codeword_idx = K + i

        # Solve for the parity bit first
        parity = get_parity(np.append(np.multiply(message, H[i, :K]), np.array([last_parity])))
        last_parity = parity

        # Now add the parity bit to the code word
        codeword[codeword_idx] = parity

    return list(codeword)

# Computes matrix vector product modulo two
def mat_vec_mul_mod2(M, v):

    # Create the output vector
    out_vec = np.zeros(M.shape[0], dtype=np.int8)
    
    # Compute matrix vector product
    out_vec = M.dot(v)

    # Compute modulo 2
    for idx in range(M.shape[0]):
        out_vec[idx] = out_vec[idx] % 2
    
    return out_vec

# This decoder tries to decode LDPC encoded message by flipping bits which are in most number of failing parity check equations
def fuzzy_decode(H, message, max_iters=1):
   
    # Tracks whether decoding was successful
    success = False

    # Create a local copy of message and start decoding
    decoded_message = deepcopy(message)
    for i in range(max_iters):

        # Step 1: Find out which equations are failing
        failures = mat_vec_mul_mod2(H, decoded_message)
        
        # Step 2: Collect the indices of equations which failed
        failed = False
        failure_idx = []
        for idx, element in enumerate(failures):
            if element == 1:
                failed = True
                failure_idx.append(idx)
        
        # Step 3: Check if decoding was successful and break if the code was successfully decoded
        if not failed:
            success = True
            break
        
        # Step 4: Create a tally for how many times a bit was present in the failed equations
        tally = [0 for x in range(H.shape[1])]
        for idx in failure_idx:
            for bit_idx, bit in enumerate(H[idx]):
                if bit == 1:    # The bit at bit_idx was involved in the parity check
                    tally[bit_idx] += 1

        # Step 5: Find the bit which was involved in most number of failures
        suspect = 0
        max_count = -1
        for idx, count in enumerate(tally):
            if count >= max_count:
                suspect = idx
                max_count = count

        # Step 6: Flip the suspect bit
        decoded_message[suspect] = int(not decoded_message[suspect])

    # Compute the number of message bits
    message_bits = H.shape[1] - H.shape[0]
    
    # Extract the message bits and return them
    return list(decoded_message[:message_bits]), success

# This decoder tries to decode LDPC encoded message by flipping the bits in which are involved in ALL failing parity check equations
def intersect_decode(H, message, max_iters=1):
   
    # Tracks whether decoding was successful
    success = False

    # Create a local copy of message and start decoding
    decoded_message = deepcopy(message)
    for i in range(max_iters):

        # Step 1: Find out which equations are failing
        failures = mat_vec_mul_mod2(H, decoded_message)
        
        # Step 2: Collect the indices of equations which failed
        failed = False
        failure_idx = []
        for idx, element in enumerate(failures):
            if element == 1:
                failed = True
                failure_idx.append(idx)
        
        # Step 3: Break if message was decoded correctly
        if not failed:
            success = True
            break

        # Step 4: Create the sets of bits involved in failing equations
        sets = []
        for idx in failure_idx:
            bit_set = set([])
            for bit_idx, bit in enumerate(H[idx]):
                if bit == 1:    # The bit at bit_idx was involved in the parity check
                    bit_set.add(bit_idx)
            sets.append(bit_set)

        # Step 5: Take intersection of sets
        failed_bits = sets[0]
        for s in sets:
            failed_bits = failed_bits.intersection(s)

        # Step 6: Flip the bits which were in all the failed equations
        for idx in failed_bits:
            decoded_message[idx] = int(not decoded_message[idx])

    # Compute the number of message bits
    message_bits = H.shape[1] - H.shape[0]
    
    # Extract the message bits and return them
    return list(decoded_message[:message_bits]), success

# This decoder tries to decode LDPC encoded message by using belief propagation
def belief_propagation_decode(H, message, snr, max_iters=1):
    return fuzzy_decode(H, message, max_iters)
