import numpy as np
from random import uniform

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

    # Next part of the codework is computed using parity
    last_parity = 0
    for i in range(N-K):
        codeword_idx = K + i

        # Solve for the parity bit first
        parity = get_parity(np.append(np.multiply(message, H[i, :K]), np.array([last_parity])))
        last_parity = parity

        # Now add the parity bit to the code word
        codeword[codeword_idx] = parity

    return codeword

def belief_propogation_decode(H, message):
    pass

def fuzzy_decode(H, message, max_iters = 1):
    pass
