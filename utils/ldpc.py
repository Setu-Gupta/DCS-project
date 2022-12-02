import numpy as np
from random import uniform
from copy import deepcopy
from ldpc_tanner import create_tanner_graph
from math import log, atanh, tanh

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
    previous_message = deepcopy(decoded_message)
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

        # Step 7: Exit if the decoded_message is the same as the previous decoded_message
        same = True
        for b1, b2 in zip(previous_message, decoded_message):
            if b1 != b2:
                same = False
                break

        # Step 8: If the message did not change, exit out of the loop
        if same:
            break
        
        # Update the previous message value
        previous_message = deepcopy(decoded_message)

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
    previous_message = deepcopy(decoded_message)
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
        
        # Step 7: Exit if the decoded_message is the same as the previous decoded_message
        same = True
        for b1, b2 in zip(previous_message, decoded_message):
            if b1 != b2:
                same = False
                break

        # Step 8: If the message did not change, exit out of the loop
        if same:
            break
        
        # Update the previous message value
        previous_message = deepcopy(decoded_message)

    # Compute the number of message bits
    message_bits = H.shape[1] - H.shape[0]
    
    # Extract the message bits and return them
    return list(decoded_message[:message_bits]), success

# Computes the log of the ratio of probability of a bit being 0 and the probability of the bit 1 given that the received bit was br
def get_log_likelihood(prob_bit_flip, br):
    prob_of_zero = 1
    prob_of_one = 1
    if br == 0:
        prob_of_zero = 1 -  prob_bit_flip
        prob_of_one = prob_bit_flip
    else:
        prob_of_zero = prob_bit_flip
        prob_of_one = 1 - prob_bit_flip

    return log(prob_of_zero/prob_of_one)


# This decoder tries to decode LDPC encoded message by using belief propagation
# Ref: https://www.youtube.com/watch?v=p7x-EOZF5zk
# Ref: https://www.youtube.com/watch?v=zAQB-jhYWOc
# Ref: https://yair-mz.medium.com/decoding-ldpc-codes-with-belief-propagation-43c859f4276d
# Ref: https://mathworld.wolfram.com/InverseHyperbolicTangent.html
def belief_propagation_decode(H, message, prob_bit_flip, max_iters=1):

    # Tracks whether decoding was successful
    success = False
    
    # Get the tanner graph for the parity matrix
    check_nodes, variable_nodes, check_neighbourhood, variable_neighbourhood = create_tanner_graph(H)

    # Step 1: Get the number of variable nodes and check nodes
    num_checks = len(check_nodes)
    num_vars = len(variable_nodes)

    # Step 2: Create a matrix to store all the variable -> check messages
    var_to_check_messages = np.zeros((num_vars, num_checks))

    # Step 3: Create a matrix to store all the variable <- check messages
    check_to_var_messages = np.zeros((num_checks, num_vars))

    # Step 4: Initialize the variable -> check messages
    for var_node in variable_nodes:
        for check_node in variable_neighbourhood[var_node]:
            # Initialize the messages to log likelihoods
            var_to_check_messages[var_node, check_node] = get_log_likelihood(prob_bit_flip, message[var_node])  
    
    # Create a local copy of message and start decoding
    decoded_message = deepcopy(message)
    for _ in range(max_iters):
        
        # Step 5: Compute variable <- check messages
        for check_node in check_nodes:
            for var_node in check_neighbourhood[check_node]:

                # Compute the big multiplication term
                x = 1
                for _var_node in check_neighbourhood[check_node]:
                    if _var_node != var_node:
                        x *= tanh(var_to_check_messages[_var_node, check_node]/2)
                
                        # Make sure that x is in (-1, 1)
                        if x <= -1:
                            x = -0.9999
                        if x => 1:
                            x = 0.9999 
		
                # Compute the variable <- check message
                check_to_var_messages[check_node, var_node] = 2 * atanh(x)

        # Step 6: Compute the variable -> check messages
        for var_node in variable_nodes:
            for check_node in variable_neighbourhood[var_node]:
                
                # Compute the big summation term
                x = 0
                for _check_node in variable_neighbourhood[var_node]:
                    if _check_node != check_node:
                        x += check_to_var_messages[_check_node, var_node]

                # Compute the variable -> check message
                var_to_check_messages[var_node, check_node] = get_log_likelihood(prob_bit_flip, decoded_message[var_node]) + x

        # Step 7: Make a decision
        for var_node in variable_nodes:
            
            # Compute the total log likelihood
            L = get_log_likelihood(prob_bit_flip, decoded_message[var_node])
            for check_node in variable_neighbourhood[var_node]:
                L += check_to_var_messages[check_node, var_node]

            # Set the bit based on total log likelihood
            decoded_message[var_node] = 1 if L < 0 else 0

        # Step 8: Check of the message has been decoded
        failures = mat_vec_mul_mod2(H, decoded_message)
        failed = False
        for element in failures:
            if element == 1:
                failed = True

        if not failed:
            success = True
            break
    
    # Compute the number of message bits
    message_bits = H.shape[1] - H.shape[0]
    
    # Extract the message bits and return them
    return list(decoded_message[:message_bits]), success
