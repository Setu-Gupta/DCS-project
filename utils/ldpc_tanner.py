import numpy as np

# Creates a tanner graph
def create_tanner_graph(H):
    
    # Get the number of check nodes and variable nodes
    num_checks = H.shape[0]
    num_variables = H.shape[1]  # Note that the variables include the parity bits as well
    
    # Create the check nodes and the variable nodes
    check_nodes = [i for i in range(num_checks)]
    variable_nodes = [i for i in range(num_variables)]

    # Create lists to store the neighbourhoods
    check_neighbourhood = {}
    variable_neighbourhood = {}

    # Compute the neighbourhoods of check nodes
    for row_idx in range(H.shape[0]):
        check_neighbourhood[row_idx] = []
        for col_idx in range(H.shape[1]):
            if H[row_idx, col_idx] == 1:
                check_neighbourhood[row_idx].append(col_idx)
    
    # Compute the neighbourhoods of variable nodes
    for col_idx in range(H.shape[1]):
        variable_neighbourhood[col_idx] = []
        for row_idx in range(H.shape[0]):
            if H[row_idx, col_idx] == 1:
                variable_neighbourhood[col_idx].append(row_idx)

    # The graph can be fully characterized by the nodes and the neighbourhoods
    return check_nodes, variable_nodes, check_neighbourhood, variable_neighbourhood
