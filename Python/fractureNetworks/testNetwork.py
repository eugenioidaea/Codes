import numpy as np
import openpnm as op

def create_openpnm_network(throat_list):
    # Flatten the throat list into a set of unique points
    all_points = np.array([point for throat in throat_list for point in throat])
    unique_points, unique_indices = np.unique(all_points, axis=0, return_inverse=True)
    
    # Map the original points to unique indices
    throat_conns = unique_indices.reshape(-1, 2)  # Reshape into (N,2) pairs
    
    # Create OpenPNM network dictionary
    network_dict = {
        "pore.coords": np.column_stack((unique_points, np.zeros(len(unique_points)))),  # Add z=0
        "throat.conns": throat_conns
    }
    
    return network_dict

# Example list of throats (each tuple contains start and end points)
throats = [
    ((0, 0), (1, 1)),
    ((1, 1), (2, 2)),
    ((2, 2), (3, 3)),
    ((1, 1), (1, 0)),
    ((2, 2), (2, 1))
]

# Convert to OpenPNM format
network = create_openpnm_network(throats)

# Convert to OpenPNM network
pn = op.network.Network(conns=network['throat.conns'], coords=network['pore.coords'])

# Print results
print("Pore Coordinates:\n", pn["pore.coords"])
print("Throat Connections:\n", pn["throat.conns"])