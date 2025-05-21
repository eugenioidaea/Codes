#"""
#   :synopsis: Driver run file for single fracture
#   :version: 2.0
#   :maintainer: Jeffrey Hyman
#.. moduleauthor:: Jeffrey Hyman <jhyman@lanl.gov>
#"""

from pydfnworks import *
import os

src_path = os.getcwd()
jobname = src_path + "/output"

DFN = DFNWORKS(jobname,
               ncpu=4)

DFN.params['domainSize']['value'] = [1.0, 1.0, 10.0]
DFN.params['h']['value'] = 0.1
DFN.params['boundaryFaces']['value'] = [0,0,0,0,1,1] # set boundary faces to -z / +z

# This is your single fracture to cross the domain
DFN.add_user_fract(shape='rect',
                   radii=6,
                   translation=[0, 0, 0],
                   normal_vector=[1, 0, 0],
                   permeability=1.0e-12)

## Add a second fracture to include a intersection, will be remved
DFN.add_user_fract(shape='rect',
                   radii=0.1,
                   translation=[0, 0, 0],
                   normal_vector=[0, 1, 0],
                   permeability=1.0e-12)

## Workflow 
DFN.make_working_directory(delete=True)
DFN.check_input()
DFN.create_network()
DFN.num_frac = 1 # this hacks the meshing and only meshes the first fracture 
DFN.mesh_network(uniform_mesh = True)
