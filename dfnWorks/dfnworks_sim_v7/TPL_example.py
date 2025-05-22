#"""
#   :synopsis: Driver run file for TPL example
#   :version: 2.0
#   :maintainer: Jeffrey Hyman
#.. moduleauthor:: Jeffrey Hyman <jhyman@lanl.gov>
#"""

from pydfnworks import *
import os

jobname = os.getcwd() + "/output"
dfnFlow_file = os.getcwd() + '/dfn_diffusion_no_flow.in'

DFN = DFNWORKS(jobname,
               dfnFlow_file=dfnFlow_file,
               ncpu=8)

DFN.params['domainSize']['value'] = [10, 10, 20] # Minimum representative volume of the shale caprock
DFN.params['h']['value'] = 0.1 # fine grid resolution to accurately model small-scale fractures [m]
DFN.params['boundaryFaces']['value'] = [0, 0, 0, 0, 1, 1]

DFN.add_user_fract(shape='rect',
                   radii=11,
                   translation=[0, 0, 0],
                   normal_vector=[1, 0, 0],
                   permeability=1.0e-10)

DFN.add_user_fract(shape='rect',
                   radii=0.1,
                   translation=[0, 0, 0],
                   normal_vector=[0, 1, 0],
                   permeability=1.0e-10)

DFN.make_working_directory(delete=True)
DFN.check_input()
DFN.create_network()
DFN.num_frac = 1 # this hacks the meshing and only meshes the first fracture

# DFN.aperture

DFN.mesh_network(uniform_mesh = True, strict = False)
DFN.dfn_flow()