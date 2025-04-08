#"""
#   :synopsis: Driver run file for TPL example
#   :version: 2.0
#   :maintainer: Jeffrey Hyman
#.. moduleauthor:: Jeffrey Hyman <jhyman@lanl.gov>
#"""

from pydfnworks import *
import os
import numpy as np

jobname = os.getcwd() + "/output"
dfnFlow_file = os.getcwd() + '/dfn_diffusion_no_flow.in'

DFN = DFNWORKS(jobname,
               dfnFlow_file=dfnFlow_file,
               ncpu=4)

DFN.params['domainSize']['value'] = [5, 5, 5] # Minimum representative volume of the shale caprock
DFN.params['h']['value'] = 0.05 # fine grid resolution to accurately model small-scale fractures [m]
DFN.params['domainSizeIncrease']['value'] = [0.1, 0.1, 0.1] # Slight domain increase to accommodate boundary effects
DFN.params['keepOnlyLargestCluster']['value'] = True # Set to True to focus on the main connected fracture network
DFN.params['ignoreBoundaryFaces']['value'] = False # Set to False to consider interactions at the domain boundaries
DFN.params['boundaryFaces']['value'] = [0, 0, 0, 0, 1, 1]
DFN.params['seed']['value'] = 42 # Arbitrary choice for reproducibility

DFN.add_fracture_family(shape="ell",
    distribution="tpl",
    alpha=2.5, # it should reflect the observed fracture size distribution in shale
    min_radius=1.0,
    max_radius=5.0,
    kappa=10.0, # it determines the clustering of fracture orientations: higher value indicates tighter clustering around the mean orientation
    theta=30.0, # angle with the z-axis in degrees (set based on regional stress orientations)
    phi=60.0, # azimuthal angle in degrees (set based on regional stress orientations)
    aspect=1.5, # defines the elongation of fractures: elliptical fractures with 1.5:1 aspect ratio
    beta_distribution=1, # specify if fractures have a preferred in-plane rotation: constant rotation
    beta=45.0, # specify if fractures have a preferred in-plane rotation: rotation angle in degrees
    p32=2.0, # fracture intensity in m²/m³
    hy_variable='aperture', # assign hydraulic aperture to fractures
    hy_function='correlated', # use a correlated model where aperture scales with fracture size
    hy_params={
        "alpha": 10**-5, # aperture scaling factor
        "beta": 0.5 # exponent for scaling
    })

DFN.make_working_directory(delete=True)
DFN.check_input()
DFN.create_network()
DFN.mesh_network(uniform_mesh = True)
DFN.dfn_flow() 
