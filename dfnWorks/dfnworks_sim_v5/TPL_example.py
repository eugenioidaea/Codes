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
               ncpu=8)

DFN.params['domainSize']['value'] = [25, 25, 50] # Minimum representative volume of the shale caprock
DFN.params['h']['value'] = 0.1 # fine grid resolution to accurately model small-scale fractures [m]
DFN.params['domainSizeIncrease']['value'] = [3.0, 3.0, 3.0] # Slight domain increase to accommodate boundary effects
DFN.params['keepOnlyLargestCluster']['value'] = True # Set to True to focus on the main connected fracture network
DFN.params['ignoreBoundaryFaces']['value'] = False # Set to False to consider interactions at the domain boundaries
DFN.params['boundaryFaces']['value'] = [0, 0, 0, 0, 1, 1]
DFN.params['seed']['value'] = 42 # Arbitrary choice for reproducibility

DFN.add_fracture_family(shape="rect",
                        distribution="log_normal",
                        p32=2.0,
                        kappa=25.78,
                        probability=.5,
                        aspect=1.0,
                        beta_distribution=1,
                        beta=0.0,
                        theta=95.47,
                        phi=23.32,
                        log_mean=1.38,
                        log_std=.06,
                        min_radius=2.0,
                        max_radius=9.0,
                        hy_variable='permeability', # assign hydraulic aperture to fractures
                        hy_function='constant', # use a correlated model where aperture scales with fracture size
                        hy_params={"mu": 2e-11})

DFN.add_fracture_family(shape="rect",
                        distribution="log_normal",
                        p32=2.0,
                        kappa=32.0,
                        probability=.5,
                        aspect=1.0,
                        beta_distribution=1,
                        beta=0.0,
                        theta=1.42,
                        phi=26.81,
                        log_mean=2.08,
                        log_std=.06,
                        min_radius=2.0,
                        max_radius=9.0,
                        hy_variable='permeability', # assign hydraulic aperture to fractures
                        hy_function='constant', # use a correlated model where aperture scales with fracture size
                        hy_params={"mu": 2e-11})

# DFN.add_fracture_family(
#     shape="ell",
#     distribution="tpl",
#     alpha=2.5, # it should reflect the observed fracture size distribution in shale
#     min_radius=1.0,
#     max_radius=3.0,
#     kappa=10.0, # it determines the clustering of fracture orientations: higher value indicates tighter clustering around the mean orientation
#     theta=0.0, # angle with the z-axis in degrees (set based on regional stress orientations)
#     phi=0.0, # azimuthal angle in degrees (set based on regional stress orientations)
#     aspect=1, # defines the elongation of fractures: elliptical fractures with 1.5:1 aspect ratio
#     # beta_distribution=1, # specify if fractures have a preferred in-plane rotation: constant rotation
#     # beta=45.0, # specify if fractures have a preferred in-plane rotation: rotation angle in degrees
#     p32=1.5, # fracture intensity in m²/m³
#     hy_variable='permeability', # assign hydraulic aperture to fractures
#     hy_function='constant', # use a correlated model where aperture scales with fracture size
#     hy_params={"mu": 2e-11})
# 
# DFN.add_fracture_family(
#     shape="ell",
#     distribution="tpl",
#     alpha=2.5, # it should reflect the observed fracture size distribution in shale
#     min_radius=1.0,
#     max_radius=3.0,
#     kappa=10.0, # it determines the clustering of fracture orientations: higher value indicates tighter clustering around the mean orientation
#     theta=0.0, # angle with the z-axis in degrees (set based on regional stress orientations)
#     phi=270.0, # azimuthal angle in degrees (set based on regional stress orientations)
#     aspect=1, # defines the elongation of fractures: elliptical fractures with 1.5:1 aspect ratio
#     # beta_distribution=1, # specify if fractures have a preferred in-plane rotation: constant rotation
#     # beta=45.0, # specify if fractures have a preferred in-plane rotation: rotation angle in degrees
#     p32=1.5, # fracture intensity in m²/m³
#     hy_variable='permeability', # assign hydraulic aperture to fractures
#     hy_function='constant', # use a correlated model where aperture scales with fracture size
#     hy_params={"mu": 2e-11})
# 
# DFN.add_fracture_family(
#     shape="ell",
#     distribution="tpl",
#     alpha=2.5, # it should reflect the observed fracture size distribution in shale
#     min_radius=1.0,
#     max_radius=3.0,
#     kappa=10.0, # it determines the clustering of fracture orientations: higher value indicates tighter clustering around the mean orientation
#     theta=45.0, # angle with the z-axis in degrees (set based on regional stress orientations)
#     phi=0.0, # azimuthal angle in degrees (set based on regional stress orientations)
#     aspect=1, # defines the elongation of fractures: elliptical fractures with 1.5:1 aspect ratio
#     # beta_distribution=1, # specify if fractures have a preferred in-plane rotation: constant rotation
#     # beta=45.0, # specify if fractures have a preferred in-plane rotation: rotation angle in degrees
#     p32=1.5, # fracture intensity in m²/m³
#     hy_variable='permeability', # assign hydraulic aperture to fractures
#     hy_function='constant', # use a correlated model where aperture scales with fracture size
#     hy_params={"mu": 2e-11})

DFN.make_working_directory(delete=True)
DFN.check_input()
DFN.create_network()
DFN.mesh_network(uniform_mesh = True)
DFN.dfn_flow()