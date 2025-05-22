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

DFN.params['domainSize']['value'] = [10, 10, 20] # Minimum representative volume of the shale caprock
DFN.params['h']['value'] = 0.1 # fine grid resolution to accurately model small-scale fractures [m]
DFN.params['domainSizeIncrease']['value'] = [3.0, 3.0, 3.0] # Slight domain increase to accommodate boundary effects
DFN.params['keepOnlyLargestCluster']['value'] = True # Set to True to focus on the main connected fracture network
DFN.params['ignoreBoundaryFaces']['value'] = False # Set to False to consider interactions at the domain boundaries
DFN.params['boundaryFaces']['value'] = [0, 0, 0, 0, 1, 1]
DFN.params['seed']['value'] = 42 # Arbitrary choice for reproducibility
DFN.params['angleOption']['value'] = 'degree'
DFN.params['orientationOption']['value'] = 2

DFN.add_fracture_family(shape="ell",
                        distribution="log_normal",
                        log_mean=0.5,
                        log_std=1,
                        min_radius=1,
                        max_radius=4,
                        p32=2.0,
                        kappa=60,
                        aspect=2,
                        strike=0,
                        dip=90,
                        hy_variable='permeability',
                        hy_function='log-normal',
                        hy_params={
                            "mu": 1e-10,
                            "sigma": 0.01
                        })

DFN.add_fracture_family(shape="ell",
                        distribution="log_normal",
                        log_mean=0.5,
                        log_std=1,
                        min_radius=1,
                        max_radius=4,
                        p32=1.0,
                        kappa=60,
                        aspect=0.5,
                        strike=40,
                        dip=10,
                        hy_variable='permeability',
                        hy_function='log-normal',
                        hy_params={
                            "mu": 1e-10,
                            "sigma": 0.01
                        })

DFN.make_working_directory(delete=True)
DFN.check_input()
DFN.create_network()
DFN.mesh_network(uniform_mesh = True)
DFN.dfn_flow()