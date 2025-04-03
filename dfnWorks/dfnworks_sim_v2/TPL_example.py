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

DFN.params['domainSize']['value'] = [5, 5, 5]
DFN.params['h']['value'] = 0.1
DFN.params['domainSizeIncrease']['value'] = [0.2, 0.2, 0.2]
DFN.params['keepOnlyLargestCluster']['value'] = True
DFN.params['ignoreBoundaryFaces']['value'] = False
DFN.params['boundaryFaces']['value'] = [0, 0, 0, 0, 1, 1]
DFN.params['seed']['value'] = 10

DFN.add_fracture_family(shape="ell",
    distribution="tpl",
    alpha=1.8,
    min_radius=1.0,
    max_radius=5.0,
    kappa=1.0,
    theta=0.0,
    phi=0.0,
    aspect=2,
    beta_distribution=1,
    beta=45.0,
    p32=1.5,
    hy_variable='aperture',
    hy_function='correlated',
    hy_params={
        "alpha": 10**-5,
        "beta": 0.5
    })

DFN.make_working_directory(delete=True)
DFN.check_input()
DFN.create_network()
DFN.mesh_network(uniform_mesh = True)
DFN.dfn_flow() 
