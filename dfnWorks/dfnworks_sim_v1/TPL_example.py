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

DFN.params['domainSize']['value'] = [10, 10, 10]
DFN.params['h']['value'] = 0.1
DFN.params['domainSizeIncrease']['value'] = [2, 2, 2]
DFN.params['keepOnlyLargestCluster']['value'] = True
DFN.params['ignoreBoundaryFaces']['value'] = False
DFN.params['boundaryFaces']['value'] = [0, 0, 0, 0, 1, 1]
DFN.params['seed']['value'] = 10

DFN.add_fracture_family(
    shape="ell",
    distribution="tpl",
    alpha=1.8,
    min_radius=1.0,
    max_radius=10.0,
    kappa=1.0,
    theta=0.0,
    phi=0.0,
    #aspect=2,
    p32=1,
    hy_variable='aperture',
    hy_function='correlated',
    number_of_points=8,
    hy_params={
        "alpha": 10**-4,
        "beta": 0.5
    })

DFN.make_working_directory(delete=True)
DFN.check_input()
DFN.create_network()
DFN.mesh_network(uniform_mesh = True)
DFN.dfn_flow() 
