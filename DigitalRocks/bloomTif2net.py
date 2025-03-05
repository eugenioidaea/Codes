import matplotlib.pyplot as plt
import tifffile as tiff
import porespy as ps
import numpy as np
from skimage.transform import resize
import openpnm as op
import matplotlib.transforms as transforms

ts = 0.5 # Threshold value: above is fracture, below is matrix
xl = 500 # 1000
xr = 2500 # 1100 # 8000
yb = 1000
yt = 2000 # 7200

# imageTif = tiff.imread("/home/eugenio/ownCloud/IDAEA/Data/Vega2022/srv/www/digrocks/portal/media/projects/415/origin/2333/images/G4.tif") # For 2D images it is a simple matrix with values between 0 (white->solid matrix) and 255 (black->fracture and pores)
imageTif = tiff.imread("/home/eugenio/ownCloud/IDAEA/Data/Vega2022/srv/www/digrocks/portal/media/projects/415/origin/2335/images/AnyConv.com__bloom_60_1.tif") # For 2D images it is a simple matrix with values between 0 (white->solid matrix) and 255 (black->fracture and pores)
imageTif = np.flipud(imageTif)
imageComplete = resize(imageTif, (imageTif.shape[0], imageTif.shape[1]))
imageCrop = imageTif[yb:yt, xl:xr]
imageCrop = resize(imageCrop, (imageCrop.shape[0], imageCrop.shape[1])) # It scales the values between 0 (white) and 1 (black)
binIm = np.zeros((yt-yb, xr-xl)) # It crops the imageCrop
binIm[imageCrop<ts] = 0 # Black
binIm[imageCrop>ts] = 1 # White

imageFull = plt.figure(figsize=(8, 8))
plt.imshow(imageComplete, cmap='gray', origin='lower')
plt.plot([xl, xr, xr, xl, xl], [yb, yb, yt, yt, yb], linewidth=3)
grayScale = plt.figure(figsize=(8, 8))
plt.imshow(imageCrop, cmap='gray', origin='lower')
binary = plt.figure(figsize=(8, 8))
plt.imshow(binIm, cmap='gray', origin='lower')

net = ps.networks.snow2(binIm) #, voxel_size=10)
pn = op.io.network_from_porespy(net.network)

print(pn)

# Transpose the pore coordinates (swap X and Y)
pn["pore.coords"][:, [1, 0]] = pn["pore.coords"][:, [0, 1]]
poreNetwork = plt.figure(figsize=(8, 8))
plt.imshow(binIm, cmap=plt.cm.bone, origin='lower')
op.visualization.plot_coordinates(ax=poreNetwork,
                                  network=pn,
                                  size_by=pn["pore.inscribed_diameter"],
                                  color_by=pn["pore.inscribed_diameter"],
                                  markersize=10)
op.visualization.plot_connections(ax=poreNetwork, 
                                  network=pn,
                                  size_by=pn['throat.inscribed_diameter'],
                                  linewidth=10)
# _ = plt.axis("off")

pn['pore.bottom']=pn['pore.coords'][:, 1]<10
pn['pore.top']=pn['pore.coords'][:, 1]>990

liquid = op.phase.Phase(network=pn) # Phase dictionary initialisation

liquid['throat.diffusive_conductance'] = np.ones(net.Nt)*1e-3



tfd = op.algorithms.TransientFickianDiffusion(network=net, phase=liquid) # TransientFickianDiffusion dictionary initialisation
tfd.run(x0=ic, tspan=simTime)