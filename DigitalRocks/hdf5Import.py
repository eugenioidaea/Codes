import matplotlib.pyplot as plt
import tifffile as tiff
import porespy as ps
import numpy as np
from skimage.transform import resize
import openpnm as op
import matplotlib.transforms as transforms

ts = 0.95 # Threshold value: above is fracture, below is matrix
xl = 7000 # 1000
xr = 9000# 1100 # 8000
yb = 5500
yt = 6500 # 7200

imageTif = tiff.imread("/home/eugenio/Downloads/G4.tif") # For 2D images it is a simple matrix with values between 0 (white->solid matrix) and 255 (black->fracture and pores)
imageTif = np.flipud(imageTif)
imageComplete = resize(imageTif, (imageTif.shape[0], imageTif.shape[0]))
imageCrop = imageTif[yb:yt, xl:xr]
imageCrop = resize(imageCrop, (imageCrop.shape[0], imageCrop.shape[1])) # It scales the values between 0 (white) and 1 (black)
binIm = np.zeros((yt-yb, xr-xl)) # It crops the imageCrop
binIm[imageCrop>ts] = 0 # Fracture
binIm[imageCrop<ts] = 1 # Matrix

imageFull = plt.figure(figsize=(8, 8))
plt.imshow(imageComplete, cmap='gray', origin='lower')
plt.plot([xl, xr, xr, xl, xl], [yb, yb, yt, yt, yb])
grayScale = plt.figure(figsize=(8, 8))
plt.imshow(imageCrop, cmap='gray', origin='lower')
binary = plt.figure(figsize=(8, 8))
plt.imshow(binIm, cmap='gray', origin='lower')

net = ps.networks.snow2(binIm) #, voxel_size=1)
pn = op.io.network_from_porespy(net.network)

# Transpose the pore coordinates (swap X and Y)
pn["pore.coords"][:, [1, 0]] = pn["pore.coords"][:, [0, 1]]
poreNetwork = plt.figure(figsize=(8, 8))
plt.imshow(np.flipud(binIm), cmap=plt.cm.bone)
op.visualization.plot_coordinates(ax=poreNetwork,
                                  network=pn,
                                  size_by=pn["pore.inscribed_diameter"],
                                  color_by=pn["pore.inscribed_diameter"],
                                  markersize=100)
op.visualization.plot_connections(ax=poreNetwork, 
                                  network=pn,
                                  size_by=pn['throat.inscribed_diameter'],
                                  linewidth=10)
# poreNetwork.axis("off")