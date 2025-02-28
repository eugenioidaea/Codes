import matplotlib.pyplot as plt
import tifffile as tiff
import porespy as ps
import numpy as np
from skimage.transform import resize
import openpnm as op
import matplotlib.transforms as transforms

ts = 0.70 # Threshold value: above is fracture (0), below is matrix (1)
xl = 7000
xr = 7100 # 10000
yb = 1000
yt = 1050 # 8000

imageTif = tiff.imread("/home/eugenio/Downloads/G4.tif") # For 2D images it is a simple matrix with values between 0 (white->solid matrix) and 255 (black->fracture and pores)
imageTif = np.flipud(imageTif)
imageComplete = resize(imageTif, (imageTif.shape[0], imageTif.shape[0]))
image = imageTif[yb:yt, xl:xr]
image = resize(image, (image.shape[0], image.shape[1])) # It scales the values between 0 (white) and 1 (black)
binIm = np.zeros((yt-yb, xr-xl)) # It crops the image
binIm[image>ts] = 0 # Binary transformation
binIm[image<ts] = 1 # Binary transformation

imageFull = plt.figure(figsize=(8, 8))
plt.imshow(imageComplete, cmap='gray', origin='lower')
plt.plot([xl, xr, xr, xl, xl], [yb, yb, yt, yt, yb])
grayScale = plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='gray')
binary = plt.figure(figsize=(8, 8))
plt.imshow(binIm, cmap='gray')

net = ps.networks.snow2(binIm) #, voxel_size=1)
pn = op.io.network_from_porespy(net.network)

# Transpose the pore coordinates (swap X and Y)
# pn["pore.coords"][:, [0, 1]] = pn["pore.coords"][:, [0, 1]]
poreNetwork = plt.figure(figsize=(8, 8))
plt.imshow(binIm, cmap=plt.cm.bone)
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