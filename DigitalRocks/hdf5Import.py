from hdf5storage import loadmat
import matplotlib.pyplot as plt

# 'H' is the dictionary key for "hangwall.mat"
# For footwall*, the dictionary key appears to be 'F'
# img = loadmat(r'/home/eugenio/ownCloud/IDAEA/Data/15 fractures of granite/srv/www/digrocks/portal/media/projects/472/origin/2594/images/hangwall1_AG1_240.mat')['H']
img = loadmat('/home/eugenio/ownCloud/IDAEA/Data/3D Collection of Binary Images/srv/www/digrocks/portal/media/projects/374/origin/1736/images/374_01_00_256.mat')['bin']

plt.figure()
plt.imshow(img)