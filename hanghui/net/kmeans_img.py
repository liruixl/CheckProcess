from PCV.tools import imtools
import numpy as np
import pickle
from scipy import *
from pylab import *
from PIL import Image
from scipy.cluster.vq import *
from PCV.tools import pca

img_dir = r'E:\DataSet\fiber\cluster_roi'
N = 15

# Uses sparse pca codepath.
imlist = imtools.get_imlist(img_dir)

# 获取图像列表和他们的尺寸
im = array(Image.open(imlist[0]))  # open one image to get the size
m, n = im.shape[:2]  # get the size of the images
imnbr = len(imlist)  # get the number of images
print("The number of images is %d" % imnbr)

# Create matrix to store all flattened images

# ValueError: setting an array element with a sequence.
# immatrix = array([array(Image.open(imname)).flatten() for imname in imlist], 'f')



immatrix = np.array([array(Image.open(imname).resize((50, 50)).convert('L')).flatten() for imname in imlist])



# PCA降维
V, S, immean = pca.pca(immatrix)

# 保存均值和主成分
#f = open('./a_pca_modes.pkl', 'wb')
f = open('./a_pca_modes.pkl', 'wb')
pickle.dump(immean,f)
pickle.dump(V,f)
f.close()


# get list of images
imlist = imtools.get_imlist(img_dir)
imnbr = len(imlist)

# load model file
with open('./a_pca_modes.pkl','rb') as f:
    immean = pickle.load(f)
    V = pickle.load(f)
# create matrix to store all flattened images
# immatrix = np.array([array(Image.open(im).resize((50, 50))).flatten() for im in imlist])

# project on the 40 first PCs
immean = immean.flatten()
projected = array([dot(V[:40],immatrix[i]-immean) for i in range(imnbr)])

# k-means
projected = whiten(projected)
centroids,distortion = kmeans(projected, N, iter=500)
code,distance = vq(projected,centroids)

# plot clusters
for k in range(N):
    ind = where(code==k)[0]
    figure()
    gray()
    for i in range(minimum(len(ind),40)):
        subplot(8,5,i+1)
        imshow(immatrix[ind[i]].reshape((50,50)))
        axis('off')
show()