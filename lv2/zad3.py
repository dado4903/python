import numpy as np
import matplotlib.pyplot as plt
img = plt.imread("tiger.png")
brightness=img*0.5
img = img[:,:,0].copy()
print(img.shape)
(h,w)=img.shape

img_rotate=np.zeros((w,h))
for i in range(0,h):
    img_rotate[:,h-1-i]=img[i,:]

img_flip=np.zeros((h,w))
for i in range(0,w):
    img_flip[:,w-1-i]=img[:,i]

img_small=img[::10,::10]

dg = w//4
gg = w//2
img_cut=img[:,:].copy()
for i in range(h):
    for j in range(w):
        if (j < dg or j > gg):
            img_cut[i][j] = 0.

plt.imshow(img_cut,cmap="gray")
plt.show()
