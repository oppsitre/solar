import os
import cv2
import numpy as np
path = '/home/lcc/code/data/SSRL_SKY_CAM_IMAGE/'
X = []
for root, dirs, files in os.walk(path):
    for name in files:
        f = os.path.join(root, name)
        print f
        img = cv2.imread(f, 0)
        if img == None:
            continue
        print img.shape
        X.append(img.ravel())

mean = np.mean(X, axis=0)
X -= mean
mean = np.reshape(mean, [100, 90])
std = np.reshape(np.std(X, axis=0), [100, 90])
print mean
print std
np.save('mean.npy', mean)
np.save('std.npy', std)
