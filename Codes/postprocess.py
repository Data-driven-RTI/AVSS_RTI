#encoding=utf-8

from skimage.feature import canny
from skimage import measure
import numpy as np

def processData(data):
    data = data.reshape(-1)
    data_ind = np.argsort(data)
    data_ind = data_ind[::-1]
    data[data_ind[4000:]] = 0.0
    data = data.reshape(360,360)
    edges = canny(data)
    labels = measure.label(edges,connectivity=2)
    props = measure.regionprops(labels)
    area = [p.area for p in props]
    if len(area) == 0:
        return np.zeros_like(data,dtype=np.float32)
    max_idx = np.argmax([p.area for p in props])
    max_contour_image = np.zeros_like(data,dtype=np.float32)
    max_contour_image[labels==max_idx + 1] = 1.0
    for j in range(360):
        indices = np.argwhere(max_contour_image[:,j]==1.0)
        if indices.shape[0] > 1:
            max_contour_image[indices[0,0]:indices[-1,0],j] = 1.0
    return max_contour_image