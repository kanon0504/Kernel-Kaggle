import csv
import numpy as np 
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import cv2


def load_data():
	xtr = []
	ytr = []
	
	with open('Xtr.csv','rb') as csvfile:
		reader = csv.reader(csvfile, delimiter = ',')
		for row in reader:
			row = np.array(row, dtype = np.float32).reshape(28,28)
                        row, H = normalize(row)
			xtr.append(list(row.astype(np.float32)))

	with open('Ytr.csv','rb') as csvfile:
		reader = csv.DictReader(csvfile, delimiter = ',')
		for row in reader:
			ytr.append(row['Prediction'])

	return xtr, ytr

def load_data_test():
	xte = []	
	with open('Xte.csv','rb') as csvfile:
		reader = csv.reader(csvfile, delimiter = ',')
		for row in reader:
			row = np.array(row, dtype = np.float32).reshape(28,28)
                        row, H = normalize(row)
			xte.append(list(row.astype(np.float32)))
	return xte

def normalize(img):
    img = cv2.normalize(img, 0.0, 1.0, norm_type = cv2.cv.CV_MINMAX)
    _, img = cv2.threshold(img, 0.5, 1.0, cv2.THRESH_BINARY)
    row_sum = np.sum(img, axis = 1)
    tmp = np.nonzero(row_sum)[0]
    min_y = tmp[0]
    max_y = tmp[-1]
    height = max_y - min_y + 1
    col_sum = np.sum(img, axis = 0)
    tmp = np.nonzero(col_sum)[0]
    min_x = tmp[0]
    max_x = tmp[-1]
    width = max_x - min_x + 1
    img = img[min_y:max_y+1, min_x:max_x+1]
    if height > width:
        ratio = height / 20.0
    else:
        ratio = width / 20.0
    img = cv2.resize(img, (int(width / ratio), int(height / ratio)))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.copyMakeBorder(img, 0, 28 - img.shape[0], 0, 28 - img.shape[1], borderType = cv2.BORDER_CONSTANT)
    M = cv2.moments(img)
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    m = np.asarray([[1, 0, 14 - cx], [0, 1, 14 - cy]])
    img = cv2.warpAffine(img, m, (28, 28))
    H = cv2.HuMoments(M)
    return img, H

def resize(img, shape = img.shape):
    """Resize an image to a desired shape

    Keyword arguments:
    img -- numpy 2d-array as input image
    shape -- tuple of form (height, width) (default img.shape)
    """

    fx = img.shape[1] / float(shape[1]) 
    fy = img.shape[0] / float(shape[0])

    # old mesh grid with new coordinates
    x = np.linspace(0, shape[1], img.shape[1])
    y = np.linspace(0, shape[0], img.shape[0])

    # interpolation
    fun = interp2d(x, y, img)

    # new mesh grid
    x_new = np.arange(shape[1])
    y_new = np.arange(shape[0])

    # resized image
    img = fun(x_new, y_new)

    return img

def flip(xtr,ytr):
	x = []
	y = []
	for i in range(len(xtr)):
		x.append(list(np.fliplr(np.array(xtr[i]))))
		y.append(ytr[i])

	return x, y

def atcw_rotate(xtr):
	x = []
	n = len(x[0])
	for i in xtr:
		x.append(np.array(i)[::-1])
	x = list(np.array(x).reshape(n,n))
	return x

def flatten(xtr):
	for i in range(len(xtr)):
		xtr[i] = np.array(xtr[i]).flatten()

	return xtr


def plot(xtr,ytr):
	plt.figure(figsize=(14, 10))
	n_rows, n_cols = 4, 8
	for k in range(n_rows * n_cols):
	    plt.subplot(n_rows, n_cols, k + 1)
	    plt.imshow(np.asarray(xtr[k]).reshape(28,28).T, cmap=plt.cm.gray, interpolation='none')
	    plt.xticks(())
	    plt.yticks(())    
	    plt.title(ytr[k], size=10)
	plt.show()

if __name__ == '__main__':
    img = cv2.imread('lena_std.tif', cv2.IMREAD_GRAYSCALE)
    img = resize(img, (756, 756))
    img = img.astype(np.uint8)
    cv2.imshow('img', img)
    cv2.waitKey()
