import csv
import numpy as np 
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import maxflow



def load_data():
	xtr = []
	ytr = []
	
	with open('Xtr.csv','rb') as csvfile:
		reader = csv.reader(csvfile, delimiter = ',')
		for row in reader:
			row = np.array(row, dtype = np.float32).reshape(28,28)
			row = preprocessing(row)
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
			row= preprocessing(row)
			xte.append(list(row.astype(np.float32)))
	return xte

def preprocessing(img):
    img = normalize(img, 0.0, 1.0)
    #img = threshold(img, 0.5, 1.0)
    img = threshold_cut(img)
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
    img = resize(img, (int(height / ratio), int(width / ratio)))
    img = copyMakeBorder(img, 0, 28 - img.shape[0], 0, 28 - img.shape[1])
    ctr = center(img)
    offset = (14 - ctr[0], 14 - ctr[1])
    m = np.asarray([[1, 0, offset[0]], [0, 1, offset[1]]])
    img = translation(img, offset)
    img = blur(img, (3, 3))
    return img

def translation(img, offset):
    dx = offset[0]
    dy = offset[1]
    dx = int(dx)
    dy = int(dy)
    width = img.shape[1]
    height = img.shape[0]
    if dx >= 0 and dy >= 0:
        img = copyMakeBorder(img, dy, 0, dx, 0)
    elif dx >= 0 and dy <0:
        img = copyMakeBorder(img, 0, dy, dx, 0)
    elif dx < 0 and dy >= 0:
        img = copyMakeBorder(img, dy, 0, 0, dx)
    else:
        img = copyMakeBorder(img, 0, dy, 0, dx)
    img = img[:height, :width]
    return img

def center(img, method = 'mass'):
    if method == 'mass':
        m00 = np.sum(img)
        
        width = img.shape[1]
        height = img.shape[0]

        tmp = np.ones((height, 1)).dot(np.arange(width)[np.newaxis, :])
        m10 = np.sum(img * tmp)
        m01 = np.sum(img * tmp.T)
        return m10 / float(m00), m01 / float(m00)
    else:
        return img.shape[1] / 2.0, img.shape[0] / 2.0

def threshold(img, thresh, beta, alpha = 0.0):
    return alpha * (img < thresh) + beta * (img >= thresh)

def g(x):
    alpha = 0.01
    beta = 1000.0
    return beta / (1 + alpha * x * x)


def threshold_cut(img):
    #G = cv2.Laplacian(img, cv2.CV_64F)
    #return G
    #ci = 1.0
    #ce = 0.0
    #width = img.shape[1]
    #height = img.shape[0]
    #graph = maxflow.Graph[float](width * height, 2*(width - 1) * (height - 1))
    #nodes = graph.add_nodes(width * height)
    #for i in range(height):
    #    for j in range(width):
    #        gi = np.abs(img[i][j] - ci)
    #        ge = np.abs(img[i][j] - ce)
    #        graph.add_tedge(nodes[i*width + j], ge, gi);

    #for i in range(height - 1):
    #    for j in range(width - 1):
    #        for k in range(2):
    #            for l in range(2):
    #                if k != l:
    #                    lbd = g((G[i][j]+G[i+l][j+k])/2)
    #                    graph.add_edge(nodes[i*width+j], nodes[(i+l)*width+(j+k)], lbd, lbd)

    #flow = graph.maxflow()

    #tmp = graph.get_grid_segments(nodes)
    #
    #return tmp.reshape((28, 28))
    #for i in range(height):
    #    for j in range(width):
    #        img[i][j] = graph.get_segment(nodes[i * width + j])
    #return img

    # Create the graph.
    g = maxflow.Graph[float]()
    # Add the nodes. nodeids has the identifiers of the nodes in the grid.
    nodeids = g.add_grid_nodes(img.shape)
    # Add non-terminal edges with the same capacity.
    g.add_grid_edges(nodeids, 0.1)
    # Add the terminal edges. The image pixels are the capacities
    # of the edges from the source node. The inverted image pixels
    # are the capacities of the edges to the sink node.
    g.add_grid_tedges(nodeids, img, 1.0-img)
    # Find the maximum flow.
    g.maxflow()
    # Get the segments of the nodes in the grid.
    sgm = g.get_grid_segments(nodeids)
    return np.int_(np.logical_not(sgm))

def filter2D(img, kernel, anchor = (-1, -1)):
    top = kernel.shape[0] / 2
    left = kernel.shape[1] / 2

    img = copyMakeBorder(img, top, top, left, left)

    img_new = np.zeros(img.shape)

    if anchor == (-1, -1):
        anchor = (top, left)

    for i in range(top, img_new.shape[0] - top):
        for j in range(left, img_new.shape[1] - left):
            for k in range(kernel.shape[0]):
                for l in range(kernel.shape[1]):
                    img_new[i][j] += kernel[k][l] * img[i + k - anchor[0]][j + l - anchor[1]]
    
    img_new = img_new[top:img_new.shape[0] - top, left:img_new.shape[1] - left]

    return img_new

def gaussian_kernel(size, size_y=None):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
    g = np.exp(-(x**2/float(size)+y**2/float(size_y)))
    return g / g.sum()

def blur(img, ksize, method = 'gaussian'):
    if method == 'gaussian':
        #kernel = gaussian_kernel(ksize[0], ksize[1])
        kernel = np.asarray([[1/16.0, 1/8.0, 1/16.0], [1/8.0, 1/4.0, 1/8.0], [1/16.0, 1/8.0, 1/16.0]])
    else:
        kernel = np.ones(ksize) / float(ksize[0] * ksize[1])
    img = filter2D(img, kernel)
    return img

def copyMakeBorder(img, top, bottom, left, right):
    for i in range(top):
        img = np.insert(img, 0, 0, axis = 0)

    for i in range(bottom):
        img = np.insert(img, img.shape[0], 0, axis = 0)

    for i in range(left):
        img = np.insert(img, 0, 0, axis = 1)

    for i in range(right):
        img = np.insert(img, img.shape[1], 0, axis = 1)
    return img
    
def normalize(img, alpha, beta):
    assert beta > alpha
    inf = np.amin(img)
    sup = np.amax(img)
    INF = inf * np.ones(img.shape)
    SUP = sup * np.ones(img.shape)
    img = alpha * np.ones(img.shape) + (img - INF) / (SUP - INF) * (beta - alpha)
    return img

def resize(img, shape):
    """Resize an image to a desired shape

    Keyword arguments:
    img -- numpy 2d-array as input image
    shape -- tuple of form (height, width) 
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
    img = img.reshape(shape)

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
    ## test resize
    #img = cv2.imread('lena_std.tif', cv2.IMREAD_GRAYSCALE)
    #img = resize(img, (128, 128))
    #img = img.astype(np.uint8)
    #cv2.imshow('img', img)
    #cv2.waitKey()
    
    ### test normalize
    #a = np.asarray([[0, 255], [128, 64]])
    #a = normalize_(a, 0, 1)
    #print a

    #a = threshold(a, 0.5, 255.0)
    #print a

    ##a = copyMakeBorder(a, 0, 1, 2, 3)
    ##print a
    #kernel = np.ones((3, 3)) / 9.0
    #a = filter2D(a, kernel)
    #print a

    img = cv2.imread('lena_std.tif', cv2.IMREAD_GRAYSCALE)
    wtf = threshold_cut(img)
    #img = blur(img, (3, 3))
    #img = img.astype(np.uint8)
    #cv2.imshow('img', img)
    #cv2.waitKey()
    #img = translation(img, (5.5, 128))
    #cv2.imshow('img', img)
    #cv2.waitKey()

