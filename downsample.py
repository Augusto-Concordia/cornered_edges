import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from utils import gaussian_kernel, filter2d


def main():

    # load the image
    im = imread('paint.jpg').astype('float')
    im = im / 255

    # number of levels for downsampling
    N_levels = 5

    # make a copy of the original image
    im_subsample = im.copy()

    # creates subplots
    fig, axes = plt.subplots(2, N_levels, figsize=(12, 6))

    # naive subsampling, visualize the results on the 1st row
    for i in range(N_levels):
        # subsample image
        im_subsample = im_subsample[::2, ::2, :]

        axes[0, i].imshow(im_subsample)
        axes[0, i].axis('off')
        axes[0, i].set_title('Level %d (%dx%d)' % (
            i+1, im_subsample.shape[0], im_subsample.shape[1]))

    # YOUR CODE HERE

    # create a Gaussian kernel
    kernel = gaussian_kernel()

    # make a copy of the original image
    im_subsample = im.copy()

    # subsampling without aliasing, visualize the results on the 2nd row
    for i in range(N_levels):
        # split the image into its RGB channels
        im_r = im_subsample[:, :, 0]
        im_g = im_subsample[:, :, 1]
        im_b = im_subsample[:, :, 2]

        # smooth image
        im_smooth = np.zeros(im_subsample.shape)
        im_smooth[:, :, 0] = filter2d(im_r, kernel)
        im_smooth[:, :, 1] = filter2d(im_g, kernel)
        im_smooth[:, :, 2] = filter2d(im_b, kernel)

        # subsample image
        im_subsample = im_smooth[::2, ::2, :]

        axes[1, i].imshow(im_subsample)
        axes[1, i].axis('off')

    plt.show()

    # END YOUR CODE


if __name__ == "__main__":
    main()
