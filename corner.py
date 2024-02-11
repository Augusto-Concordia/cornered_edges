import numpy as np
from utils import filter2d, gaussian_kernel, partial_x, partial_y, zero_pad
from skimage.feature import peak_local_max
from skimage.io import imread, imsave
import matplotlib.pyplot as plt


def show_img(img):
    plt.imshow(img, cmap='Greys')

    plt.show()


def save_img(img, name: str, is_float: bool = True):
    if (is_float):
        img = (img * 255).astype('uint8')

    imsave(name, img)


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    response = None

    # YOUR CODE HERE

    Hi, Wi = img.shape

    smoothing_kernel = gaussian_kernel(l=window_size)
    padded_img = zero_pad(img, window_size // 2, window_size // 2)
    x_derived_img = filter2d(partial_x(padded_img), smoothing_kernel)
    y_derived_img = filter2d(partial_y(padded_img), smoothing_kernel)

    response = np.zeros((Hi, Wi))

    for m in range(Hi):
        for n in range(Wi):
            I_x = x_derived_img[m:m+window_size, n:n+window_size]
            I_y = y_derived_img[m:m+window_size, n:n+window_size]

            I_xx = filter2d((I_x ** 2), smoothing_kernel)
            I_xy = filter2d(I_x * I_y, smoothing_kernel)
            I_yy = filter2d((I_y ** 2), smoothing_kernel)

            smm_det = I_xx * I_yy - (I_xy ** 2)
            smm_trace = I_xx + I_yy

            r = smm_det - k * (smm_trace ** 2)

            response[m, n] = np.sum(r)

    # END YOUR CODE

    return response


def main():
    img = imread('building.jpg', as_gray=True)

    # YOUR CODE HERE

    # settings specific to the "building.jpg" image
    window_size = 3
    sensitivity = 0.05
    threshold = 0.0007

    # Compute Harris corner response
    response = harris_corners(img, window_size, sensitivity)
    print("Computed Harris response")

    # Threshold on response
    threshed_response = response.copy()
    threshed_response[threshed_response <
                      threshold * threshed_response.max()] = 0
    print("Thresholded Harris response")

    # Perform non-max suppression by finding peak local maximum
    corners = peak_local_max(threshed_response, min_distance=window_size)
    print("Found corner points")

    # Visualize results (in 3 subplots in the same figure)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    ax1.imshow(response, cmap='inferno')
    ax1.set_title('Harris Response')
    ax1.axis('off')

    ax2.imshow(threshed_response, cmap='inferno')
    ax2.set_title('Thresholded Harris Response')
    ax2.axis('off')

    ax3.imshow(1 - img, cmap='Greys')
    ax3.plot(corners[:, 1], corners[:, 0], 'r+', markersize=5)
    ax3.set_title('Corners')
    ax3.axis('off')

    plt.show()

    # END YOUR CODE


if __name__ == "__main__":
    main()
