import numpy as np
import matplotlib.pylab as plt
from skimage import io
from utils import gaussian_kernel, filter2d, partial_x, partial_y


def gradient_magnitude(x_derived_img, y_derived_img):
    if (x_derived_img.shape != y_derived_img.shape):
        return

    magnitude = np.sqrt(x_derived_img * x_derived_img +
                        y_derived_img * y_derived_img)

    return magnitude


def main():
    image_name = 'images/iguana'

    # Load image
    img = io.imread(image_name + '.png', as_gray=True)

    # YOUR CODE HERE

    # Smooth image with Gaussian kernel
    smoothing_kernel = gaussian_kernel()
    smoothed_img = filter2d(img, smoothing_kernel)

    # Compute x and y derivate on smoothed image
    x_derived_img = partial_x(smoothed_img)
    y_derived_img = partial_y(smoothed_img)

    # Compute gradient magnitude
    magnitude_img = gradient_magnitude(x_derived_img, y_derived_img)

    # Visualize results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    ax1.imshow(x_derived_img, cmap='gray')
    ax1.set_title('X derived image')
    ax2.imshow(y_derived_img, cmap='gray')
    ax2.set_title('Y derived image')
    ax3.imshow(magnitude_img, cmap='gray')
    ax3.set_title('Gradient magnitude')
    plt.show()

    # END YOUR CODE


if __name__ == "__main__":
    main()
