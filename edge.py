import numpy as np
import matplotlib.pylab as plt
from skimage import io
from utils import gaussian_kernel, filter2d, partial_x, partial_y


def show_img(img):
    plt.imshow(1 - img, cmap='Greys')

    plt.show()


def save_img(img, name: str, is_float: bool = True):
    if (is_float):
        img = (img * 255).astype('uint8')

    io.imsave(name, img)


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
    show_img(smoothed_img)
    show_img(x_derived_img)
    show_img(y_derived_img)
    show_img(magnitude_img)

    # Save results
    # save_img(x_derived_img, image_name + '_x_derived.png')
    # save_img(y_derived_img, image_name + '_y_derived.png')
    # save_img(magnitude_img, image_name + '_magnitude_derived.png')

    # END YOUR CODE


if __name__ == "__main__":
    main()
