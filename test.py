from skimage import io
import utils


def save_img(img, name: str, is_float: bool = True):
    # Save the smoothed image
    if (is_float):
        img = (img * 255).astype('uint8')

    io.imsave(name, img)


def main():
    image_name = 'out/iguana'

    # Load image
    img = io.imread(image_name + '.png', as_gray=True)

    gaussian_kernel = utils.gaussian_kernel()
    smoothed_image = utils.filter2d(img, gaussian_kernel)
    save_img(smoothed_image, image_name + '_smoothed.png')

    x_derived_img = utils.partial_x(smoothed_image)
    save_img(x_derived_img, image_name + '_x_derived.png')

    y_derived_img = utils.partial_y(smoothed_image)
    save_img(y_derived_img, image_name + '_y_derived.png')


if __name__ == "__main__":
    main()
