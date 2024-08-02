import dm_pix as pix
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained('microsoft/resnet-50')


def load_image_hf(path):
    """Load an image with HF api.

    This method is a little slower, bc all operations are done with numpy.
    It also returns a dictionary with a list inside, which needs to be extracted out.
    The advantage of this method is that we use the same image transformations that was used to train the model.

    This method will also be cached to further speed up loading times.
    """
    return image_processor(open_image(path), data_format='channels_last', return_tensors='jax')[
        'pixel_values'
    ][0]


# image augmentations are implemented in jax for speed


def random_rotate(key, image, lower, upper):
    angle = jax.random.uniform(key, (), minval=lower, maxval=upper)
    return pix.rotate(image, angle)


# jittable
def random_augment_image(key, image):
    flip_key, rotate_key = jrand.split(key)
    image = pix.random_flip_left_right(flip_key, image, probability=0.5)
    return random_rotate(rotate_key, image, -15, 15)
    # return image.block_until_ready()


# the following replicates the HF transformation pipeline for reference.


def open_image(path):
    with Image.open(path) as image:
        return image.convert('RGB')


def normalize(image, mean, std):
    return (image - mean) / std


def normalize_imagenet(image):
    return normalize(
        image,
        mean=jnp.array([0.485, 0.456, 0.406]),
        std=jnp.array([0.229, 0.224, 0.225]),
    )


@jax.jit
def center_crop_rescale_normalize(image):
    image = pix.center_crop(image, 224, 224)
    image = image / 255  # rescale
    return normalize_imagenet(image)


def get_resize_dims_with_aspect(image, shortest_side=256):
    height = image.height
    width = image.width
    if width == height:
        new_width = shortest_side
        new_height = shortest_side
    elif width < height:
        new_width = shortest_side
        new_height = int(shortest_side * height / width)
    else:
        new_height = shortest_side
        new_width = int(shortest_side * width / height)
    return new_height, new_width


# jax resize has greatly lower precision
# image = jax.image.resize(image, (new_width, new_height, 3), method='cubic') # jax resize precision good for bfloat16


def preprocess(image):
    new_height, new_width = get_resize_dims_with_aspect(image)
    image = image.resize(
        (new_width, new_height), resample=3
    )  # hardcoded from inspecting the autoimageprocessor config
    image = jnp.array(image)
    return center_crop_rescale_normalize(image)


# casting to bfloat16 is necessary bc of type casting rules
# https://jax.readthedocs.io/en/latest/jep/9407-type-promotion.html


def load_image_jax(path):
    return preprocess(open_image(path))


def load_augment(path, key=42):
    key = jrand.PRNGKey(key) if isinstance(key, int) else key
    return random_augment_image(key, load_image_jax(path))


def test_resize(path):
    # test that the custom jax pipeline gives the same result as the autoimageprocessor pipeline
    image_hf = load_image_hf(path)
    image_jax = load_image_jax(path)
    # image_jax_resize = preprocess_jax_resize(image)

    # test a few different tolerances
    print(f'Testing {path}')
    for rtol in [1e-05, 1e-04]:
        for atol in [1e-08, 1e-07]:
            # allclose is not symmetric
            if not (
                np.allclose(image_hf, image_jax, rtol=rtol, atol=atol)
                and np.allclose(image_jax, image_hf, rtol=rtol, atol=atol)
            ):
                print(f'Jax and HF differ for rtol={rtol}, atol={atol}')


def test_augment(path):
    # assert that the augmentation is different, then save images for visual inspection
    image = open_image(path)
    new_height, new_width = get_resize_dims_with_aspect(image)
    image = image.resize(
        (new_width, new_height), resample=3
    )  # hardcoded from inspecting the autoimageprocessor config
    image = jnp.array(image)
    # image = pix.center_crop(image, 256, 256)
    key = jrand.key(0)
    # perform 8 augmentations
    # import tempfile
    # from pathlib import Path

    # tmpdir = tempfile.mkdtemp()
    for i in range(8):
        key, augment_key = jrand.split(key)
        augmented_image = random_augment_image(augment_key, image)
        print(augmented_image)
        augmented_image = Image.fromarray(np.asarray(augmented_image))
        # save to tmpdir
        # augmented_image.save(Path(tmpdir) / Path(Path(f'{path}').stem + f'_augment{i}.png'))
    # image = Image.fromarray(np.asarray(image))
    # image.save(Path(tmpdir) / Path(Path(f'{path}').stem + '_og.png'))
    # return tmpdir
    # make sure to clean tmpdir afterwards


if __name__ == '__main__':
    mimic_path = '/srv/store/Data/MIMIC-CXR-JPG/image_here.jpg'
    nih_path = '/srv/store/Data/NIH-ChestXray-14/images/00000001_001.png'

    # our method is close enough in precision
    test_resize(mimic_path)
    test_resize(nih_path)

    # visual inspection
    test_augment(mimic_path)
    test_augment(nih_path)
