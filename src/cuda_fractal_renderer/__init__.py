import argparse

import numpy as np

# ruff: noqa: F401
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from PIL import Image
from pycuda.compiler import SourceModule

WIDTH = 4096
HEIGHT = 4096
SCALE_FACTOR = 4
ITERATIONS = 1000000  # 1 million


def load_kernel():
    with open("src/cuda_fractal_renderer/kernel.cu", "r") as f:
        cuda_code = f.read()

    source_module = SourceModule(cuda_code)
    kernel = source_module.get_function("render_fractal")

    return kernel


def generate_parameters(seed: int) -> np.ndarray:
    np.random.seed(seed)

    # chosen by trial and error to get a nice fractal
    min_scale = 1
    max_scale = 1.5
    min_angle = 0
    max_angle = 2 * np.pi
    min_angle_difference = 0.5 * np.pi
    max_angle_difference = 1 * np.pi
    min_bias = 0.25
    max_bias = 0.75

    def generate_half() -> list[float]:
        angle1 = np.random.uniform(min_angle, max_angle)
        angle2 = angle1 + np.random.uniform(min_angle_difference, max_angle_difference)
        bias_angle = np.random.uniform(min_angle, max_angle)
        scale1 = np.random.uniform(min_scale, max_scale)
        scale2 = np.random.uniform(min_scale, max_scale)
        bias_scale = np.random.uniform(min_bias, max_bias)
        return [
            np.cos(angle1) * scale1,
            np.cos(angle2) * scale2,
            np.cos(bias_angle) * bias_scale,
            np.sin(angle1) * scale1,
            np.sin(angle2) * scale2,
            np.sin(bias_angle) * bias_scale,
        ]

    return np.concatenate([generate_half(), generate_half()])


def generate_fractals(kernel, seeds: list[int]) -> list[Image.Image]:
    parameters = []
    for seed in seeds:
        parameters.append(generate_parameters(seed))
    parameters = np.concatenate(parameters)

    seeds_gpu = gpuarray.to_gpu(np.array(seeds, dtype=np.int32))

    parameters_gpu = gpuarray.to_gpu(parameters.astype(np.float32))

    width = np.int32(WIDTH)
    height = np.int32(HEIGHT)

    x_range = np.float32(2.0)
    y_range = np.float32(2.0)
    max_iterations = np.int32(ITERATIONS)
    # chosen by trial and error to get a nice fractal
    power = np.float32(0.2)

    threads_per_block = 1024
    num_seeds = np.int32(len(seeds))
    blocks_per_grid = len(seeds)

    pixels = gpuarray.zeros(width * height * len(seeds), dtype=np.int32)

    print("rendering fractals")
    kernel(
        pixels,
        width,
        height,
        x_range,
        y_range,
        max_iterations,
        parameters_gpu,
        power,
        seeds_gpu,
        num_seeds,
        block=(threads_per_block, 1, 1),
        grid=(blocks_per_grid, 1),
    )

    pixels = pixels.get()
    print("done rendering fractals, converting to images")

    pixels = pixels.reshape((len(seeds), HEIGHT * WIDTH))

    images = []
    for image_index in range(len(seeds)):
        # normalize the pixels to 0-255
        pixel = pixels[image_index].astype(np.float32)
        quantile = np.quantile(pixel, 0.95)

        if quantile == 0:
            positive_pixels = pixel[pixel > 0]
            if len(positive_pixels) == 0:
                quantile = 1
            else:
                quantile = np.min(positive_pixels)

        pixel /= quantile
        pixel = np.clip(pixel, 0, 1)
        pixel *= 255
        pixel = pixel.astype(np.uint8)

        pixel = pixel.reshape((HEIGHT, WIDTH))
        # scale the image down
        image = Image.fromarray(pixel, mode="L").resize(
            (WIDTH // SCALE_FACTOR, HEIGHT // SCALE_FACTOR)
        )
        images.append(image)

    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=1)
    args = parser.parse_args()

    kernel = load_kernel()

    images = []
    seeds = []

    for row in range(args.grid_size):
        for col in range(args.grid_size):
            seed = args.seed + row * args.grid_size + col
            seeds.append(seed)

    images = generate_fractals(kernel, seeds)

    images_grid = []

    for row in range(args.grid_size):
        images_row = []
        for col in range(args.grid_size):
            image = images[row * args.grid_size + col]
            images_row.append(image)
        images_grid.append(images_row)

    image_width = WIDTH // SCALE_FACTOR
    image_height = HEIGHT // SCALE_FACTOR

    # combine the images into a single image
    combined_image = Image.new(
        "RGB",
        (args.grid_size * image_width, args.grid_size * image_height),
    )
    for row in range(args.grid_size):
        for col in range(args.grid_size):
            combined_image.paste(
                images_grid[row][col], (col * image_width, row * image_height)
            )
    # resize the combined image to image size
    combined_image = combined_image.resize(
        (image_width, image_height)
    )

    combined_image.save("output.png")
