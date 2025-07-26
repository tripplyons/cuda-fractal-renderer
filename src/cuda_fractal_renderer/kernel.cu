__global__ void render_fractal(int *pixels, int width, int height, float x_range, float y_range, int max_iterations, float *parameters, float power, int *seeds, int num_seeds)
{
    // where to start in the pixel array
    int base_pixel_idx = blockIdx.x * width * height;
    // where to start in the parameter array (12 parameters per seed)
    int base_parameter_idx = blockIdx.x * 12;
    // start IFS iteration at the origin
    float x = 0.0f;
    float y = 0.0f;
    // PRNG state depends on the thread index and the seed
    int xorshift_state = seeds[blockIdx.x] * 1024 + threadIdx.x;
    for (int i = 0; i < max_iterations; i++)
    {
        // xorshift32 PRNG step
        xorshift_state ^= xorshift_state << 13;
        xorshift_state ^= xorshift_state >> 17;
        xorshift_state ^= xorshift_state << 5;
        // randomly select one of the two 6-parameter transformations
        int parameters_idx = base_parameter_idx + 6 * (xorshift_state & 1);
        // parameters for the selected transformation
        float a = parameters[parameters_idx];
        float b = parameters[parameters_idx + 1];
        float c = parameters[parameters_idx + 2];
        float d = parameters[parameters_idx + 3];
        float e = parameters[parameters_idx + 4];
        float f = parameters[parameters_idx + 5];
        // apply the transformation
        x = a * x + b * y + c;
        y = d * x + e * y + f;
        float distance = pow(x * x + y * y, power);
        x /= distance;
        y /= distance;
        // don't render until the system has "warmed up" away from the origin
        if (i < 100) {
            continue;
        }
        // convert the point to a pixel coordinate
        int pixel_x = (x + x_range) * width / (2 * x_range);
        // check for boundary conditions
        if (pixel_x < 0 || pixel_x >= width) {
            continue;
        }
        int pixel_y = (y + y_range) * height / (2 * y_range);
        // check for boundary conditions
        if (pixel_y < 0 || pixel_y >= height) {
            continue;
        }
        int pixel_idx = base_pixel_idx + pixel_x + pixel_y * width;
        // brighten the current pixel (thread-safe)
        atomicAdd(&pixels[pixel_idx], 1);
    }
}
