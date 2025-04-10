# William Branam
# CISC 442 - PR 1


import cv2 as cv
import numpy as np
import os

#################################################################

# Write a function Convolve (I, H). I is an image of varying size, H is a kernel of varying size.
# The output of the function should be the convolution result that is displayed.

def convolve(I, H):
    h, w, c = I.shape
    kh, kw = H.shape
    pad_h, pad_w = kh // 2, kw // 2
    I_padded = np.pad(I, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')
    result = np.zeros_like(I)

    for i in range(h):
        for j in range(w):
            for ch in range(c):
                region = I_padded[i:i+kh, j:j+kw, ch]
                result[i, j, ch] = np.sum(region * H)
    return result.astype(np.uint8)

#################################################################

# Write a function Reduce(I) that takes image I as input and outputs a copy of the image resampled
# by half the width and height of the input. Remember to Gaussian filter the image before reducing it; 
# use separable 1D Gaussian kernels.

def reduce(I):
    kernel_1d = cv.getGaussianKernel(5, 1)
    kernel = kernel_1d @ kernel_1d.T
    blurred = convolve(I, kernel)
    return blurred[::2, ::2, :]

#################################################################

# Write a function Expand(I) that takes image I as input and outputs a copy of the image expanded, 
# twice the width and height of the input.

def expand(I):
    return cv.resize(I, (I.shape[1]*2, I.shape[0]*2), interpolation=cv.INTER_LINEAR)

#################################################################

# Use the Reduce() function to write the GaussianPyramid(I,n) function, where n is the no. of levels.

def gaussianPyramid(I, n):
    pyramid = [I]
    for i in range(1, n):
        I = reduce(I)
        pyramid.append(I)
        cv.imwrite(f"output/gaussian_level_{i}.png", I)
    return pyramid

#################################################################

# Use the above functions to write LaplacianPyramids(I,n) that produces n level Laplacian pyramid of I.

def laplacianPyramid(I, n):
    g_pyr = gaussianPyramid(I, n)
    l_pyr = []
    for i in range(n-1):
        expanded = expand(g_pyr[i+1])
        if expanded.shape != g_pyr[i].shape:
            expanded = cv.resize(expanded, (g_pyr[i].shape[1], g_pyr[i].shape[0]))
        laplacian = cv.subtract(g_pyr[i], expanded)
        l_pyr.append(laplacian)
        cv.imwrite(f"output/laplacian_level_{i}.png", laplacian)
    l_pyr.append(g_pyr[-1])
    return l_pyr

#################################################################

# Write the Reconstruct(LI,n) function which collapses the Laplacian pyramid LI of n levels 
# to generate the original image. Report the error in reconstruction using image difference.

def reconstruct(LI, n):
    current = LI[-1]
    for i in range(n-2, -1, -1):
        expanded = expand(current)
        if expanded.shape != LI[i].shape:
            expanded = cv.resize(expanded, (LI[i].shape[1], LI[i].shape[0]))
        current = cv.add(expanded, LI[i])
    return current

#################################################################

# Main execution to generate and save outputs for questions 1-6
if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    img_path = os.path.join("images", "lena.png")
    image = cv.imread(img_path)

    # 1. Convolution Output
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    convolved = convolve(image, kernel)
    cv.imwrite("output/convolved.png", convolved)

    # 2. Reduce Output
    reduced = reduce(image)
    cv.imwrite("output/reduced.png", reduced)

    # 3. Expand Output
    expanded = expand(reduced)
    cv.imwrite("output/expanded.png", expanded)

    # 4. Gaussian Pyramid Output handled in gaussianPyramid()
    # 5. Laplacian Pyramid Output handled in laplacianPyramid()
    laplacian_pyr = laplacianPyramid(image, 5)

    # 6. Reconstruction
    reconstructed = reconstruct(laplacian_pyr, 5)
    cv.imwrite("output/reconstructed.png", reconstructed)

    # Error Visualization
    diff = cv.absdiff(image, reconstructed)
    cv.imwrite("output/difference.png", diff)
