
# Gaussian and Laplacian Pyramids — PR1

**Author:** William Branam  
**Course:** CISC 442 — Computer Vision  
**Assignment:** Programming Assignment 1

---

## Description

This project implements key image processing operations using OpenCV and NumPy, including:

- Custom convolution using arbitrary kernels
- Image reduction and expansion using Gaussian blur and interpolation
- Gaussian pyramid construction
- Laplacian pyramid construction
- Image reconstruction from Laplacian pyramids
- Visualization of image reconstruction error

Each output is saved in an organized `output/` directory for easy viewing and grading.

---

## Project Structure

```
project_root/
│
├── images/
│   └── lena.png                  # Input image used for processing
│
├── output/
│   ├── convolved.png             # Output of custom convolution
│   ├── reduced.png               # Reduced image
│   ├── expanded.png              # Expanded image
│   ├── gaussian_level_*.png      # Gaussian pyramid levels
│   ├── laplacian_level_*.png     # Laplacian pyramid levels
│   ├── reconstructed.png         # Image reconstructed from Laplacian pyramid
│   └── difference.png            # Difference between original and reconstructed image
│
├── template.py  # Main script
└── README.txt
```

---

## How to Run

1. **Set up a virtual environment** (recommended):
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

2. **Install dependencies**:
    ```bash
    pip install opencv-python numpy
    ```

3. **Run the script**:
    ```bash
    python template.py
    ```

4. **Check the `output/` folder** for results.

---

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy

Install via:
```bash
pip install opencv-python numpy
```

---

## Features Implemented

| Feature                | Function                  |
|------------------------|---------------------------|
| Custom Convolution     | `convolve(I, H)`          |
| Image Reduction        | `reduce(I)`               |
| Image Expansion        | `expand(I)`               |
| Gaussian Pyramid       | `gaussianPyramid(I, n)`   |
| Laplacian Pyramid      | `laplacianPyramid(I, n)`  |
| Image Reconstruction   | `reconstruct(LI, n)`      |

---

## Notes

- Input image must be placed in the `images/` directory and named `lena.png`.
- All generated outputs are saved in the `output/` folder automatically.

---
