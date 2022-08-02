from pad_image import pad_image
import numpy as np


def init_kernel(kernel_type="blur", kernel_size=3):
    """Funkcija inicijalizira oblik i velicinu kernela

    Args:
        kernel_type (_type_): Kernel koji zelimo primjeniti(findLines,findLinesBig,sharpen,horizontalEdge)
        kernel_size (_type_): x*x dimenzije kernela - 0 za isti, 1 za veci, 3 za veci"""

    # Lista kernela (Unesi kernel size za velicinu)
    if kernel_type == "findLines":
        kernel = np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1, ],
        ])
        kernel = pad_image(kernel, kernel_size-3)

    if kernel_type == "blur":
        kernel = np.array([
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625]
        ], dtype=np.float32)

        kernel = pad_image(kernel, kernel_size-3,np.float32)

    if kernel_type == "sharpen":
        kernel = np.array([
            [0, -1, 0, ],
            [-1, 5, -1, ],
            [0, -1, 0, ],
        ])

        kernel = pad_image(kernel, kernel_size-3)

    if kernel_type == "horizontalEdge":
        kernel = np.array([[2,  2, 2, 2, 2],
                           [1,  1, 1, 1, 1],
                           [0,  0, 0, 0, 0],
                           [-1,  -1, -1, -1, -1],
                           [-2,  -2, -2, -2, -2]
                           ])
        kernel = pad_image(kernel, kernel_size-3, type="int")

    if kernel_type == "erode/dilate":
        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]
                           ])
        kernel = pad_image(kernel, kernel_size-3, type="int")

    print(kernel, "<------ kernel")
    return kernel
