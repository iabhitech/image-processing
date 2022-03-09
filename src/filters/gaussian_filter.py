import cv2 as cv
import numpy as np
import sys
import os

sys.path.append(os.path.realpath('D:/Coding/Image Processing'))


def show(title, img_path):
    cv.imshow(title, cv.imread(img_path))
    cv.waitKey(0)
    cv.destroyAllWindows()


def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros(
            (image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding),
                    int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (
                            kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output


def gaussian_filter():
    img = cv.imread("assets/ballon-400x300.jpeg", cv.IMREAD_COLOR)
    (B, G, R) = cv.split(img)
    cv.imshow("Original Image", img)
    # zeros = np.zeros(img.shape[:2], dtype="uint8")
    # cv.imshow("Red", cv.merge([zeros, zeros, R]))
    # cv.imshow("Green", cv.merge([zeros, G, zeros]))
    # cv.imshow("Blue", cv.merge([B, zeros, zeros]))

    # merged = cv.merge([B,G,R])
    # cv.imshow("Merged", merged)



    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]])
    kernel = np.divide(kernel, 16)

    b_chan = convolve2D(B, kernel)
    g_chan = convolve2D(G, kernel)
    r_chan = convolve2D(R, kernel)

    output = cv.merge([b_chan, g_chan, r_chan])

    cv.imwrite('2DConvolved.jpg', output)
    show('Output', '2DConvolved.jpg')
    cv.waitKey(0)
    cv.destroyAllWindows()


gaussian_filter()
