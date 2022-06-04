import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.util as util
from scipy import ndimage
from skimage.exposure import rescale_intensity


def convolve(img, kernel):
    # check dimensions
    (sh, sw) = img.shape[:2]
    (kh, kw) = kernel.shape[:2]
    # use padding to keep original size
    pad = (kw - 1) // 2
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                             cv2.BORDER_REPLICATE)
    # the output can contain float point values
    output = np.zeros((sh, sw), dtype="float32")

    for y in np.arange(pad, sh + pad):
        for x in np.arange(pad, sw + pad):
            # crop a square around the centre pixel
            roi = img[y - pad:y + pad + 1, x - pad:x + pad + 1]
            # sum all values from element-wise multiplications
            k = (roi * kernel).sum()
            # replace original pixel value with convolution results
            output[y - pad, x - pad] = k

        # make sure the range is still 0-255
    output = rescale_intensity(output, in_range=(0, 255))
    output = (255 * output).astype("uint8")

    return output


#  read image and convert to greyscale
rose = cv2.imread("Resources/rose.jpeg")
rose = cv2.resize(rose, (480, 480))
grey = cv2.cvtColor(rose, cv2.COLOR_RGB2GRAY)

#  add salt and pepper noise
grey_sp = util.random_noise(grey, 's&p', clip=True)

#  results returned was scaled between 0 and 1. Change it to 0 and 255
grey_sp = (grey_sp * 255).astype(np.uint8)

#  add Gaussian noise
grey_gauss = util.random_noise(grey, 'gaussian', clip=True)

grey_gauss = (grey_gauss * 255).astype(np.uint8)
# cv2.imshow('grey_sp', grey_sp)
# cv2.imshow('grey_gauss', grey_gauss)

kernel_avg = 1 / 9 * np.array((
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]), dtype="int")

# kernel as a 5 by 5 average filter
kernel_avg2 = np.ones((5, 5)) / 25

# use the average filter to denoise Gaussian noise
convoleOutput = convolve(grey_gauss, kernel_avg2)

# show results
# cv2.imshow("Convloved Output 1 Average filketer to denoise Guassian noise", convoleOutput)

# use the average filter to denoise s&p noise
convoleOutput = convolve(grey_sp, kernel_avg2)

# show results
# cv2.imshow("Convolved Output 2 Average filter to denoise S&p noise", convoleOutput)

# use a median filter to denoise s&p noise
med_denoised = ndimage.median_filter(grey_sp, 5)
# cv2.imshow("med Denoised", med_denoised)
edges = cv2.Canny(np.uint8(grey), 200, 400)  # 200, 400 here are the double thresholds used for hysteresis analysis
cv2.imshow("edges", edges)

grey = np.float32(grey)
dst = cv2.cornerHarris(grey, blockSize=3, ksize=1, k=0.06)
# blockSize - the size of neighbourhood considered for corner detection,
# size - Aperture parameter of Sobel derivative used,
# k - Harris detector free parameter in the equation. Normally a smaller k gives you more corners

# result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
rose[dst > 0.01 * dst.max()] = [0, 0, 255]

cv2.imshow("rose", rose)
cv2.waitKey(0)
cv2.destroyAllWindows()
