# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv


def main():
    print('loading images...')
    #imgR = cv.pyrDown(cv.imread(cv.samples.findFile('aloeR.jpg')))


    disp1 = cv.imread('mono_depth/00000_depth.png', 0)
    disp2 = cv.imread('stereo_syn_depth/00000_depth.png', 0)

    conf1 = cv.imread('uncertainty/post/raw/uncert/000000_10.png', 0)
    conf2 = cv.imread('uncertainty2/post/raw/uncert/000000_10.png', 0)

    refine = np.zeros_like(disp1, np.float)


    # grab the image dimensions
    h = disp1.shape[0]
    w = disp1.shape[1]

    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            refine[y, x] = disp1[y, x] if conf1[y, x] >= conf2[y, x] else disp2[y, x]


    refine_after = cv.medianBlur(refine, 3)
    # return the thresholded image

    cv.imshow('refine', refine_after)

    cv.waitKey(0)

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
