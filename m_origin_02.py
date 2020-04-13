import cv2
import sys,os,glob
import numpy as np
import gif2numpy
from skimage.segmentation import watershed
'''gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('gradient', gradient)

tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv2.imshow('tophat', tophat)

blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('blackhat', blackhat)'''
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

def process(img):
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imshow('open', opening)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return opening

def seg_s(img):
    xg = img[:,:,1]
    # xg = img
    cv2.imshow('grean', xg)
    cv2.waitKey()
    cv2.destroyAllWindows()
    xt = np.copy(xg)
    m1 = xt.mean()
    xt[xt>m1] = m1#[xt > m1]

    m2 = xt.mean()
    xt[xt>m2] = m2#[xt>m2]
    m3 = xt.mean()
    xt[xt<m3]= 0
    xt[xt>=m3] = 1
    n = xt.sum()

    cv2.imshow('wtf', xt)
    cv2.waitKey()
    cv2.destroyAllWindows()

    xin = xg * xt
    xout = xin
    m1 = xout.sum() / n
    xout[xout> m1 ] = m1
    m2 = xout.sum()/ n
    xout[xout> m2 ] = m2
    m3 = xout.sum()/ n
    xout[xout > m3] = 0
    xout[xout >0] = 255

    cv2.imshow('wtf', xout)
    cv2.waitKey()
    cv2.destroyAllWindows()

def seg(img):
    # xg = img[:,:,1]

    xg = np.copy(img[:,:,1])
    cv2.imshow('img', xg)
    cv2.waitKey()
    cv2.destroyAllWindows()

    mask = np.copy(xg)
    m1 = mask.mean()
    mask[mask > m1] = m1  # [xt > m1]

    m2 = mask.mean()
    mask[mask > m2] = m2  # [xt>m2]
    m3 = mask.mean()
    mask[mask < m3] = 0
    tmp = np.copy(mask)
    tmp[tmp >= m3] = 255
    mask[mask >= m3] = 1
    n = mask.sum()
    cv2.imshow('img', tmp)
    cv2.waitKey()
    cv2.destroyAllWindows()

    xin = xg * mask
    xout = xin
    m1 = xout.sum() / n
    xout[xout > m1] = m1
    m2 = xout.sum() / n
    xout[xout > m2] = m2
    m3 = xout.sum() / n
    xout[xout > m3] = 0
    tmp = np.copy(xout)
    tmp[tmp > 0] = 255
    xout[xout > 0] = 2
    # tmp = cv2.morphologyEx(tmp, cv2.MORPH_OPEN, kernel)
    cv2.imshow('om', tmp)
    cv2.waitKey()
    cv2.destroyAllWindows()

    xout[xout == 1] = 0
    image_segmented = xout.astype(np.uint8)
    image_segmented [image_segmented >1 ] =255
    # closed = cv2.erode(image_segmented, None, iterations=2)
    cv2.imshow('img res', image_segmented)
    cv2.waitKey()

    cv2.destroyAllWindows()
    return image_segmented


if __name__ =='__main__':

    data_path = glob.glob(os.path.join(os.path.dirname(__file__), 'data', '*'))

    data = {}

    for d in data_path:
        if 'manual1' in d:
            gif = gif2numpy.convert(d)
            continue
        elif '24' not in d:
            continue
        data[d.split('/')[-1]] = cv2.imread(d)
        print(d)
    gt = gif[0][0]
    cv2.imshow('gt', gt)
    cv2.waitKey()
    cv2.destroyAllWindows()
    gt =  gt[:,:,0] == 255


    img = list(data.values())[0]
    cv2.imshow('Origin', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    res = seg(img)
    acc = np.count_nonzero(res == gt)/( res.flatten().shape[0])

    print( acc)


    pass