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
    a,b = np.histogram(xg.flatten(), bins=20, range=[15, 255])
    idx = a.argmax()
    range_S = b[idx]
    # range_B = b[idx+1]
    # range_S = (range_S+range_B)/2
    range_B = b[-1]

    # range_S_2 = b[-5]
    # range_B_2 = b[-1]
    # tmp = np.zeros_like(xg)
    # tmp[(xg >= range_S) *(xg <= range_B) ] = 255
    # tmp[(xg <= b[idx-2 if idx >=2 else 0])] = 255
    # # tmp[(xg <= b[idx-2 if idx >=2 else 0])] = 255
    # # tmp[(img >= range_S_2) *(img <= range_B_2)] = 255
    # cv2.imshow('map', tmp)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    m1 = xg.mean()
    mask = np.zeros_like(xg)
    mask[xg > m1] = 1  #
    # cv2.imshow('img', mask)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    canny = cv2.Canny(xg, 20, 60)*mask
    # canny = cv2.dilate(canny, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # canny = cv2.dilate(canny, kernel, iterations=1)
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=2)
    canny = cv2.erode(canny, kernel, iterations=1)
    # cv2.imshow('canny', canny )
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    closed = np.zeros_like(canny)
    closed[mask == 0] = 1
    closed[(xg >= range_S) *(xg <= range_B)] = 1
    # closed[(xg <= b[idx-1 if idx >=2 else 0])] = 1
    closed[canny>0] = 2


    image_segmented = watershed(xg, markers = closed)#, beta=1,beta=20, mode='bf'
    image_segmented[image_segmented == 1] = 0
    image_segmented = image_segmented.astype(np.uint8)
    image_segmented [image_segmented >1 ] =255
    # # closed = cv2.erode(image_segmented, None, iterations=2)
    # cv2.imshow('img res', image_segmented)
    # cv2.waitKey()
    #
    # cv2.destroyAllWindows()
    return image_segmented


if __name__ =='__main__':

    data_path = glob.glob(os.path.join(os.path.dirname(__file__), 'data', '*'))

    data = {}

    for d in data_path:
        if 'manual1' in d:
            gif = gif2numpy.convert(d)
            continue
        elif '02' not in d:
            continue
        data[d.split('/')[-1]] = cv2.imread(d)
        print(d)

    img = list(data.values())[0]
    # cv2.imshow('Origin', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    res = seg(img)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extras')
    cv2.imwrite(os.path.join(output_dir, 'watershed_res_02.jpg'), res)

    pass