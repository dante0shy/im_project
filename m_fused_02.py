import cv2
import sys,os,glob
import numpy as np
import gif2numpy
from skimage.segmentation import watershed

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
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
    # tmp = np.copy(mask)
    # tmp[tmp >= m3] = 255
    mask[mask >= m3] = 1
    n = mask.sum()
    # cv2.imshow('img', tmp)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    xin = xg * mask
    xout = xin
    m1 = xout.sum() / n
    xout[xout > m1] = m1
    m2 = xout.sum() / n
    xout[xout > m2] = m2
    m3 = xout.sum() / n
    xout[xout > m3] = 0
    # tmp = np.copy(xout)
    # tmp[tmp > 0] = 255
    xout[xout > 0] = 2
    # tmp = cv2.morphologyEx(tmp, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('om', tmp)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    a,b = np.histogram(xg.flatten(), bins=20, range=[15, 255])
    idx = a.argmax()
    range_S = b[idx]
    range_B = b[-1]

    canny = cv2.Canny(xg, 20, 60)*mask
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
    closed[(xg <= b[idx-1 if idx >=2 else 0])] = 1
    closed[canny>0] = 2

    # tmp = np.copy(canny)
    # tmp[mask == 0] = 1
    # tmp[(xg >= range_S) *(xg <= range_B)] = 128
    # tmp[(xg <= b[idx-1 if idx >=2 else 0])] = 128
    # tmp[canny>0] = 255
    # cv2.imshow('closed', tmp)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # xout

    fused = np.zeros_like(closed)
    fused[(closed == 2) * (xout == 2)] = 2
    fused[(closed == 2) * (xout == 0)] = 2
    fused[(closed == 1) * (xout == 0)] = 1
    fused[(closed == 1) * (xout == 2)] = 0
    # tmp = np.copy(fused)
    # tmp[fused ==2] = 255
    # tmp[fused ==1] = 128
    # tmp[fused ==0] = 0
    # cv2.imshow('closed', tmp)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # image_segmented = watershed(xg, markers = closed)#, beta=1,beta=20, mode='bf'
    image_segmented = watershed(fused, markers = closed)#, beta=1,beta=20, mode='bf'
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
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extras')

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
    cv2.imwrite(os.path.join(output_dir, 'fused_res_02.jpg'), res)

    pass