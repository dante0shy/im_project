import cv2
import sys,os,glob
import numpy as np
import gif2numpy
from skimage.segmentation import watershed

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'extras')
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
    xout[xout > (m3)] = 0
    tmp = np.copy(xout)
    tmp[tmp > 0] = 255
    xout[xout > 0] = 2
    # tmp = cv2.morphologyEx(tmp, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('om', tmp)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    xout[xout == 1] = 0
    image_segmented = xout.astype(np.uint8)

    image_segmented [image_segmented >1 ] =255
    # closed = cv2.erode(image_segmented, None, iterations=2)
    cv2.imshow('img res', image_segmented)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return image_segmented


if __name__ =='__main__':
    # 85.5389 %
    data_path = glob.glob(os.path.join(os.path.dirname(__file__), 'data', '*'))
    data = {}
    for d in data_path:
        if 'manual1' in d:
            gif = gif2numpy.convert(d)
            continue
        elif '24_training' not in d:
            continue
        data[d.split('/')[-1]] = cv2.imread(d)
        print(d)
    gt = gif[0][0]
    img = list(data.values())[0]
    gt =  gt[:,:,0] == 255

    # cv2.imshow('Origin', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    res = seg(img)

    acc = np.count_nonzero(res == gt)/( res.flatten().shape[0])
    print('accuracy {:02f}'.format(acc))
    cv2.imwrite(os.path.join(output_dir,'res_origin_24.jpg'), res)

    pass