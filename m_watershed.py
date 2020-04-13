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
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'extras')

def process(img):
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imshow('open', opening)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return opening

def seg(img):
    # xg = img[:,:,1]
    # cv2.imwrite(os.path.join(output_dir,'watershed_g_24.jpg'), img[:,:,1])
    # cv2.imwrite(os.path.join(output_dir,'watershed_b_24.jpg'), img[:,:,0])
    # cv2.imwrite(os.path.join(output_dir,'watershed_r_24.jpg'), img[:,:,2])

    xg = np.copy(img[:,:,1])
    # cv2.imshow('img', xg)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

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
    # tmp[(img >= range_S_2) *(img <= range_B_2)] = 255
    # cv2.imshow('map', tmp)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # cv2.imwrite(os.path.join(output_dir, 'watershed_hist_24.jpg'), tmp)

    m1 = xg.mean()
    mask = np.zeros_like(xg)
    mask[xg > m1] = 1  #
    # cv2.imshow('img', mask)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    canny = cv2.Canny(xg, 20, 60)*mask
    # # canny = cv2.dilate(canny, kernel, iterations=1)
    # cv2.imshow('canny', canny )
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # cv2.imwrite(os.path.join(output_dir, 'watershed_canny_24.jpg'), canny)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # canny = cv2.dilate(canny, kernel, iterations=1)
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel,iterations= 2)
    # cv2.imwrite(os.path.join(output_dir, 'watershed_close_24.jpg'), canny)

    canny = cv2.erode(canny, kernel, iterations=1)
    # cv2.imshow('canny', canny )
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # cv2.imwrite(os.path.join(output_dir, 'watershed_erode_24.jpg'), canny)

    closed = np.zeros_like(canny)
    closed[mask == 0] = 1
    closed[(xg >= range_S) *(xg <= range_B)] = 1
    closed[(xg <= b[idx-1 if idx >=2 else 0])] = 1
    closed[canny>0] = 2


    image_segmented = watershed(xg, markers = closed)#, beta=1,beta=20, mode='bf'
    image_segmented[image_segmented == 1] = 0
    image_segmented = image_segmented.astype(np.uint8)
    image_segmented [image_segmented >1 ] =255
    # closed = cv2.erode(image_segmented, None, iterations=2)
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
        elif '24_training' not in d:
            continue
        data[d.split('/')[-1]] = cv2.imread(d)
        print(d)
    gt = gif[0][0]
    # cv2.imshow('gt', gt)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    gt =  gt[:,:,0] == 255


    img = list(data.values())[0]
    # cv2.imshow('Origin', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    res = seg(img)
    acc = np.count_nonzero(res == gt)/( res.flatten().shape[0])

    print( acc)
    cv2.imwrite(os.path.join(output_dir,'watershed_res_24.jpg'), res)


    pass