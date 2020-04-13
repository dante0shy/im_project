import cv2
import sys,os,glob


if __name__ =='__main__':

    data_path = glob.glob(os.path.join(os.path.dirname(__file__), 'data', '*'))

    data = {}

    for d in data_path:
        if 'manual1' in d:
            continue
        data[d.split('/')[-1]] = cv2.imread(d)
        print(d)
        cv2.imshow(d.split('/')[-1]
                   , data[d.split('/')[-1]])
        cv2.waitKey()
        cv2.destroyAllWindows()