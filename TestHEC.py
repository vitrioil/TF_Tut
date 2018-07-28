import numpy as np
import cv2
if __name__ == '__main__':
    #regularize("/home/vitrioil/Downloads/DLBeg/train/","/home/vitrioil/Downloads/DLBeg/regTrain/")
    #regularize("/home/vitrioil/Downloads/DLBeg/test/","/home/vitrioil/Downloads/DLBeg/regTest/")
    X_train,X_test = np.load("D:/HEC/X_train.npy"),np.load("D:/HEC/X_test.npy")#get_data("/home/vitrioil/Downloads/DLBeg/")
    Y_train = np.load("D:/HEC/Y_train.npy")
    split = 10000
    X_tr,X_ts,Y_tr,Y_ts = X_train[:split]/255,X_train[split:]/255,Y_train[:split],Y_train[split:]
    print(np.argmax(Y_tr[-300]))
    cv2.imshow("a",X_tr[-300])
    cv2.waitKey()
    cv2.destroyAllWindows()
   