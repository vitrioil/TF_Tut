from CNN import CNN
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image,ImageOps
import os
class CAP(CNN):
    def __init__(self,Xs,Ys,batch_size,epoch,lr):
        CNN.__init__(Xs,Ys,batch_size,epoch,lr)
        self.assign_model()
        
    def assign_model(self):
        CNN.model = self.model
    
    def model(self,x_t,training = True):
        with tf.variable_scope("Conv1"):
            z = self.conv2d(x_t, 16, 3, 2)
            z = self.max_pool(z, 2, 2)
        with tf.variable_scope("Conv2"):
            z = self.conv2d(z, 32, 3, 2)
            z = self.max_pool(z, 2, 2)
        with tf.variable_scope("Flatten1"):
            z = tf.layers.flatten(z)
        with tf.variable_scope("Dense1"):
            z = tf.layers.dense(z,units=256,activation=tf.nn.relu)
            z = tf.layers.dropout(z,rate=0.15,training=training)
        with tf.variable_scope("Dense1"):
            logits = tf.layers.dense(z,units=self.Y_train.shape[1],name="logits")
        y = tf.nn.softmax(logits,name="y_hat")
        return y,logits
    
def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
    reverse = False
    i = 0
     
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
    	reverse = True
     
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
    	i = 1
     
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    	key=lambda b:b[1][i], reverse=reverse))
     
    # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes

def gen_dataset():
    path = r"C:\Users\HiteshOza\Documents\TF_Tut\solving_captchas_code_examples\generated_captcha_images"
    X,Y = [],[]
    files = []
    for _,_,f in os.walk(path):
        files = f
        
    for file_name in files:
        Y.append(file_name[:-4])
        X.append(cv2.imread(os.path.join(path,file_name),0))
    X,Y = np.array(X),np.array(Y)
    X,Y = X[:,:,:,np.newaxis],Y
    np.save("C:/Users/HiteshOza/Documents/TF_Tut/CNN/CAP/"+"X.npy",X)
    
    np.save("C:/Users/HiteshOza/Documents/TF_Tut/CNN/CAP/"+"Y.npy",Y)
    
def similar(img1,img2,c1,c2):
    if np.abs(c1[0] - c2[0]) < 4:
        check = (np.abs(img1.shape[0] - img2.shape[0])<2) and (
                np.abs(img1.shape[1] - img2.shape[1])<2) 
        if check and img1.shape[0] == img2.shape[0] and img1.shape[1] == img2.shape[1]:
            return check and (np.sum(np.abs(img1-img2)) < 5)
        return True
    return False
    
    
def img_break(path,colour = False,image=None,show=False,print_str=""):
    if path is not None:
        image=cv2.imread(path,0)
    if colour:
        image = cv2.pyrMeanShiftFiltering(image,11,31)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,threshold = cv2.threshold(image, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 2))
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    _,contours,_ = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    D,C = [],[]
    left = False
    contours = sort_contours(contours,method="left-to-right")[0]
    prev_x,prev_y,prev_w,prev_h = -1,-1,-1,-1
    for c in contours:
        c = cv2.boundingRect(c)
        x,y,w,h = c
        if w>5 and h>5 and not (w<8 and np.abs(w-h)<3):
            if not w>=14:
                left = False
                img = image[y:y+h,x:x+w]
            else:
                if not left:
                    c = x,y,w-w//2,h
                    x,y,w,h = c
                    left = True
                else:
                    c = x+w//2,y,w,h
                    x,y,w,h = c
                    left = False
            similar_img = False
            for indx,d in enumerate(D):
                if similar(d,img,C[indx],c):
                    similar_img = True
            if not similar_img:
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
                D.append(img)
                C.append(c)
            prev_x,prev_y,prev_w,prev_h = x,y,w,h
    if show or len(D) !=4 :
        for d in D:
            bs("image",d)
    if len(D) != 4:
        print(print_str)
        print(len(D))
        assert False
    return D
def bs(name,img):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()
def seg_and_make(X,Y):
    final_X,final_Y = [],[]
    for indx,img in enumerate(X):
        D = img_break(None,image=img,print_str = Y[indx])
        
        for (digit,label) in zip(D,[i for i in Y[indx]]):
            final_X.append(digit)
            final_Y.append(label)

    return np.array(final_X),np.array(final_Y)
            
if __name__ == "__main__":
    #img = "C:/Users/HiteshOza/Documents/TF_Tut/solving_captchas_code_examples/generated_captcha_images/2KD2.png"
    #img_break(img,show=True)
    #gen_dataset()
    training_size = 7000
    X,Y = np.load("C:/Users/HiteshOza/Documents/TF_Tut/CNN/CAP/"+"X.npy"),np.load("C:/Users/HiteshOza/Documents/TF_Tut/CNN/CAP/"+"Y.npy")
    
    X,Y = seg_and_make(X,Y)
    print(X.shape,Y.shape)
    Xs = [X[:training_size,:,:],X[training_size,:,:]]
    Ys = [Y[:training_size],Y[training_size]]
    