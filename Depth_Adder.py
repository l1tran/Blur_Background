from cv2 import cv2
import numpy as np
class img:
    def __init__(self,pic):
        self.main = cv2.imread(pic)

        #downsize the resolution if w or h over 1000 pix
        while self.main.shape[0] >= 1000 or self.main.shape[1] >= 1000:
            self.scale = 80
            self.width = int(self.main.shape[1] * self.scale/100)
            self.height = int(self.main.shape[0] * self.scale/100)
            self.main = cv2.resize(self.main,(self.width,self.height),cv2.INTER_NEAREST)

        #create copies of og image to draw on
        self.copy = self.main.copy()
        self.copy2 = self.main.copy()
        #create masks same size as main image to store the mask
        self.mask = np.zeros(self.main.shape[:2],np.uint8)
    
    #Vars to check if mouse click is finished
    drawingLine = False
    drawingRect = False
    drawingLineDone = False
    drawingRectDone = False
    maskType = True
    blurSet = False
    #corners of rectangle for grabcut
    x1,y1,x2,y2 = -1,-1,-1,-1

    #reset the copy image when needed to redraw on image
    def reset_pic(self):
        self.copy = self.main.copy()
        self.copy2 = self.main.copy()
    

def click_event(event,x,y,flags,param):
    #create a rectangle using middle button
    if event == cv2.EVENT_MBUTTONDOWN:
        img1.drawingRect = True
        img1.x1 = x
        img1.y1 = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if img1.drawingRect == True:
            img1.reset_pic()
            cv2.rectangle(img1.copy,(img1.x1,img1.y1),(x,y),(255,0,0),3)
            img1.x2 = x
            img1.y2 = y
    elif event == cv2.EVENT_MBUTTONUP:
        img1.drawingRect = False
        img1.drawingRectDone = True

    #draw line using left mouse button
    if event == cv2.EVENT_LBUTTONDOWN:
        img1.drawingLine = True
        if img1.maskType == True:
            cv2.circle(img1.copy,(x,y),5,(255,255,255),-1)
            cv2.circle(img1.mask,(x,y),5,cv2.GC_FGD,-1)
        else:
            cv2.circle(img1.copy,(x,y),5,(150,150,150),-1)
            cv2.circle(img1.mask,(x,y),5,cv2.GC_BGD,-1)
    elif event == cv2.EVENT_MOUSEMOVE:
        if img1.drawingLine == True:
            if img1.maskType == True:
                cv2.circle(img1.copy,(x,y),5,(255,255,255),-1)
                cv2.circle(img1.mask,(x,y),5,cv2.GC_FGD,-1)
            else:
                cv2.circle(img1.copy,(x,y),5,(150,150,150),-1)
                cv2.circle(img1.mask,(x,y),5,cv2.GC_BGD,-1)
    elif event == cv2.EVENT_LBUTTONUP:
        img1.drawingLine = False
        img1.drawingLineDone = True

def nothing(x):
    img1.blurSet = True
#Create an image class object 
img1 = img('JapanWalk.JPG')

#std variables for Grabcut
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

#create a trackbars for threshold and canny vals
cv2.namedWindow('Masking')
cv2.createTrackbar('Blur','Masking',10,20,nothing)

#detect when 'm' is pressed and toggle between mask pen mode
keyPress = 0
result = img1.main.copy()
fgMask = np.ones(img1.main.shape[:2],np.uint8)
fgMask = (fgMask * 255).astype("uint8")
while True:
    cv2.imshow('Masking',img1.copy)
    cv2.setMouseCallback('Masking', click_event)

    cv2.imshow('Result',result)
    cv2.imshow('Foreground',fgMask)

    #get position of trackbar each time its moved
    kernel_val = cv2.getTrackbarPos('Blur','Masking')
    if kernel_val == 0:
        kernel_val = 1

    if img1.drawingLineDone == True or img1.drawingRectDone == True or img1.blurSet == True:
        #perform grabcut to get mask depending on rect or line mask
        if img1.drawingRectDone == True:
            rect = [img1.x1,img1.y1,img1.x2,img1.y2]
            img1.mask,_,_ = cv2.grabCut(img1.main,img1.mask,rect,bgdModel,
                fgdModel,5,mode=cv2.GC_INIT_WITH_RECT)
        elif img1.drawingLineDone == True:
        #create a mask with drawn outline and merge with rectangular mask
            img1.mask,_,_ = cv2.grabCut(img1.main,img1.mask,None,bgdModel,fgdModel,
                5,cv2.GC_INIT_WITH_MASK)
    

        #extract fg and bg from mask
        fgMask = np.where((img1.mask == cv2.GC_BGD)|(img1.mask == cv2.GC_PR_BGD),0,1).astype('uint8')
        bgMask = np.where((img1.mask == cv2.GC_FGD)|(img1.mask == cv2.GC_PR_FGD),0,1).astype('uint8')
        fgMask = (fgMask * 255).astype("uint8")
        bgMask = (bgMask * 255).astype("uint8")
        fgOutput = cv2.bitwise_and(img1.main,img1.main,mask = fgMask)
        bgOutput = cv2.bitwise_and(img1.main,img1.main,mask = bgMask)

        
        
        #blur background of 
        kernel = np.ones((kernel_val,kernel_val),np.float32)/(kernel_val**2)
        blur = cv2.filter2D(img1.copy2,-1,kernel)
        #apply background mask onto blurred background
        blur_with_mask = cv2.bitwise_and(blur,blur,mask = bgMask)
        #apply blur background mask to foreground+foreground mask
        result = cv2.add(blur_with_mask,fgOutput)
        cv2.imshow('Foreground',fgMask)
        cv2.imshow('Result',result)
        img1.drawingLineDone = False
        img1.drawingRectDone = False
        img1.blurSet = False
    #quit if ESC is pressed
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('m'):
        if keyPress == 0:
            img1.maskType = False
            keyPress = 1
        elif keyPress == 1:
            img1.maskType = True 
            keyPress = 0

cv2.destroyAllWindows()