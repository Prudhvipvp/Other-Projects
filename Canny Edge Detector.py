from skimage.exposure import rescale_intensity
import numpy as np
import cv2
from skimage.measure import compare_ssim

def convolvea(image, kernel):
    (ih,iw)=image.shape
    (kh,kw)=kernel.shape
    pad=(kw-1)//2
    image=cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_CONSTANT,0)
    output=np.zeros((ih,iw),dtype="float")
    for y in np.arange(pad,ih+pad):
      for x in np.arange(pad,iw+pad):
        roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]  
        k = (roi * kernel).sum()
        output[y - pad, x - pad] = k        
    output = rescale_intensity(output,in_range=(0,255),out_range=(0,255)).astype("uint8")
    return output

def convolve3d(image, kernel):
    (ih,iw,d)=image.shape
    (kh,kw)=kernel.shape
    pad=(kw-1)//2
    image=cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_CONSTANT,0)
    output=np.zeros((ih,iw,d),dtype="float")
    for y in np.arange(pad,ih+pad):
      for x in np.arange(pad,iw+pad):
        roi = image[ :,:,0][y - pad:y + pad + 1, x - pad:x + pad + 1]  
        k = (roi * kernel).sum()
        output[:,:,0][y - pad, x - pad] = k
        roi = image[:,:,1][y - pad:y + pad + 1, x - pad:x + pad + 1]  
        k = (roi * kernel).sum()
        output[:,:,1][y - pad, x - pad] = k
        roi = image[:,:,2][y - pad:y + pad + 1, x - pad:x + pad + 1]  
        k = (roi * kernel).sum()
        output[:,:,2][y - pad, x - pad] = k
    output = rescale_intensity(output,in_range=(0,255),out_range=(0,255)).astype("uint8")
    return output

def convolve(image, kernel):
    (ih,iw)=image.shape
    (kh,kw)=kernel.shape
    pad=(kw-1)//2
    image=cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_CONSTANT,0)
    output=np.zeros((ih,iw),dtype="float")
    for y in np.arange(pad,ih+pad):
      for x in np.arange(pad,iw+pad):
        roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]  
        k = (roi * kernel).sum()
        output[y - pad, x - pad] = k
    return output

def gradient(gray,sobelX,sobelY):
    imagea =convolve(gray, sobelX)
    imageb=convolve(gray, sobelY)
    image=np.hypot(imagea,imageb)
    image= rescale_intensity(image,out_range=(0,255)).astype("uint8")
    return image
    
def gradient3d(image,sobelX,sobelY):
    imagea =convolve3d(image, sobelX)
    imageb=convolve3d(image, sobelY)
    image=np.hypot(imagea,imageb)
    image= rescale_intensity(image,in_range=(0,255),out_range=(0,255)).astype("uint8")
    return image
    
def getangle(teta):
                if(0<=teta<=22.5) or (157.5 <= teta<= 180):
                   return 0
                elif(22.5<teta<=67.5):
                   return 45
                elif(67.5<teta<=112.5):
                   return 90
                elif(112.5<teta<=157.5):
                  return 135
                                                 
def nonmaxsup(image,teta):
        (ih,iw)=image.shape
        pad=1
        output=np.zeros((ih,iw),dtype="float")
        image=cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_CONSTANT,0)
        teta[teta<0]+=180
        for y in range(pad,pad+ih):
           for x in range(pad,pad+iw):
                y1=y-pad
                x1=x-pad
                angle=getangle(teta[y1][x1])
                if(angle== 0) :
                   i=image[y][x+1]
                   j=image[y][x-1]
                elif(angle==45):
                   i=image[y-1][x+1]
                   j=image[y+1][x-1]
                elif(angle==90):
                   i=image[y+1][x]
                   j=image[y-1][x]
                elif(angle==135):
                   i=image[y-1][x-1]
                   j=image[y+1][x+1]            
                if(image[y][x]>i and image[y][x]>j):
                   output[y-pad, x-pad]=image[y][x]
                else:
                   output[y-pad,x-pad]=0      
        return output
  
def hysthreshold(output1,teta,high,low):
        strong=255
        weak=140
        pad=1
        (ih,iw)=output1.shape
        for y in range(ih):
            for x in range(iw):
                q=0
                if(output1[y][x]>high):
                   q=1
                elif(output1[y][x]<=low):
                   q=-1
                if(q==1):
                   output1[y][x]=strong
                elif(q==0):
                   output1[y][x]=weak
                else : 
                  output1[y][x]=0
        output=output1  
        output1=cv2.copyMakeBorder(output1,pad,pad,pad,pad,cv2.BORDER_CONSTANT,0)     
        for y in range(pad,pad+ih):
            for x in range(pad,pad+iw):
                if(output[y-pad][x-pad]==strong):
                  angle=getangle(teta[y-pad][x-pad])  
                  if(angle==0):
                     if(output1[y][x+1]==weak): output[y-pad][x+1-pad]=strong
                     if(output1[y][x-1]==weak): output[y-pad][x-1-pad]=strong
                  if(angle==45):
                     if(output1[y-1][x+1]==weak): output[y-1-pad][x+1-pad]=strong
                     if(output1[y+1][x-1]==weak) :output[y+1-pad][x-1-pad]=strong
                  if(angle==90):
                     if(output1[y+1][x]==weak): output[y+1-pad][x-pad]=strong
                     if(output1[y-1][x]==weak): output[y-1-pad][x-pad]=strong
                  if(angle==135):
                     if(output1[y+1][x+1]==weak): output[y+1-pad][x+1-pad]=strong
                     if(output1[y-1][x-1]==weak): output[y-1-pad][x-1-pad]=strong 
        return output             
                       
def MyCannyEdgeDetector(image,threshold):
        gray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray=cv2.GaussianBlur(gray,(3,3),0.6)  
        grad=gradient(gray,sobelX,sobelY)
        convx =convolve(gray, sobelX)
        convy=convolve(gray, sobelY)
        teta=np.arctan2(convy,convx)
        teta=teta*180/np.pi
        gray=nonmaxsup(grad,teta)
        low=high/9
        gray=hysthreshold(gray,teta,threshold,low)
        return gray
        
sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="float32")
sobelY = np.array((
	[1, 2, 1],
	[0, 0, 0],
	[-1,-2, -1]), dtype="float32")
gauss=np.array((
        [1,4,7,4,1],
        [4,16,26,16,4],
        [7,26,41,26,7],
        [4,16,26,16,4],
        [1,4,7,4,1]),dtype="float")
gauss/=273
    
Blur = np.ones((5, 5), dtype="float") * (1.0 / (5 * 5))
template=cv2.imread('template.png')
image2=cv2.imread('waldo.png')
image=image2
templategrad=gradient3d(template,sobelX,sobelY)
waldograd=gradient3d(image2,sobelX,sobelY)
cv2.imshow('Waldo gradient',waldograd)
cv2.imshow('Template gradient',templategrad)
grayed=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
conv1 = convolvea(grayed, Blur)
gauss=convolvea(grayed,gauss)
conv3=convolve3d(image,Blur)
cv2.imshow('Color Blur',conv3)
cv2.imshow('Gaussian',gauss)
cv2.imshow('Gray Blur',conv1)
high=138
gray=cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
gray=MyCannyEdgeDetector(image2,high)
gray2=cv2.Canny(image2,high/9,high)
cv2.imshow("Inbuilt Canny output",gray2)
cv2.imshow("My canny output",gray)
cv2.imshow("Original",image2)
gray2=gray2.astype("uint8")
gray=gray.astype("uint8")
(score, diff) = compare_ssim(gray,gray2, full=True)
print("SSIM Score = "),
print(score*100),
print("%")
cv2.waitKey(0)           
cv2.destroyAllWindows() 
