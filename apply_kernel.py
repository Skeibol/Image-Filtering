from pad_image import pad_image
import numpy as np
import cv2

def apply_kernel_rgb(img,new_img,kernel,pad_amount):

    h_kernel,_ = kernel.shape
    h,w,channel = new_img.shape
    padded_img = pad_image(img,h_kernel) #stvaranje paddane slike
    row=pad_amount #kernel na prosirenoj slici ima pocetnu tocku jednaku kolicini piksela koje smo dodali kada smo obavili pre processing   
    while row<h+pad_amount: #kretanje po pikselima slike
        column=pad_amount
        while column<w+pad_amount:
            # multiplikacija kernela sa uzetim komadicem slike
            #print(padded_img[row-pad_amount:row+pad_amount+1,column-pad_amount:column+pad_amount+1,0])
            #print(padded_img.shape)
            b = np.multiply(padded_img[row-pad_amount:row+pad_amount+1,column-pad_amount:column+pad_amount+1,0] , kernel).sum()
            g = np.multiply(padded_img[row-pad_amount:row+pad_amount+1,column-pad_amount:column+pad_amount+1,1] , kernel).sum()
            r = np.multiply(padded_img[row-pad_amount:row+pad_amount+1,column-pad_amount:column+pad_amount+1,2] , kernel).sum()
            if r<0: r=0
            if r>255: r=255
            if g<0: g=0
            if g>255: g=255 
            if b<0: b=0
            if b>255: b=255     
            #pohrana dobivene vrijednosti u kanale nove slike
            new_img[row-pad_amount,column-pad_amount,0] = b
            new_img[row-pad_amount,column-pad_amount,1] = g
            new_img[row-pad_amount,column-pad_amount,2] = r
            column+=1 
        row+=1  
    new_img = new_img.astype(np.int16)

    return new_img

def apply_kernel_grayscale(img,new_img,kernel,pad_amount):
    
    h_kernel,_ = kernel.shape
    h,w = new_img.shape
    padded_img = pad_image(img,h_kernel) #stvaranje paddane slike
    row=pad_amount #kernel na prosirenoj slici ima pocetnu tocku jednaku kolicini piksela koje smo dodali kada smo obavili pre processing
    progress=0
    while row<h+pad_amount: #kretanje po pikselima slike
        column=pad_amount
        while column<w+pad_amount:
            # multiplikacija kernela sa uzetim komadicem slike
            gray = np.multiply(padded_img[row-pad_amount:row+pad_amount+1,column-pad_amount:column+pad_amount+1] , kernel).sum()
            if gray<0: gray=0
            if gray>255: gray=255 
            #pohrana dobivene vrijednosti u kanale nove slike
            new_img[row-pad_amount,column-pad_amount] = gray
            column+=1
        print("{:.2f}".format(((progress)/(h))*100),r"% done.",end="\r") #Progress bar
        row+=1
        progress+=1
    
    new_img = new_img.astype(np.int16)
    return new_img


def apply_kernel_erosion(img,new_img,kernel,pad_amount):
    
    h_kernel,_ = kernel.shape
    h,w = new_img.shape
    padded_img = pad_image(img,h_kernel) #stvaranje paddane slike
    row=pad_amount #kernel na prosirenoj slici ima pocetnu tocku jednaku kolicini piksela koje smo dodali kada smo obavili pre processing
    progress=0
    while row<h+pad_amount: #kretanje po pikselima slike
        column=pad_amount
        while column<w+pad_amount:
            # multiplikacija kernela sa uzetim komadicem slike
            gray = np.multiply(padded_img[row-pad_amount:row+pad_amount+1,column-pad_amount:column+pad_amount+1] , kernel).min()
            if gray<0: gray=0
            if gray>255: gray=255 
            #pohrana dobivene vrijednosti u kanale nove slike
            new_img[row-pad_amount,column-pad_amount] = gray
            column+=1
        print("{:.2f}".format(((progress)/(h))*100),r"% done.",end="\r") #Progress bar
        row+=1
        progress+=1
    
    new_img = new_img.astype(np.int16)
    return new_img

def apply_kernel_dilation(img,new_img,kernel,pad_amount):
    
    h_kernel,_ = kernel.shape
    h,w = new_img.shape
    padded_img = pad_image(img,h_kernel) #stvaranje paddane slike
    row=pad_amount #kernel na prosirenoj slici ima pocetnu tocku jednaku kolicini piksela koje smo dodali kada smo obavili pre processing
    progress=0
    while row<h+pad_amount: #kretanje po pikselima slike
        column=pad_amount
        while column<w+pad_amount:
            # multiplikacija kernela sa uzetim komadicem slike
            gray = np.multiply(padded_img[row-pad_amount:row+pad_amount+1,column-pad_amount:column+pad_amount+1] , kernel).max()
            if gray<0: gray=0
            if gray>255: gray=255 
            #pohrana dobivene vrijednosti u kanale nove slike
            new_img[row-pad_amount,column-pad_amount] = gray
            column+=1
        print("{:.2f}".format(((progress)/(h))*100),r"% done.",end="\r") #Progress bar
        row+=1
        progress+=1
    
    new_img = new_img.astype(np.int16)
    return new_img
    
def apply_kernel_erosion_rgb(img,new_img,kernel,pad_amount):

    h_kernel,_ = kernel.shape
    h,w,channel = new_img.shape
    padded_img = pad_image(img,h_kernel) #stvaranje paddane slike
    row=pad_amount #kernel na prosirenoj slici ima pocetnu tocku jednaku kolicini piksela koje smo dodali kada smo obavili pre processing   
    while row<h+pad_amount: #kretanje po pikselima slike
        column=pad_amount
        while column<w+pad_amount:
            
            b = padded_img[row-pad_amount:row+pad_amount+1,column-pad_amount:column+pad_amount+1,0].min()
            g = padded_img[row-pad_amount:row+pad_amount+1,column-pad_amount:column+pad_amount+1,1].min()
            r = padded_img[row-pad_amount:row+pad_amount+1,column-pad_amount:column+pad_amount+1,2].min()    
            #pohrana dobivene vrijednosti u kanale nove slike
            new_img[row-pad_amount,column-pad_amount,0] = b
            new_img[row-pad_amount,column-pad_amount,1] = g
            new_img[row-pad_amount,column-pad_amount,2] = r
            column+=1 
        row+=1  
    new_img = new_img.astype(np.int16)

    return new_img

def apply_kernel_dilation_rgb(img,new_img,kernel,pad_amount):

    h_kernel,_ = kernel.shape
    h,w,channel = new_img.shape
    padded_img = pad_image(img,h_kernel) #stvaranje paddane slike
    row=pad_amount #kernel na prosirenoj slici ima pocetnu tocku jednaku kolicini piksela koje smo dodali kada smo obavili pre processing   
    while row<h+pad_amount: #kretanje po pikselima slike
        column=pad_amount
        while column<w+pad_amount:
            # multiplikacija kernela sa uzetim komadicem slike
            #print(padded_img[row-pad_amount:row+pad_amount+1,column-pad_amount:column+pad_amount+1,0])
            #print(padded_img.shape)
            b = padded_img[row-pad_amount:row+pad_amount+1,column-pad_amount:column+pad_amount+1,0].max()
            g = padded_img[row-pad_amount:row+pad_amount+1,column-pad_amount:column+pad_amount+1,1].max()
            r = padded_img[row-pad_amount:row+pad_amount+1,column-pad_amount:column+pad_amount+1,2].max()   
            #pohrana dobivene vrijednosti u kanale nove slike
            new_img[row-pad_amount,column-pad_amount,0] = b
            new_img[row-pad_amount,column-pad_amount,1] = g
            new_img[row-pad_amount,column-pad_amount,2] = r
            column+=1 
        row+=1  
    new_img = new_img.astype(np.int16)

    return new_img

def apply_kernel_matchPattern(img,heatmap,pattern,original):
    kernel = pattern
    h_kernel,w_kernel = kernel.shape
    h,w = heatmap.shape
    if len(original.shape)==3:rect_color=(0,255,0)
    else: rect_color = 255
    padh=0
    padw=0
    if h_kernel%2!=0: padh=1
    if w_kernel%2!=0: padw=1
    
    for i in range(0+h_kernel//2,(h-h_kernel//2)-2):
        for i1 in range(0+w_kernel//2,(w-w_kernel//2)-2):
            sum_roi = img[i-h_kernel//2:i+h_kernel//2+padh,i1-w_kernel//2:i1+w_kernel//2+padw] - kernel
            sum_roi = np.average(np.absolute(sum_roi))
            if sum_roi<115:
                top_left = (i1-w_kernel//2,i-h_kernel//2)
                bot_right = (i1+w_kernel//2,i+h_kernel//2)
                sum_roi = 0 
                cv2.rectangle(original,top_left,bot_right,rect_color,thickness=1)
               
            else: sum_roi = 255

            heatmap[i,i1] = sum_roi
    
    return original, heatmap
       
def apply_kernel_matchPattern_rgb(img,heatmap,pattern,original):
    kernel = pattern
    h_kernel,w_kernel,c_kernel = kernel.shape
    h,w,c = img.shape
    if len(original.shape)==3:rect_color=(0,255,0)
    else: rect_color = 255
    padh=0
    padw=0
    if h_kernel%2!=0: padh=1
    if w_kernel%2!=0: padw=1
    
    for i in range(0+h_kernel//2,(h-h_kernel//2)-2):
        for i1 in range(0+w_kernel//2,(w-w_kernel//2)-2):
            sum_roi_r = img[i-h_kernel//2:i+h_kernel//2+padh,i1-w_kernel//2:i1+w_kernel//2+padw,0] - kernel[:,:,0]
            sum_roi_b = img[i-h_kernel//2:i+h_kernel//2+padh,i1-w_kernel//2:i1+w_kernel//2+padw,1] - kernel[:,:,1]
            sum_roi_g = img[i-h_kernel//2:i+h_kernel//2+padh,i1-w_kernel//2:i1+w_kernel//2+padw,2] - kernel[:,:,2]
            sum_roi_r = np.average(np.absolute(sum_roi_r))
            sum_roi_g = np.average(np.absolute(sum_roi_g))
            sum_roi_b = np.average(np.absolute(sum_roi_b))
            sum_roi = np.average([sum_roi_b,sum_roi_g,sum_roi_r])
            if sum_roi<115:
                top_left = (i1-w_kernel//2,i-h_kernel//2)
                bot_right = (i1+w_kernel//2,i+h_kernel//2)
                sum_roi = 0 
                cv2.rectangle(original,top_left,bot_right,rect_color,thickness=1)
                
            else: sum_roi = 255

            heatmap[i,i1] = sum_roi
        
    return original, heatmap
       