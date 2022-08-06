import numpy as np

def bitor(image,image2,shape,shape2):
    new_img = np.zeros_like(image)

    for i in range(shape[0]):
        for i1 in range(shape[1]):
            if any(image[i,i1])!=0:
                new_img[i,i1] = image[i,i1]
            if any(image2[i,i1])!=0 and any(image[i,i1])==0: 
                new_img[i,i1] = image2[i,i1]
            

    return new_img

def bitand(image,image2,shape,shape2):

    new_img = np.zeros_like(image)

    for i in range(shape[0]):
        for i1 in range(shape[1]):
            if any(image[i,i1])!=0 and any(image2[i,i1])!=0: 
                new_img[i,i1] = image[i,i1]
            

    return new_img