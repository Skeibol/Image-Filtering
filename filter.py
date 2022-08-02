from apply_kernel import apply_kernel_grayscale, apply_kernel_rgb, apply_kernel_dilation, apply_kernel_erosion, \
            apply_kernel_dilation_rgb, apply_kernel_erosion_rgb, apply_kernel_matchPattern, apply_kernel_matchPattern_rgb
from init_kernel import init_kernel
import numpy as np

def filter(img,kernel_type="blur",kernel_size=3):
    
    """Funkcija primjenjuje kernel na sliku
    
    Args:
        img (_type_): Ulazna slika na koju cemo primjeniti kernel
        kernel_type (_type_): Kernel koji zelimo primjeniti(findLines,findLinesBig,sharpen,horizontalEdge)
        kernel_size (_type_): x*x dimenzije kernela - 0 za isti, 1 za veci, 3 za veci"""
    
    #Lista kernela (Unesi kernel size za velicinu)
    kernel=init_kernel(kernel_type,kernel_size)
    h_kernel,_ = kernel.shape
    pad_amount = h_kernel//2 #Cijelobrojno djeljenje dimenzije kernela sa dvojkom nam daje kolicinu piksela koje trebamo dodati na sliku da nebi bili out of range
    
    new_img = np.zeros_like(img) #stvaranje nove prazne slike u koju cemo ubacivati filtrirane piksele
    print(new_img.shape,"<------ shape slike na kojoj radimo")

    if len(new_img.shape)==3:
        new_img = apply_kernel_rgb(img,new_img,kernel,pad_amount)
        #Dodat novi nacin filtriranja?
          
    else:
        new_img = apply_kernel_grayscale(img,new_img,kernel,pad_amount)
    
    new_img = new_img.astype(np.int16)
    return new_img

def rgbToGray(img):
    h,w,c = img.shape
    new_img = np.zeros_like(img)
    for i in range(h):
        for i1 in range(w):
            new_img[i,i1] = (img[i,i1,0]*0.11)+(img[i,i1,1]*0.59)+(img[i,i1,2]*0.3)

    return new_img

def dilate(img, kernel_size=3,kernel_type="erode/dilate"):
    """Funkcija primjenjuje kernel na sliku

    Args:
        img (_type_): Ulazna slika na koju cemo primjeniti kernel
        kernel_type (_type_): Kernel koji zelimo primjeniti(findLines,findLinesBig,sharpen,horizontalEdge)
        kernel_size (_type_): x*x dimenzije kernela - 0 za isti, 1 za veci, 3 za veci"""

    # Lista kernela (Unesi kernel size za velicinu)
    kernel = init_kernel(kernel_type, kernel_size)
    h_kernel, _ = kernel.shape
    pad_amount = h_kernel//2  # Cijelobrojno djeljenje dimenzije kernela sa dvojkom nam daje kolicinu piksela koje trebamo dodati na sliku da nebi bili out of range

    # stvaranje nove prazne slike u koju cemo ubacivati filtrirane piksele
    new_img = np.zeros_like(img)
    print(new_img.shape, "<------ shape slike na kojoj radimo")

    if len(new_img.shape) == 3:
        new_img = apply_kernel_dilation_rgb(img, new_img, kernel, pad_amount)

    else:
        new_img = apply_kernel_dilation(img, new_img, kernel, pad_amount)

    new_img = new_img.astype(np.int16)
    return new_img


def erode(img, kernel_size=3,kernel_type="erode/dilate"):
    """Funkcija primjenjuje kernel na sliku

    Args:
        img (_type_): Ulazna slika na koju cemo primjeniti kernel
        kernel_type (_type_): Kernel koji zelimo primjeniti(findLines,findLinesBig,sharpen,horizontalEdge)
        kernel_size (_type_): x*x dimenzije kernela - 0 za isti, 1 za veci, 3 za veci"""

    # Lista kernela (Unesi kernel size za velicinu)
    kernel = init_kernel(kernel_type, kernel_size)
    h_kernel, _ = kernel.shape
    pad_amount = h_kernel//2  # Cijelobrojno djeljenje dimenzije kernela sa dvojkom nam daje kolicinu piksela koje trebamo dodati na sliku da nebi bili out of range

    # stvaranje nove prazne slike u koju cemo ubacivati filtrirane piksele
    new_img = np.zeros_like(img)
    print(new_img.shape, "<------ shape slike na kojoj radimo")

    if len(new_img.shape) == 3:
        new_img = apply_kernel_erosion_rgb(img, new_img, kernel, pad_amount)

    else:
        new_img = apply_kernel_erosion(img, new_img, kernel, pad_amount)

    new_img = new_img.astype(np.int16)
    return new_img


def matchPattern(img, pattern, ret_heatmap=False):
    """Funkcija pronalazi pattern u slici

    Args:
        img (_type_): Ulazna slika na kojoj trazimo pattern
        pattern (_type_): Pattern koji trazimo na ulaznoj slici
        ret_heatmap (_type_): Da li funkcija vraca heatmap pronadenih objekata

    """
    
    # stvaranje nove prazne slike u koju cemo ubacivati filtrirane piksele
    # NEEDS WORK

    if len(img.shape)==3:
        new_img = np.zeros_like(img[:,:,0])
        img_copy = np.array(img)
        new_img, heatmap = apply_kernel_matchPattern_rgb(img,new_img,pattern,img_copy)
    else: 
        new_img=np.zeros_like(img)
        img_copy = np.array(img)
        new_img, heatmap = apply_kernel_matchPattern(img,new_img,pattern,img_copy)

    heatmap = heatmap.astype(np.int16)
    new_img = new_img.astype(np.int16)

    if ret_heatmap == True:
        return new_img, heatmap
    else:
        return new_img
