from asyncio.windows_events import NULL
import numpy as np


def pad_image(img, amount, type=np.int16):
    amount = amount//2
    """Paddanje slike da pikseli nebi bili out of range prilikom primjene kernela

    Args:
        img (_type_): Ulazna slika na koju cemo dodati rub radi primjene kernela(range)
        amount (_type_): Kolicina piksela koje cemo dodati na svaku stranu slike (zavisi od velicine kernela)
    """   

    if (len(img.shape)<3):
        h_original,w_original=img.shape
        c_original=1
        padded_img = np.zeros(
        (h_original+amount*2, w_original+amount*2), type)
        padded_img[amount:h_original+amount, amount:w_original+amount] = img   
        padded_img = padded_img.reshape(((h_original+amount*2),(w_original+amount*2),1))
    else:
        h_original, w_original, c_original = img.shape
        padded_img = np.zeros(
        (h_original+amount*2, w_original+amount*2, c_original), type)
        padded_img[amount:h_original+amount, amount:w_original+amount] = img   

    h, w, c = padded_img.shape

    # stavljamo originalnu sliku u prethodnu matricu

    padded_img=pad_borders(padded_img,amount,h,w,c)
    padded_img=pad_corners(padded_img, amount, h, w,c)

    #padded_img = padded_img.astype(type)
    padded_img = padded_img.squeeze().astype(type)
    return padded_img


def pad_borders(padded_img, amount, h, w, c=NULL):
    for i in range(c):  # iteriranje kroz kanale 
        padded_img[0:amount, 1:-1, i] = padded_img[amount,
                                                   1:-1, i]  # padding gornjih redaka

        padded_img[h-amount:h, amount:w-amount, i] = padded_img[h -
                                                                amount-1, amount:w-amount, i]  # padding donjih redaka

        for amt in range(amount):  # For loop radi dodavanja
            padded_img[amount:h-amount, w-amount+amt, i] = padded_img[amount:h -
                                                                      amount, w-amount-1, i]  # padding desnog ruba

            padded_img[amount:h-amount, amt, i] = padded_img[amount:h -
                                                             amount, amount, i]  # padding lijevog ruba

    return padded_img


def pad_corners(padded_img, amount, h, w, c=NULL):
    for i in range(c):
        padded_img[0:amount, 0:amount,i] = padded_img[amount,
                                                    amount, i]  # gornji lijevi kut
        padded_img[0:amount, w-amount:w,i] = padded_img[amount,
                                                      w-amount-1,i]  # gornji desni kut
        padded_img[h-amount:h, w-amount:w,i] = padded_img[h -
                                                        amount-1, w-amount-1,i]  # donji desni kut
        padded_img[h-amount:, :amount,i] = padded_img[h -
                                                amount-1, amount,i]  # donji lijevi kut
    return padded_img


