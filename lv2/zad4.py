import numpy as np
import matplotlib.pyplot as plt
def slika(x,y):
    black=np.zeros((20,20))
    white=255*np.ones((20,20))
    red1=np.hstack((black,white)*(y//2))
    if y%2==0:
        red1=np.hstack((red1,white))
    else:
        red1=np.hstack((red1,black))
    red2=np.hstack((white,black)*(y//2))
    if y%2==0:
        red2=np.hstack((red2,white))
    else:
        red2=np.hstack((red2,black))
    img=np.vstack((red1,red2)*(x//2))
        
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()
slika(5,5)