import numpy as np
from matplotlib import pyplot
from matplotlib import pyplot as plt

data = np.loadtxt(open("mtcars.csv","rb"), usecols=(1,2,3,4,5,6), delimiter=",", skiprows=1)
print(data)

pyplot.scatter(data[:,0],data[:,3],c='b',s=data[:,5]*10)
plt.xlabel("mpg")
plt.ylabel("hp")
plt.title("Primjer")
for x in range(2,32):
    plt.text(data[x,0],data[x,3],s=data[x,5],fontsize=6)

print("minimalna vrijednost potrosnje:",min(data[:,0]))
print("maksimalna vrijednost potrosnje:",max(data[:,0]))
print("srednja vrijednost potrosnje:",sum(data[:,0])/len(data[:,0]))
arr=[]

for i,item in enumerate(data[:,1]):
    if item >=6:
        arr.append(data[i,0])
        
print("minimalna vrijednost potrosnje 6 cilindara:",min(arr))
print("maksimalna vrijednost potrosnje 6 cilindara:",max(arr))
print("srednja vrijednost potrosnje 6 cilindara:",sum(arr)/len(arr))
    
plt.show()