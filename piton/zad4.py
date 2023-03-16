name=input("filename")
list=[]
word = 'X-DSPAM-Confidence'
with open(name, 'r') as fp:
    lines = fp.readlines()
    for line in lines:
        if line.find(word) != -1:
            linea=line.rsplit()
            """zbroj+=float(linea[1])"""
            list.append(float(linea[1]))
zbroj=float(0)
i=0
for item in list:
    zbroj+=item
zbroj/=len(list)
print(zbroj)
fp.close
