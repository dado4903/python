y=input("unesi broj izmedu 0 i 1")
try:
    x=float(y)
    if 0.9<=x<=1:
        print("A")
    elif 0.8<=x<=0.9:
        print("B")
    elif 0.7<=x<=0.8:
        print("C")
    elif 0.6<=x<=0.7:
        print("D")
    elif 0<=x<=0.6:
        print("F")
    else:
        print("krivi unos")

except:
    print("krivi unos")
