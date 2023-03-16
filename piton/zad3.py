lista = []
while 1:

    x = input()
    if x == "done":
        break
    if x.isdigit():
        lista.append(float(x))

lista.sort()
print("max ", lista[-1], " min", lista[0])
print(len(lista), "elemenata")
print(lista)
