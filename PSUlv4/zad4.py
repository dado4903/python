import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Učitavanje podataka
df = pd.read_csv('cars_processed.csv')

# 1. Koliko mjerenja (automobila) je dostupno u datasetu?
print("Broj mjerenja (automobila) u datasetu:", len(df))

# 2. Kakav je tip pojedinog stupca u dataframeu?
print("\nTipovi podataka u dataframeu:")
print(df.dtypes)

# 3. Koji automobil ima najveću cijenu, a koji najmanju?
max_price = df.loc[df['selling_price'].idxmax()]
min_price = df.loc[df['selling_price'].idxmin()]
print("\nAutomobil s najvećom cijenom:\n", max_price)
print("\nAutomobil s najmanjom cijenom:\n", min_price)

# 4. Koliko automobila je proizvedeno 2012. godine?
year_2012 = df[df['year'] == 2012]
print("\nBroj automobila proizvedenih 2012. godine:", len(year_2012))

# 5. Koji automobil je prešao najviše kilometara, a koji najmanje?
max_km = df.loc[df['km_driven'].idxmax()]
min_km = df.loc[df['km_driven'].idxmin()]
print("\nAutomobil s najvećom prijeđenom kilometražom:\n", max_km)
print("\nAutomobil s najmanjom prijeđenom kilometražom:\n", min_km)

# 6. Koliko najčešće automobili imaju sjedala?
most_common_seats = df['seats'].mode().values[0]
print("\nNajčešći broj sjedala u automobilima:", most_common_seats)

# 7. Kolika je prosječna prijeđena kilometraža za automobile s dizel motorom, a koliko za automobile s benzinskim motorom?
diesel_km = df[df['fuel']=='Diesel']['km_driven'].mean()
petrol_km = df[df['fuel']=='Petrol']['km_driven'].mean()
print("\nProsječna prijeđena kilometraža za dizel automobile:", diesel_km)
print("Prosječna prijeđena kilometraža za benzinske automobile:", petrol_km)

# 8. Linearni regresijski model
X = df[['km_driven']].values
y = df[['selling_price']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Prijeđena kilometraža vs. cijena')
plt.xlabel('Prijeđena kilometraža')
plt.ylabel('Cijena')
plt.show
