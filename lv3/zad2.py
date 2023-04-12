import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('mtcars.csv')
data.head()
df = pd.DataFrame(data)

mpg = df['mpg'].head(33)
cyl8 = df['cyl'].head(33)

print(mpg)
print(cyl8)
# Figure Size
fig = plt.figure(figsize =(10, 7))
 
# Horizontal Bar Plot
plt.bar(cyl8,mpg)
 
# Show Plot
plt.show()