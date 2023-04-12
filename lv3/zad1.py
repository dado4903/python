import numpy as np
import pandas as pd

mtcars=pd.read_csv('mtcars.csv')
print(mtcars.sort_values(by=['mpg']).head(5))
print(mtcars[mtcars.cyl==8].sort_values(by=['mpg']).tail(3))

new_mtcars = mtcars[mtcars.cyl==6]
print("average 6 cyl mpg:",new_mtcars['mpg'].mean())

cyl4_mtcars = mtcars[(mtcars.wt>=2.0) & (mtcars.wt<=2.2)]
print("average 4 cyl mpg:",cyl4_mtcars['mpg'].mean())

autom_mtcars=mtcars[mtcars.am==1]
print("automatski:",autom_mtcars['am'].count())

man_mtcars=mtcars[mtcars.am==0]
print("manualni:",man_mtcars['am'].count())

autohp_mtcars=mtcars[(mtcars.am==1)&(mtcars.hp>100)]
print("automatski iznad 100:",autohp_mtcars['am'].count())

mtcars["masskg"]=mtcars["wt"]*1000*0.45359237
print(mtcars[["car","masskg"]])

