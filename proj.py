%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import seaborn as sns
import statistics as st
import re

df=pd.read_csv('kc_house_data.csv')

df.describe()

df.columns

i=0;
# for cleaning date
for index,row in df.date.iteritems():
    temp=re.split("[T]",row);
    if(i==1000):
        break;
    if(temp[1]!='000000'):
        printf(temp[1])
    print(temp[0])

print(df.dtypes)

// droping id
df = df.drop('id', 1)
print(df.dtypes)

#droping these index as they have less then 0.5 bathroom
li=[]
for index,row in df.bathrooms.iteritems():
    if(row<0.5):
        li.append(index)
        print(df.loc[index])

df=df.drop(df.index[li])


year_12=set()

for index,row in df.yr_built.iteritems():
    year_12.add(row)
year_12=list(year_12)
sorted(year_12)

bd12=[]
for yr in range(len(year_12)):
    bd12.append(0)

for yr in range(len(year_12)):
    for index,row in df.bedrooms.iteritems():
        if (row==1 or row==2):
            if(df.yr_built.loc[index]==year_12[yr]):
                bd12[yr]+=1;

bd34=[]
for yr in range(len(year_12)):
    bd34.append(0)

for yr in range(len(year_12)):
    for index,row in df.bedrooms.iteritems():
        if (row==3 or row==4):
            if(df.yr_built.loc[index]==year_12[yr]):
                bd34[yr]+=1;

bd5a=[]
for yr in range(len(year_12)):
    bd5a.append(0)

for yr in range(len(year_12)):
    for index,row in df.bedrooms.iteritems():
        if (row>=5):
            if(df.yr_built.loc[index]==year_12[yr]):
                bd5a[yr]+=1;

plt.plot(year_12, bd12, linewidth=1.5 )
plt.plot(year_12, bd34,  linewidth=1.5)
plt.plot(year_12, bd5a,  linewidth=1.5)
fig = plt.gcf()
fig.set_size_inches(15,8)
plt.legend(['y= 1 and 2 bedrooms','y=3 and 4 bedrooms','y= 5 and more bedrooms'],loc='upper left')

op=0
for index,row in df.yr_built.iteritems():
    if (row==2014 or row==2015):
        if(df.bedrooms.loc[index]==4 or df.bedrooms.loc[index]==3):

            op+=1
print(op)

bd1=[]
for yr in range(len(year_12)):
    bd1.append(0)

for yr in range(len(year_12)):
    for index,row in df.bedrooms.iteritems():
        if (row==1):
            if(df.yr_built.loc[index]==year_12[yr]):
                bd1[yr]+=1;

bd2=[]
for yr in range(len(year_12)):
    bd2.append(0)

for yr in range(len(year_12)):
    for index,row in df.bedrooms.iteritems():
        if (row==2):
            if(df.yr_built.loc[index]==year_12[yr]):
                bd2[yr]+=1;

bd3=[]
for yr in range(len(year_12)):
    bd3.append(0)

for yr in range(len(year_12)):
    for index,row in df.bedrooms.iteritems():
        if (row==3):
            if(df.yr_built.loc[index]==year_12[yr]):
                bd3[yr]+=1;
bd4=[]
for yr in range(len(year_12)):
    bd4.append(0)

for yr in range(len(year_12)):
    for index,row in df.bedrooms.iteritems():
        if (row==4):
            if(df.yr_built.loc[index]==year_12[yr]):
                bd4[yr]+=1;
bd5=[]
for yr in range(len(year_12)):
    bd5.append(0)

for yr in range(len(year_12)):
    for index,row in df.bedrooms.iteritems():
        if (row==5):
            if(df.yr_built.loc[index]==year_12[yr]):
                bd5[yr]+=1;

plt.plot(year_12, bd1, linewidth=1.2 )
plt.plot(year_12, bd2,  linewidth=1.2)
plt.plot(year_12, bd3,  linewidth=1.2)
plt.plot(year_12, bd4,  linewidth=1.2)
plt.plot(year_12, bd5,  linewidth=1.2)
fig = plt.gcf()
fig.set_size_inches(20,8)
plt.legend(['y= 1 bedroom ','y=2 bedrooms','y=3 bedrooms','y=4 bedrooms','y=5 and more bedrooms'],loc='upper left')

no_houses=[]
for yr in range(len(year_12)):
    no_houses.append(0)

for yr in range(len(year_12)):
    for index,row in df.yr_built.iteritems():
        if(year_12[yr]==row):
            no_houses[yr]+=1;

no_houses

plt.bar(year_12,no_houses, width=0.6)
fig = plt.gcf()

fig.set_size_inches(20,8)
plt.show()
#number of houses to year

lpercent=[]
lpercentmed=[]



for yr in range(len(year_12)):
    lpercent.append(0)
    lpercentmed.append(0)


for yr in range(len(year_12)):
    temp=[]
    for index,row in df.sqft_above.iteritems():
        if(year_12[yr]==df.yr_built.loc[index]):
            temp.append((row/(df.sqft_lot.loc[index]))*100)
    lpercentmed[yr]=(np.median(temp))
    lpercent[yr]=(np.mean(temp))

#plot of percent liv to year
plt.plot(year_12,lpercent)
plt.plot(year_12,lpercentmed)
fig = plt.gcf()
fig.set_size_inches(20,8)
plt.ylim(0,100)
plt.legend(['y=mean of all houses','y= median of all houses'],loc='upper left')

plt.plot(year_12,yesq)
plt.plot(year_12,yesqmed)
fig = plt.gcf()
fig.set_size_inches(20,8)

plt.legend(['y=mean of all houses','y= median of all houses'],loc='upper left')

difbedromnumber = []
for index,row in  df1.bedrooms.iteritems():
    if(row not in difbedromnumber):
        difbedromnumber.append(row)


count_of_each_room = []
bedroom = list(df1.bedrooms)
for i in difbedromnumber:
    count_of_each_room.append(bedroom.count(i))
for i in range(len(difbedromnumber)):
    difbedromnumber[i] = str(difbedromnumber[i])
from_one_to_five = ["1","2","3","4","5",">=6"]

number_from_125 = count_of_each_room[:5]
sum(count_of_each_room)

number_from_125.append(sum(count_of_each_room[5:]))

plt.bar(from_one_to_five,number_from_125)
plt.xlabel("Number of rooms")
plt.ylabel("Number of houses")

number_of_floors = []
for index,row in df1.floors.iteritems():
    if(row not in number_of_floors):
        number_of_floors.append(row)

number_of_floors.sort()
number_of_floors_count = []
for i in number_of_floors:
    number_of_floors_count.append(list(df1.floors).count(i))
for i in range(len(number_of_floors)):
    number_of_floors[i] = str(number_of_floors[i])
plt.bar(number_of_floors,number_of_floors_count)
fig = plt.gcf()
fig.set_size_inches(15,6)
plt.xlabel("Number of floors")
plt.ylabel("Number of houses")


def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b
X = df1.sqft_above
Y = df1.price
a, b = best_fit(X, Y)
plt.scatter(X, Y,marker='.',c = 'green')
yfit = [a + b * xi for xi in X]
plt.plot(X, yfit)
plt.xlabel("sqft_above")
plt.ylabel("price")
fig = plt.gcf()
fig.set_size_inches(15,7)


bedroom = []
for index,row in  df1.bedrooms.iteritems():
    if(row not in bedroom):
        bedroom.append(row)

bedroom.sort()
median_price_nbr = []
median_price_br = []
for i in bedroom:
    temp1 = []
    temp2 = []
    for index,row in df1.bedrooms.iteritems():
        if(row == i and df1.sqft_basement[index] == 0):
            temp1.append(df1.price[index])
        if(row == i and df.sqft_basement[index] != 0):
            temp2.append(df1.price[index])
    median_price_nbr.append(np.median(temp1))
    median_price_br.append(np.median(temp2))

x = np.arange(11)
plt.bar(x+0.0,median_price_nbr,color = 'b', width = 0.25,label="without basement")
plt.bar(x+0.25,median_price_br,color = 'g', width = 0.25,label="with basement")
fig = plt.gcf()
fig.set_size_inches(15,8)
plt.xlabel("Number of rooms (blue without basement & grean with basement)")
plt.ylabel("price")
plt.legend()
plt.show()

count = 1
for index,row in df1.bedrooms.iteritems():
    if(df.bedrooms.loc[index] == 0):
        count+=1
print(count)
one ,two,three,four,five,six,seven,= [],[],[],[],[],[],[]

for index,row in df1.bedrooms.iteritems():
    if row == 1:
        one.append(df1.price.loc[index])
    elif row == 2:
        two.append(df1.price.loc[index])
    elif row == 3:
        three.append(df1.price.loc[index])
    elif row == 4:
        four.append(df1.price.loc[index])
    elif row == 5:
        five.append(df1.price.loc[index])
    elif row == 6:
        six.append(df1.price.loc[index])
    else :
        seven.append(df1.price.loc[index])
from pylab import axes
data = [one,two,three,four,five,six,seven]
plt.boxplot(data, patch_artist=True)
fig = plt.gcf()
fig.set_size_inches(15,8)
axe = plt.gca()
plt.xlabel("numbers of rooms")
plt.ylabel("price")
axe.set_ylim([0,2000000])
ax = axes()
ax.set_xticklabels(['1','2','3','4','5','6','7_above'])
plt.show()

o25,six,s,e,n,t,elnab =[],[],[],[],[],[],[]
for index,row in df.grade.iteritems():
    if row < 5:
        o25.append(df.price.loc[index])
    elif row == 6:
        six.append(df.price.loc[index])
    elif row == 7:
        s.append(df.price.loc[index])
    elif row == 8:
        e.append(df.price.loc[index])
    elif row == 9:
        n.append(df.price.loc[index])
    elif row == 10:
        t.append(df.price.loc[index])
    else:
        elnab.append(df.price.loc[index])

x = ["Upto5","six","seven","eight","nine","ten","eleven above"]
y = [np.mean(o25),np.mean(six),np.mean(s),np.mean(e),np.mean(n),np.mean(t),np.mean(elnab)]

plt.bar(x,y)
fig = plt.gcf()
fig.set_size_inches(15,6)
plt.xlabel("Grade")
plt.ylabel("Cost")
