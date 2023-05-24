

# # 1. Different numpy array operations




import numpy as np


# ## arange( )




a=np.arange(1.5,10.5,2,dtype=float)
print(a)


# ## zeros( )




zero=np.zeros((3,3),dtype=int,order='C')
print(zero)


# ## ones ( )




ones=np.ones((3,2),dtype=int,order='F')
print(ones)


# ## eye ( )




b=np.eye(4,4,-1,dtype=int)
print(b)


# ## identity ( )




c=np.eye(4,4,dtype=int)
print(c)


# ## empty ( )




d=np.empty((6,2),dtype=int)
print(d)


# ## full ( )




e=np.full((3,3),5,dtype=int)
print(e)





## shape, size, reshape

f=np.array([[1,2,3],[4,5,6]])
print(f)





print(f.shape)





print(f.size)





print(f.reshape((3,2)))


# ## ndim
# 




g=np.array([[[3,2,22]]])
print('Dimensions : ',np.ndim(g))


# ## concatenate ( )




h=np.random.randint(100,size=(5,5))
i=np.random.randint(100,size=(5,5))
print("\narray h : \n",h)
print("\narray i : \n",i)
print("\nRow Concatenated \n",np.concatenate((h,i),axis=0)) #row
print("\nColumn Concatenated \n",np.concatenate((h,i),axis=1)) #column


# ## append ( )




print(np.append(h,i))


# # 2. Statistical Functions

# ## amin




j=np.random.randint(50,size=(10,10))
print(j)





print(np.amin(j))


# ## amax




print(np.amax(j))


# ## ptp -> peak to peak




print('row wise\n',np.ptp(j,axis=0))
print('column wise\n',np.ptp(j,axis=1))


# ## percentile




print(np.percentile(j,q=25))


# ## mean 




print(np.mean(j))


# ## meadian




print(np.median(j))


# ## standard deviation




print(np.std(j))


# ## variance




print(np.var(j))


# ## average




k=np.array([12,14,15,13,19,20])
weights=np.array([0.25,0.25,0.125,0.0625,0.0625,0.25])
print(np.average(k,weights=weights))


# # 3. Trigonometric Functions 
# * sin cos tan
# * sinh cosh tanh
# * arcsin arccos arctan 
# * deg2rad rad2deg




print(np.sin(np.pi/2))





print(np.cos(0))





print(np.tan(np.pi/4))





print(np.sinh(np.pi/2))





print(np.cosh(0))





print(np.tanh(np.pi/4))





#inverse of trigonometric functions

print("radian : ",np.arcsin(1))
print("degree : ",np.rad2deg(np.arcsin(1)))





print("radian : ",np.arccos(0.5))
print("degree : ",np.rad2deg(np.arccos(0.5)))





print("radian : ",np.arctan(1))
print("degree : ",np.rad2deg(np.arctan(1)))





deg=45
rad=np.pi

print(f"45 Degree to radian = {np.deg2rad(deg)}")
print(f"pi radian to degree = {np.rad2deg(rad)}")


# # 4. Pandas : Various Operations using iloc




import pandas as pd

data=pd.DataFrame({'Roll No':[1,2,3,4,5,6,7,8,9,10],
                  'Name':['A','B','C','D','E','F','G','H','I','J'],
                   'Marks':[54,34,12,76,98,23,54,87,34,94],
                  'City':['Kolhapur','Mumbai','Pune','Satara','Kolhapur','Mumbai','Pune','Satara','Patan','Karad'],
                  })
print(data)





#fetching all rows and all column

print(data.iloc[:])





#fetching all rows and some columns
print(data.iloc[:,1:3])





#fetching some rows and all columns
print(data.iloc[1:6])





#fetching some rows and some columns
print(data.iloc[1:6,[1,3]])





#fethcing all rows excluding some columns
print(data.iloc[:,[1,3]])


# # 5. Numpy : Indexing and Slicing operations on 2D numpy array
# 




#Create a 2D array

arr2D=np.arange(0,100,4).reshape(5,5)
print(arr2D)





#indexing : print 3rd element from 2nd column
print(arr2D[3][2])





#row slice
print(arr2D[1:3,:])





#column slice
print(arr2D[:,2:4])





#slicing first 3 rows and 4 columns

print(arr2D[:3,:4])


# # 6. Numpy : Indexing and Slicing operations on 3D array




#Create a 3D array

arr3D=np.arange(0,100,2).reshape(2,5,5)
print(arr3D)





#indexing : print 3rd element from 4th column of 0th plane

print(arr3D[0][3][4])





#plane slicing

print(arr3D[1,:,:])





#row slicing
print(arr3D[:,1:4,:])





#column slicing
print(arr3D[:,:,3:5])


# # 7. Handling Null Values using pandas




import pandas as pd

data=pd.read_csv('data.csv')
print(data)





#detecting null values for each column
print(data.isnull().sum())





#detecting not null values
print(data.notnull().sum())





#drop null values (rows with NA values are dropped)
df1=data.dropna()
print(df1)





#fill null values with 0
df2=data.fillna(0)
print(df2)





#fill null values using forward fill
df3=data.fillna(method='ffill')
print(df3)





#fill null values using backward fill
df4=data.fillna(method='bfill')
print(df4)





#replace
df5=data.fillna(0)
df5=df5.replace(to_replace=0,value=160)
print(df5)





#interpolate
df6=data.interpolate(method='linear')
print(df6)


# # 8. Handling Null Values from pandas dataframe
# 




dtf=pd.DataFrame({
    'A':[1,2,3,4,5],
    'B':[2,np.nan,np.nan,8,7],
    'C':[np.nan,6,np.nan,np.nan,2]
})
print(dtf)





#isnull
print(dtf.isnull().sum())





#notnull
print(dtf.notnull().sum())





#dropna
print(dtf.dropna())





#forward fill
print(dtf.fillna(method='ffill'))





#backward fill
print(dtf.fillna(method='bfill'))





#fill with value
print(dtf.fillna(1))





#replace
print(dtf.fillna(0).replace(to_replace=0,value=20))





#interpolate
print(dtf.interpolate())


# # 9. Matplotlib 2D plots 
# * Line plot
# * Scatter plot
# * Bar plot
# * Histogram
# * Stem plot
# * Box plot
# * Pie Chart
# * Stack Plot




import matplotlib.pyplot as plt





#line plot
x=np.linspace(0,2*np.pi)
y=np.sin(x)

plt.plot(x,y,label='Sine Wave / Curve',color='Blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()





#scatterplot
y=np.cos(x)
plt.scatter(x,y,label='Cosine Curve',color='Red')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()





#barplot 

names=np.array(['Abhishek','Atharv','Akash','Amar'])
width=0.2

r3=np.array(range(0,len(names)))
r4=width+r3

marks=np.array([90,53,47,84])
marks2=np.array([92,67,11,34])

plt.figure(figsize=(5,3))
plt.bar(r3,marks,label="Mathematics",color='green',width=0.2, edgecolor='black')
plt.bar(r4,marks2,label="Biology",color='blue',width=0.2, edgecolor='black')
plt.xlabel("Names")
plt.ylabel("Marks")
plt.xticks(r3+0.1, names)
plt.legend()
plt.show()





#stemplot

x = np.linspace(0, 2*np.pi)
y = np.cos(x-np.pi/2)


plt.stem(x,y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cosine Stem Plot')
plt.show()





# Stackplot

days=np.array([1,2,3,4,5,6])
study=np.array([8,9,6,7,7,4])
work=np.array([4,5,4,6,4,5])
sleep=np.array([8,6,7,8,7,9])
extra=np.array([4,4,7,3,6,6])

plt.stackplot(days,study,work,sleep,extra, labels=['days','study','work','sleep','extra'],colors=['blue','green','yellow','red'])
plt.xlabel('Days')
plt.ylabel('Hours ')
plt.title('Weekly Data')
plt.legend()
plt.show()





# boxplot

data1=np.random.normal(100,40,200)
data2=np.random.normal(100,30,400)
data3=np.random.normal(100,10,600)
plt.figure(figsize=(5,3))
plt.boxplot([data1,data2,data3])
plt.show()





# Pie Chart

nutrients=['Protein','Carbohydrates','Vitamins','Minerals','Fats']
percentage=[69.5,13.5,10,5,2]
plt.pie(percentage,labels=nutrients)
plt.title('Muscleblaze Whey Protein Supplement Nutritional Value')
plt.show()





# Histogram

data=np.random.binomial(100,0.6,size=1000)

# Create a histogram with 20 bins
plt.hist(data, bins=20)

# Set the x-axis label
plt.xlabel('X')

# Set the y-axis label
plt.ylabel('Y')

# Set the title of the plot
plt.title('Histogram')

# Display the plot
plt.show()


# # 10. Matplotlib : 3D plots
# * Scatter Plot
# * Line Plot
# * Surface Plot
# * Contour Plot
# * Density Plot




#Line Plot

z=np.linspace(0,np.pi*2)
x=np.sin(z)
y=np.cos(z)

ax=plt.figure().add_subplot(projection='3d')
ax.plot(x,y,z)
ax.set_xlabel('sin x')
ax.set_ylabel('cos x')
ax.set_zlabel('Angle')
plt.show()





#scatter plot

ax=plt.figure().add_subplot(projection='3d')
ax.scatter(x,y,z)
ax.set_xlabel('sin x')
ax.set_ylabel('cos x')
ax.set_zlabel('Angle')
plt.show()





# Surface plot

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)

X, Y = np.meshgrid(X, Y)

Z = np.tan(np.sqrt(np.sin(X)**2 + np.cos(Y)**2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot the surface plot
ax.plot_surface(X, Y, Z)

plt.show()
     





# Density and contour plot
# create data for the density and contour plot
x = np.random.normal(size=1000)
y = np.random.normal(size=1000)
z = (x**2+y**2)/np.pi

# create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# plot the density and contour plot
hist, xedges, yedges = np.histogram2d(x, y, bins=30, density=True)
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
Z = hist.T
ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8) #viridis plasma inferno magma cividis


# set the labels for the axes
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

#show plot
plt.show()





# Contour Plot

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.contour(X, Y, Z, cmap='magma')


ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

plt.show()


# # 11. EDA : One Hot Encoding




remark_data=pd.DataFrame({
    'empid':[1,2,3,4,5,6,7,8,9,10],
    'remark':['nice','great','good','nice','great','good','nice','great','good','nice'],
    'type':['male','female', 'male','female','male','female','male','female','male','female']
    }
)
print(remark_data)





#apply OneHotEncoder for data transformation
from category_encoders import OneHotEncoder

onehot_encoded = OneHotEncoder(cols=['remark','type']).fit(remark_data).transform(remark_data)
print(onehot_encoded)





#dummy encoder

dummy_encoded=pd.get_dummies(remark_data, columns=['remark','type'])
print(dummy_encoded.astype(int))


# # 12. EDA : Pearson Correlation
# 
# ```r = (nΣxy - ΣxΣy) / sqrt((nΣx^2 - (Σx)^2) * (nΣy^2 - (Σy)^2))```
# 
# ```r = (Σ(Xi-mean(x)) * (Yi-mean(Y)))/(sqrt(Σ(Xi-mean(x)^2) * Σ((Yi-mean(Y))^2 ))```




hwdata=pd.DataFrame({
    'Height':[140,152,162,178,163,190,173,167,157,180],
    'Weight':[50,58,59,64,54,79,72,75,53,69]
})
print(hwdata)
     





def correlation(x,y):
    r=np.sum((x-np.mean(x))*(y-np.mean(y)))/np.sqrt(np.sum((x-np.mean(x))**2)*np.sum((y-np.mean(y))**2))
    return r

print(correlation(hwdata['Height'],hwdata['Weight']))





#find correlation coefficient

r=hwdata['Height'].corr(hwdata['Weight'])
print("Pearson's correlation coefficient:", r)

if r>=0.5:
  print("Highly positive Correlation")
elif r>0.1 and r<0.5:
  print("Low positive Correlation")
elif r<0.1 and r>-0.5:
  print("Low Negaive Correlation")
elif r<-0.5 and r>=-1:
  print("Highly Negative Correlation")
else:
  print("No Correlation")
     





corr_matrix=hwdata.corr()
print(corr_matrix)


# # 13. DBSCAN 




import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('dbscan.csv')

# Select the features for clustering
features = ['Age', 'Income', 'Score']

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan.fit_predict(data_scaled)

# Identify the outliers
outliers = data[clusters == -1]

# Remove the outliers from the dataset
data_cleaned = data[clusters != -1]

# Print the outliers
print("Outliers:")
print(outliers)

# Print the cleaned dataset
print("Cleaned Dataset:")
print(data_cleaned)





# Import necessary libraries
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# Generate sample data
X, y = make_blobs(n_samples=1000, centers=3)

# Instantiate DBSCAN object
dbscan = DBSCAN(eps=0.5, min_samples=5)

# Fit the model to the data
dbscan.fit(X)

# Get the cluster labels and number of clusters
labels = dbscan.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Plot the data points colored by their cluster labels
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for i in range(n_clusters):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], s=50, c=colors[i % len(colors)], label='Cluster %d' % i)
plt.legend()
plt.show()


# # 14. Skewness and Kurtosis
# 
# ```skewness = u^3 / (n-1)*sigma^3 ```
# 
# ```kurtosis = u^4 / sigma^4```

# ## Skewness




from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import statistics
     





x=np.array([0,5,10,15,17,20,23,25,30,40,65,70,75,85,87,87,89,95,80,76])
y=np.linspace(1,20,20)
     





print(skew(x))





# Calculate the skewness using formula
skewness = np.sum(pow((x - np.mean(x)),3)) / ((np.size(x) - 1) * pow(np.sqrt(np.sum((x - np.mean(x))**2) / np.size(x)),3))
print("Skewness:", round(skewness,2))
if skewness>0:
  print("Right Skew")
elif skewness<0:
  print("Left Skew")
else:
  print('Normal Distribution')

print(f'Mean={np.mean(x)}\nMedian={np.median(x)}\nMode={statistics.mode(x)}')





plt.plot(x)
plt.xlabel('Marks')
plt.show()


# ## Kurtosis




# Calculate the kurtosis using scipy
kurtosis_val = kurtosis(x)
print("Kurtosis:", kurtosis_val+3)





#calculate Kurtosis using formula
#u4 and sigma4
def kurtosis(arr):
    u4=np.sum(np.power(arr-np.mean(arr),4))
    s4=np.power(np.sqrt(np.sum(np.power(arr-np.mean(arr),2)/arr.size)),4)
    return u4/(arr.size*s4)

print(kurtosis(x))


# # 17. Combining Datasets

# ## Concatenate




a=pd.DataFrame(
    {'A':[10,20,30],
     'B':[40,50,60],
     'C':[70,80,90]}
)
b=pd.DataFrame(
    {'A':[11,22,33],
     'B':[44,55,66],
     'C':[77,88,99]}
)
c=pd.DataFrame(
    {'D':[1,2,3],
     'E':[4,5,6]}
)
print(a)
print(b)
print(c)





#concatenate vertically -> concatenate rows
combined=pd.concat([a,b],axis=0)
combined=combined.reset_index(drop=True)
print(combined)





#concatenate horizontally -> concatenate columns
combined1=pd.concat([a,c],axis=1)
print(combined1)


# ## Merge and Join




df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value1': [1, 2, 3, 4]})
df2 = pd.DataFrame({'key': ['B', 'D', 'E', 'F'], 'value2': [5, 6, 7, 8]})

merged_df = pd.merge(df1, df2, on='key', how='outer')

print(merged_df)





#join
left = pd.DataFrame(
    {"A": ["A0", "A1", "A2", "A3"],
     "B": ["B0", "B1", "B2", "B3"],
     "key": ["K0", "K1", "K0", "K1"],}
)
right = pd.DataFrame({"C": ["C0", "C1"], "D": ["D0", "D1"]}, index=["K0", "K1"])
result = left.join(right, on="key")
print(result)

