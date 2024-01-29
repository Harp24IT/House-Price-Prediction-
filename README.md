# House-Price-Prediction-



Skip to main content
HOUSEPRICEPREDCTION.ipynb
HOUSEPRICEPREDCTION.ipynb_Notebook unstarred
All changes saved
[2]
0s

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
%matplotlib inline
## Display all the columns of the dataframe

pd.pandas.set_option('display.max_columns',None)
[3]
1s
House=pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/House%20Prices.csv')
[4]
0s
print(House)
output
          ID             Date     Price  Bedrooms  Bathrooms  Sqft_living  \
0          1  20140916T000000  280000.0         6       3.00         2400   
1          2  20150422T000000  300000.0         6       3.00         2400   
2          3  20140508T000000  647500.0         4       1.75         2060   
3          4  20140811T000000  400000.0         3       1.00         1460   
4          5  20150401T000000  235000.0         3       1.00         1430   
...      ...              ...       ...       ...        ...          ...   
21608  21609  20140725T000000  365000.0         5       2.00         1600   
21609  21610  20150311T000000  380000.0         2       1.00         1040   
21610  21611  20140624T000000  339000.0         3       1.00         1100   
21611  21612  20140703T000000  399900.0         2       1.75         1410   
21612  21613  20141030T000000  268950.0         3       1.00         1320   

       Sqft_lot  Floors  Waterfront  View  Condition  Grade  Sqft_above  \
0          9373     2.0           0     0          3      7        2400   
1          9373     2.0           0     0          3      7        2400   
2         26036     1.0           0     0          4      8        1160   
3         43000     1.0           0     0          3      7        1460   
4          7599     1.5           0     0          4      6        1010   
...         ...     ...         ...   ...        ...    ...         ...   
21608      4168     1.5           0     0          3      7        1600   
21609      7372     1.0           0     0          5      7         840   
21610      4128     1.0           0     0          4      7         720   
21611      1005     1.5           0     0          3      9         900   
21612      8100     1.0           0     0          3      6         880   

       Sqft_basement  Yr_built  Yr_renovated  zipcode      Lat     Long  \
0                  0      1991             0    98002  47.3262 -122.214   
1                  0      1991             0    98002  47.3262 -122.214   
2                900      1947             0    98166  47.4444 -122.351   
3                  0      1952             0    98166  47.4434 -122.347   
4                420      1930             0    98168  47.4783 -122.265   
...              ...       ...           ...      ...      ...      ...   
21608              0      1927             0    98126  47.5297 -122.381   
21609            200      1939             0    98126  47.5285 -122.378   
21610            380      1942             0    98126  47.5296 -122.379   
21611            510      2011             0    98027  47.5446 -122.018   
21612            440      1943             0    98166  47.4697 -122.351   

       Sqft_living15  Sqft_lot15  
0               2060        7316  
1               2060        7316  
2               2590       21891  
3               2250       20023  
4               1290       10320  
...              ...         ...  
21608           1190        4168  
21609           1930        5150  
21610           1510        4538  
21611           1440        1188  
21612           1000        8100  

[21613 rows x 21 columns]
[5]
0s
House.head()

output

[7]
0s
House.shape
output
(21613, 21)
[8]
0s
obj = (House.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))
 
int_ = (House.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:",len(num_cols))
 
fl = (House.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:",len(fl_cols))
output
Categorical variables: 1
Integer variables: 15
Float variables: 5
[9]
4s

plt.figure(figsize=(12, 6))
sns.heatmap(House.corr(),
            cmap = 'BrBG',
            fmt = '.2f',
            linewidths = 2,
            annot = True)
output

[10]
0s

unique_values = []
for col in object_cols:
  unique_values.append(House[col].unique().size)
plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols,y=unique_values)
output

[ ]
House.groupby('Condition')['Price'].median().plot()
plt.xlabel('Condition')
plt.ylabel('Median House Price')
plt.title(" Price vs Condition")
output

[ ]
## Here we will compare the difference between All years feature with SalePrice

for feature in year_feature:
    if feature!='Condition':
        data=House.copy()
        ## We will capture the difference between year variable and year the house was sold for
        data[feature]=data['Condition']-data[feature]

        plt.scatter(data[feature],data['Price'])
        plt.xlabel(feature)
        plt.ylabel('Price')
        plt.show()

output

[ ]
House['Bedrooms'].value_counts().plot(kind='bar')
plt.title('number of Bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine
output

[ ]
plt.figure(figsize=(10,10))
sns.jointplot(x=House.Price.values, y=House.Condition.values, size=10)
plt.ylabel('Price', fontsize=12)
plt.xlabel('Condition', fontsize=12)
plt.show()
plt1=plt()
sns.despine

output

[ ]
plt.scatter(House.Price,House.Sqft_living15)
plt.title("Price vs Sqft_living")
output

[ ]
plt.scatter(House.Price,House.Floors)
plt.xlabel("Price")
plt.ylabel('Floors')
plt.title("Floors vs Price")
output

[ ]
train1 = House.drop(['ID', 'Price'],axis=1)
train1.head()
output

[ ]
House.Floors.value_counts().plot(kind='bar')
output

[ ]
plt.scatter(House.Floors,House.Price)
output

[ ]
plt.scatter(House.Condition,House.Price)
output

[11]
5s
plt.figure(figsize=(18, 36))
plt.title('Categorical Features: Distribution')
plt.xticks(rotation=90)
index = 1
 
for col in object_cols:
    y = House[col].value_counts()
    plt.subplot(11, 4, index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index += 1
output

[13]
0s
House.drop(['ID'],
             axis=1,
             inplace=True)
[15]
0s
House.isnull().sum()
output
Date             0
Price            0
Bedrooms         0
Bathrooms        0
Sqft_living      0
Sqft_lot         0
Floors           0
Waterfront       0
View             0
Condition        0
Grade            0
Sqft_above       0
Sqft_basement    0
Yr_built         0
Yr_renovated     0
zipcode          0
Lat              0
Long             0
Sqft_living15    0
Sqft_lot15       0
dtype: int64
[17]
0s

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
 
X = House.drop(['Price'], axis=1)
Y = House['Price']
 
# Split the training set into 
# training and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=0)
[20]

[ ]

Colab paid products - Cancel contracts here
done
Connected to Python 3 Google Compute Engine backend
Runtime disconnected
