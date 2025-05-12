# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

 FEATURE SCALING
 
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/c26403c9-c6d4-4f65-9d42-fde63fae30b8)
```
df.head()
```
![image](https://github.com/user-attachments/assets/cf0a41a5-0520-4097-abc6-f08f898e669a)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/43571d09-b1d6-47cd-a85b-aca60ab623c1)
```
max_vals=np.max(np.abs(df[['Height']]))
max_vals1=np.max(np.abs(df[['Weight']]))
print("Height =",max_vals)
print("Weight =",max_vals1)
```
![image](https://github.com/user-attachments/assets/d14c4b67-1bc6-43ae-ab2e-736f26ace936)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![image](https://github.com/user-attachments/assets/fb43e8be-7660-4cab-96b5-0553afaa5e3d)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/1ee797db-4196-4815-864b-9b3991eddbdd)
```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/1ef58e22-51de-4536-8994-96ec0a73fc44)
```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/aaff6dbf-2b9a-4606-a621-26fe0e1526c1)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/fd61b1f0-8820-40ce-bdc1-8c276a354378)

FEATURE SELECTION
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/user-attachments/assets/1264a873-7d4c-4c9f-9d04-1b5d5850b95a)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/874d4ead-8a45-43f3-889c-e2443401079d)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/8f7b6be1-8715-415a-ac91-bd1d8d45b8c1)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/8def4d04-06c2-42d1-9b31-9fb833e68bea)
```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/9651fe05-8984-4d4a-ac3a-7c1e1dd03952)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/b149efe2-5df2-4cd4-b5df-1231232d43b5)
```
data2
```
![image](https://github.com/user-attachments/assets/265d5cef-ea50-4ae4-a32f-47614a739e6c)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/c4116321-cadb-4172-b03f-fb1f43b35cbf)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/d405c1a2-14a6-433c-80be-50891340ac11)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/4f07e69c-b56c-44ab-b432-c669c320e7ff)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/b57e99d3-1601-43b8-8259-ddd422f9a0ba)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/a3128607-7a8f-4102-843b-1a3fc4a28874)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/97206fe8-d966-465b-a39c-b8eb7dc113ae)
```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/0b765d50-fe76-4f93-ac2f-83f8084c34ab)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/5c6d1c49-7b6c-4758-9f48-47e7b2bf90fd)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data= {
    'Feature1' : [1,2,3,4,5],
    'Feature2' : ['A','B','C','A','B'],
    'Feature3' : [0,1,1,0,1],
    'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)
X=df[['Feature1','Feature3']]
y=df['Target']
selector = SelectKBest(score_func= mutual_info_classif, k=1)
X_new=selector.fit_transform(X,y)
selected_feature_indices = selector.get_support(indices=True)
selected_features=X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/7c25174d-afaf-4857-9e26-8e8e3c4f82f2)
CONTINGENCY TABLE:
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/f3776df9-a4bb-4d0c-8e2b-995a8b50ed7f)
```
contingency_table = pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/b06e60b5-f6d2-4966-bbc2-0ef9417df76e)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/user-attachments/assets/e4153f05-853b-4494-8628-a824f842c470)

# RESULT:
 Thus The feature encoding,Feature Scaling from the given Data set has identified successfully
