## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method
# CODING AND OUTPUT:
```

     import pandas as pd 
     df= pd.read_csv("/content/Encoding Data.csv")
     df

```


  <img width="1489" height="645" alt="image" src="https://github.com/user-attachments/assets/d8126c5c-c88f-475a-9126-c5dac8de8dcb" />

  
```

# ORDINAL ENCODING
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])

```

<img width="1452" height="404" alt="image" src="https://github.com/user-attachments/assets/c6d8a30d-d89b-49fc-bca6-3618d9bc9961" />

```

df['bo2']= e1.fit_transform(df[["ord_2"]])
df

```

<img width="1426" height="522" alt="image" src="https://github.com/user-attachments/assets/ec5f02af-10a4-488b-b1d8-27a681304c6f" />

```

# Label Encoder ( orders in alphabetical order)
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc

```

<img width="1427" height="650" alt="image" src="https://github.com/user-attachments/assets/daedf944-b6ba-4c85-a78d-a58a18b4b360" />


```

# ONE HOT ENCODING
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2= pd.concat([df2,enc],axis=1)
df2

```

<img width="1418" height="705" alt="image" src="https://github.com/user-attachments/assets/f73a84ab-3688-4511-b0ea-180f41055cf8" />


```

pd.get_dummies(df2,columns=["nom_0"])


```
<img width="1398" height="498" alt="image" src="https://github.com/user-attachments/assets/4ac93370-9991-4ce1-9dbb-368d4dfe3b77" />


```
pip install --upgrade category_encoders

```

<img width="1415" height="510" alt="image" src="https://github.com/user-attachments/assets/7b737932-72f8-478a-a01f-69fda8371d21" />


```

# BINARY ENCODER
from category_encoders import BinaryEncoder
df= pd.read_csv("/content/data.csv")
df

```

<img width="1414" height="632" alt="image" src="https://github.com/user-attachments/assets/ba4412f7-4f94-4223-adc7-b2aa0a75e4cf" />


```

be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```

<img width="1417" height="625" alt="image" src="https://github.com/user-attachments/assets/310d5851-5c71-4ad8-b544-d1c983b6c2da" />


```

# MEAN ENCODING
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC

```

<img width="1421" height="725" alt="image" src="https://github.com/user-attachments/assets/14757b67-0636-4e4b-8270-9e5410e143b3" />

```

# FEATURE TRANSFORMATION
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df

```

<img width="1408" height="763" alt="image" src="https://github.com/user-attachments/assets/5defbe27-66d8-4d13-b345-c56cd5bbd413" />

```

df.skew()

```

<img width="1421" height="321" alt="image" src="https://github.com/user-attachments/assets/978650ac-760a-475b-82b1-055351b75559" />

```
# 1. LOG TRANSFORMATION
np.log(df["Highly Positive Skew"])

```

<img width="1416" height="670" alt="image" src="https://github.com/user-attachments/assets/d3d57467-25ee-4f70-855f-46c80542a3a1" />


```

# 2. RECIPROCAL TRANSFORMATION
np.reciprocal(df["Moderate Positive Skew"])

```

<img width="1416" height="648" alt="image" src="https://github.com/user-attachments/assets/fc903d92-5cda-4b9c-a3da-09ed19698960" />

```

# 3. SQUARE ROOT TRANSFORMATION
np.sqrt(df["Highly Positive Skew"])

```

<img width="1421" height="651" alt="image" src="https://github.com/user-attachments/assets/bd5ab5a6-0ece-4d33-8f48-63213c96fa39" />


```

# 4. SQUARE TRANSFORMATION
np.square(df["Highly Positive Skew"])

```

<img width="1414" height="663" alt="image" src="https://github.com/user-attachments/assets/62ff64be-9a59-45f1-bfe2-3c2e00bb5e9c" />


```

# POWER TRANSFORMATIONS
# BOX COX
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df

```

<img width="1415" height="713" alt="image" src="https://github.com/user-attachments/assets/fa5974cc-7781-4049-a967-c2ea6f9e1653" />


```

df.skew()

```

<img width="1404" height="348" alt="image" src="https://github.com/user-attachments/assets/2e4aeaa7-5e3c-459b-9026-72ca4380075b" />

```

# YEO_JOHNSON
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()

```

<img width="1413" height="450" alt="image" src="https://github.com/user-attachments/assets/185334b9-6770-4dc9-9d67-7da2f48309fe" />

```

# QUANTILE TRANSFORMATION
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

```

<img width="1413" height="698" alt="image" src="https://github.com/user-attachments/assets/5c7b5e80-4e5f-41b9-b5b6-d54b7942afaf" />


```

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```

<img width="1422" height="725" alt="image" src="https://github.com/user-attachments/assets/fe926e0a-1725-4a4c-a43e-52768209da48" />


```

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

```

<img width="1401" height="629" alt="image" src="https://github.com/user-attachments/assets/f59859d2-8aa9-4e25-9c14-8353c2f1696a" />


```

 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()

```
<img width="1420" height="712" alt="image" src="https://github.com/user-attachments/assets/5497a5b3-2f43-49bb-bc85-1a9203252a35" />


# RESULT:

Thus the given data, Feature Encoding, Transformation process and save the data to a file  was performed successfully


       
