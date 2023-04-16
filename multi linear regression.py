#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('house_pred.csv')
test = pd.read_csv('test.csv')


# In[3]:


data.info()


# In[4]:


columns = ['Alley' , 'FireplaceQu' , 'PoolQC' , 'Fence' , 'MiscFeature']


# In[5]:


data=data.drop(columns , axis = 1)
test=test.drop(columns , axis = 1)


# In[6]:


data=data.dropna(subset = ['Electrical' , 'MasVnrType' , 'MasVnrArea'])


# In[7]:


category = data.select_dtypes(exclude = ['int' , 'float']).columns


# In[8]:


category


# In[9]:


columns=['BsmtQual',
'BsmtCond',
'BsmtExposure',
'BsmtFinType1' ,
'BsmtFinType2' ,
'GarageType' ,
'GarageFinish' ,
'GarageQual' ,
'GarageCond' ]


# In[10]:


map(lambda x: data[x].fillna(data[x].mode()[0] , inplace = True),columns)


# In[11]:


data.info()


# In[12]:


data['LotFrontage'].fillna(data['LotFrontage'].median() , inplace = True)
data['GarageYrBlt'].fillna(data['GarageYrBlt'].median() , inplace = True)


# In[13]:


groups = data.groupby( list(category))


# In[14]:


data.info()


# In[15]:


data=data.dropna()


# In[16]:


data.info()


# In[17]:


columns = ['OpenPorchSF' , 'EnclosedPorch' , '3SsnPorch' , 'ScreenPorch' ]


# In[18]:


data.drop(columns , inplace = True , axis = 1)
test.drop(columns , inplace = True , axis = 1)


# In[19]:


def IQR(column , dataset = data ):
    df = dataset[column]
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3 - q1
    lower = q1 - (1.5 * IQR)
    upper = q3 + (1.5 * IQR)
    return lower , upper


# In[20]:


numerical = test.select_dtypes(exclude = 'object').columns
category = test.select_dtypes(exclude = ['int' , 'float']).columns


# In[21]:


test.info()


# In[22]:


dummies_data =  pd.get_dummies(data[category] , drop_first = False)
dummies_test =  pd.get_dummies(test[category] , drop_first = False)


# In[23]:


dummies_data


# In[24]:


dummies_test


# In[25]:


l = []
for i in dummies_data.columns:
    if i not in dummies_test.columns:
        print(i)
        l+=[i]


# In[26]:


dummies_data.drop(l , inplace = True , axis = 1)


# In[27]:


dummies = pd.get_dummies(data[category] , drop_first = False)


# In[28]:


data.drop(category , axis = 1 , inplace = True)


# In[29]:


data.head()


# In[30]:


data = data.join(dummies_data)


# In[31]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()


# In[32]:


x = data.drop('SalePrice' , axis = 1)
y = data['SalePrice']


# In[33]:


col = x.columns


# In[34]:


data.info()


# In[35]:


vif["feature"] = list(col)
vif["Variance Inflation "] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]


# In[36]:


vif_filtered = vif[vif['Variance Inflation '] < 3]


# In[37]:


vif_filtered


# In[38]:


from sklearn.linear_model import LinearRegression


# In[39]:


linearreg = LinearRegression()


# In[40]:


x = data[vif_filtered['feature']]
y = data['SalePrice']


# In[41]:


y


# In[42]:


x


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.25 , random_state = 42)


# In[45]:


linearreg.fit(x_train , y_train )


# In[46]:


y_pred = linearreg.predict(x_test)


# In[47]:


from sklearn.metrics import r2_score


# In[48]:


r2_score(y_pred , y_test)


# In[49]:


linearreg.fit(x , y)


# In[50]:


test


# In[51]:


x.columns


# In[52]:


category = test.select_dtypes(exclude = ['int' , 'float']).columns


# In[53]:


category


# In[54]:


dummies = pd.get_dummies(test[category])


# In[55]:


test.info()


# In[56]:


MSE = np.square(np.subtract(y_test , y_pred)).mean() 
RMSE = math.sqrt(MSE)
RMSE


# In[ ]:




