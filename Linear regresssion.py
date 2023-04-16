#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


# In[44]:


data = pd.read_csv("data1.csv")


# In[45]:


x = data["x"]
y = data["y"]
x_train,x_test = x[:100],x[100:]
y_train,y_test = y[:100],y[100:]


# In[46]:


data


# In[47]:


# intercept = ((sum(ytrain)*(xtrain@xtrain.T) - (sum(xtrain)*(xtrain@ytrain.T)))/((xtrain@xtrain.T)-sum(xtrain)**2))
# incorrect


# In[48]:


slope = (len(x_train)*((x_train@y_train.T))-(sum(x_train)*sum(y_train)))/((len(x_train)*(x_train@x_train.T))-sum(x_train)**2)


# In[49]:


slope


# In[50]:


intercept = np.mean(y_train) - slope*np.mean(x_train)


# In[51]:


intercept


# In[52]:


x_test.shape


# In[53]:


mat=[intercept,slope]


# In[54]:


y_pred = slope*x_test+intercept


# In[55]:


len(y_pred)


# In[56]:


plt.scatter( x_test, y_test, color = 'blue' )
      
plt.plot( x_test, y_pred , color = 'orange' )


# In[57]:


mae = sum(np.abs(y_test - y_pred))/20


# In[58]:


mae


# In[59]:


# y_test, y_pred


# In[60]:


rmse=math.sqrt(np.square(np.subtract(y_test,y_pred)).mean())


# In[61]:


rmse


# In[ ]:




