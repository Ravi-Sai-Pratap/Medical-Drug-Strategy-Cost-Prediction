#!/usr/bin/env python
# coding: utf-8

# # Extent of loss of potential sales in 2015

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("data.csv")


# In[3]:


df.head()


# Dropping unwanted features which had no impact on dependent variable sales

# In[4]:


df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.head()


# In[5]:


df1=df.copy()


# Furthur extension of month column inorder to get much more insights of sales for each of individual year/month/day features

# In[6]:


df1['year'] = pd.DatetimeIndex(df['month']).year
df1['month'] = pd.DatetimeIndex(df['month']).month
df1['day'] = pd.DatetimeIndex(df['month']).day
df1.head()


# Obtaining all the features data within the year 2015 for starting 6 months so can access the data before the new compBrand is introduced

# In[7]:


before_mid_2015 = df1[df1.year == 2015 ][df1.month<6][['year','qty','accType','strategy1','strategy2','strategy3','salesVisit1','salesVisit2','salesVisit3','salesVisit4','salesVisit5','compBrand','sales']]
before_mid_2015.head()


# In[8]:


before_mid_2015.shape


# Obtaining all the features data within the year 2015 for after 6 months so can access the data after the new compBrand is introduced

# In[9]:


after_mid_2015 = df1[df1.year == 2015 ][df1.month>6][['year','qty','accType','strategy1','strategy2','strategy3','salesVisit1','salesVisit2','salesVisit3','salesVisit4','salesVisit5','compBrand','sales']]
after_mid_2015.head()


# In[10]:


after_mid_2015.shape


# In[11]:


import seaborn as sns


# Plotting all the Strategies with respect to Sales

# In[12]:


before_mid_2015.groupby('qty')['sales'].median().plot()
plt.xlabel('Quantity')
plt.ylabel('Sales')
plt.title("Sales Prediction by Quantity")


# In[13]:


before_mid_2015.groupby('strategy1')['sales'].median().plot()
plt.xlabel('Strategy1')
plt.ylabel('Sales')
plt.title("Sales Prediction by Strategy1")


# In[14]:


before_mid_2015.groupby('strategy2')['sales'].median().plot()
plt.xlabel('strategy2')
plt.ylabel('Sales')
plt.title("Sales Prediction by strategy2")


# Strategy3 had no impact on Sales

# In[15]:


before_mid_2015.groupby('strategy3')['sales'].median().plot()
plt.xlabel('strategy3')
plt.ylabel('Sales')
plt.title("Sales Prediction by strategy3")


# In[16]:


before_mid_2015['strategy3'].unique()


# In[17]:


after_mid_2015.groupby('qty')['sales'].median().plot()
plt.xlabel('Quantity')
plt.ylabel('Sales')
plt.title("Sales Prediction by Quantity")


# In[18]:


after_mid_2015.groupby('strategy1')['sales'].median().plot()
plt.xlabel('strategy1')
plt.ylabel('Sales')
plt.title("Sales Prediction by strategy1")


# In[19]:


after_mid_2015.groupby('strategy2')['sales'].median().plot()
plt.xlabel('strategy2')
plt.ylabel('Sales')
plt.title("Sales Prediction by strategy2")


# In[20]:


after_mid_2015.groupby('strategy3')['sales'].median().plot()
plt.xlabel('strategy3')
plt.ylabel('Sales')
plt.title("Sales Prediction by strategy3")


# In[21]:


import seaborn as sns
plt.figure(figsize=(10,10))
ax = sns.boxplot(data=before_mid_2015, x='accType',y='sales',palette='plasma')


# In[22]:


import seaborn as sns
plt.figure(figsize=(10,10))
ax = sns.boxplot(data=after_mid_2015, x='accType',y='sales',palette='plasma')


# In[23]:


before_mid_2015.head()


# Dropping few features in order to avoid multi collinearlity for better model accuracy

# In[24]:


before_mid_2015.drop(['year','accType','compBrand'],axis=1,inplace=True)
after_mid_2015.drop(['year','accType','compBrand'],axis=1,inplace=True)


# In[25]:


before_mid_2015.head()


# In[26]:


after_mid_2015.head()


# In[27]:


before_mid_2015['strategy3'].unique()


# In[28]:


after_mid_2015['strategy3'].unique()


# # Prediction of all the Sales in the year 2015 before the competitor drug entrance 

# Dependent feature Sales is stored in variable Y

# In[29]:


y=before_mid_2015['sales']
y


# All independent features are stored in variable X

# In[30]:


X=before_mid_2015.copy()
X.drop(['sales'],axis=1,inplace=True)
X.head()


# In[31]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# In[33]:


X_train.shape


# In[34]:


y_test.shape


# In[35]:


X_test.shape


# In[36]:


y_train.shape


# In[37]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[38]:


pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('lr',LinearRegression())
])


# In[39]:


pipe.fit(X_train,y_train)


# In[40]:


lr_pred=pipe.predict(X_test)


# In[41]:


pd.DataFrame({'original test set':y_test, 'predictions': lr_pred})


# In[42]:


from sklearn import metrics
print('r2:', np.sqrt(metrics.r2_score(y_test, lr_pred)))


# In[43]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lr_pred)))


# In[44]:


plt.scatter(y_test,lr_pred)


# In[45]:


from sklearn.model_selection import cross_val_score,KFold
cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[46]:


import scikitplot as skplt
skplt.estimators.plot_learning_curve(lr,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='LinearRegression')


# In[47]:


from sklearn.linear_model import Lasso
lass=Lasso(alpha=1.0)
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('lass',Lasso(alpha=1.0))
])


# In[48]:


pipe.fit(X_train,y_train)


# In[49]:


lasso_pred=pipe.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': lasso_pred})


# In[50]:


print('r2:', np.sqrt(metrics.r2_score(y_test, lasso_pred)))


# In[51]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lasso_pred)))


# In[52]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[53]:


skplt.estimators.plot_learning_curve(lass,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='Lasso')


# In[54]:


import xgboost as xgb
xg=Lasso(alpha=1.0)
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('xgb',xgb.XGBRegressor())
])


# In[55]:


pipe.fit(X_train,y_train)


# In[56]:


xgb_pred=pipe.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': xgb_pred})


# In[57]:


print('r2:', np.sqrt(metrics.r2_score(y_test, xgb_pred)))


# In[58]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, xgb_pred)))


# In[59]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[60]:


skplt.estimators.plot_learning_curve(xg,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='XGB')


# In[61]:


from sklearn.svm import SVR
sv=SVR()
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('svr',SVR(kernel='rbf'))
])


# In[62]:


pipe.fit(X_train,y_train)


# In[63]:


svr_pred=pipe.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': svr_pred})


# In[64]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svr_pred)))


# In[65]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[66]:


skplt.estimators.plot_learning_curve(sv,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='SVR')


# In[104]:


from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('dtr',DecisionTreeRegressor())
])


# In[105]:


pipe.fit(X_train,y_train)


# In[106]:


dtr_pred=pipe.predict(X_test)


# In[107]:


pd.DataFrame({'original test set':y_test, 'predictions': dtr_pred})


# In[108]:


print('r2:', np.sqrt(metrics.r2_score(y_test, dtr_pred)))


# In[109]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dtr_pred)))


# In[110]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[111]:


skplt.estimators.plot_learning_curve(dtr,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='Decision Tree')


# In[112]:


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor()
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('rfr',RandomForestRegressor())
])


# In[113]:


pipe.fit(X_train,y_train)


# In[114]:


rfr_pred=pipe.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': rfr_pred})


# In[115]:


print('r2:', np.sqrt(metrics.r2_score(y_test, rfr_pred)))


# In[116]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfr_pred)))


# In[117]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[118]:


plt.scatter(y_test,rfr_pred)


# In[119]:


skplt.estimators.plot_learning_curve(rfr,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='Random Forest')


# In[120]:


from sklearn.model_selection import GridSearchCV
param_grid = {  'bootstrap': [True,False], 'max_depth': [5, 10, 15], 'n_estimators': [5, 6, 7, 8, 9, 10, 11, 12, 13, 15],'min_samples_split' : [2, 5, 10, 15, 100],'min_samples_leaf' : [1, 2, 5, 10]}
cv=KFold(n_splits=10,random_state=1,shuffle=True)
search = GridSearchCV(estimator = rfr, param_grid = param_grid,cv = cv,scoring='neg_mean_squared_error', n_jobs = -1, verbose = 0, return_train_score=True)
search.fit(X_train,y_train)


# In[121]:


print(search.best_params_)


# In[122]:


print(search.best_score_)


# In[123]:


rfr_search=search.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': rfr_search})


# In[124]:


print('r2:', np.sqrt(metrics.r2_score(y_test, rfr_search)))


# In[125]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfr_search)))


# In[126]:


params = {'alpha': [1e-10,1e-6,1e-2,3,5,7,15,100,200,300,500,700]} # It will check from 1e-08 to 1e+08
lasso = Lasso()
cv=KFold(n_splits=10,random_state=1,shuffle=True)
lasso_model = GridSearchCV(lasso, params, cv = cv,scoring='neg_mean_squared_error')
lasso_model.fit(X_train, y_train)
print(lasso_model.best_params_)
print(lasso_model.best_score_)


# In[127]:


lass_modelpred=lasso_model.predict(X_test)
result=pd.DataFrame({'original test set':y_test, 'predictions': lass_modelpred})
result


# In[128]:


plt.scatter(y_test,lass_modelpred)


# In[129]:


print('r2:', np.sqrt(metrics.r2_score(y_test, lass_modelpred)))


# In[130]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lass_modelpred)))


# In[131]:


lasso_model.best_params_


# In[132]:


lasso_model.best_score_


# In[133]:


xgb1 = xgb.XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}
cv=KFold(n_splits=10,random_state=1,shuffle=True)
xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = cv,
                        n_jobs = -1,
                        verbose=True,scoring='neg_mean_squared_error')

xgb_grid.fit(X_train,y_train)


# In[134]:


print(xgb_grid.best_score_)
print(xgb_grid.best_params_)


# In[135]:


xgb_gridpred=xgb_grid.predict(X_test)


# In[136]:


pd.DataFrame({'original test set':y_test, 'predictions': xgb_gridpred})


# In[137]:


print('r2:', np.sqrt(metrics.r2_score(y_test, xgb_gridpred)))


# In[138]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, xgb_gridpred)))


# Out of all regression machine learning algorithms Lasso Regressor had best r2 score for strategy3 so we consider it as best performed algorithm and stored it in a data frame named "result"

# Calculating the mean of the final prediction of sales inorder to obtain the percentage of sales before mid 2015

# In[139]:


mean_of_sales_before_mid_2015=result['predictions'].mean()
mean_of_sales_before_mid_2015


# Calculating the individual percentages for the original and predictions of sales before mid 2015

# In[140]:


result['Percenatge_of_sales_before_mid_2015']=(result['predictions']/result['predictions'].sum())*100
result


# Calculating the average/mean value of all the final percenatge prediction of sales for Strategy1

# In[141]:


Percenatge_of_sales_before_mid_2015=result['Percenatge_of_sales_before_mid_2015'].mean()
Percenatge_of_sales_before_mid_2015


# # Prediction of all the Sales in the year 2015 after the competitor drug entrance 

# In[142]:


y=after_mid_2015['sales']
y


# In[143]:


X=after_mid_2015.copy()
X.drop(['sales'],axis=1,inplace=True)
X.head()


# In[144]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[145]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# In[146]:


X_train.shape


# In[147]:


y_test.shape


# In[148]:


X_test.shape


# In[149]:


y_train.shape


# In[150]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[151]:


pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('lr',LinearRegression())
])


# In[152]:


pipe.fit(X_train,y_train)


# In[153]:


lr_pred=pipe.predict(X_test)


# In[154]:


pd.DataFrame({'original test set':y_test, 'predictions': lr_pred})


# In[155]:


from sklearn import metrics
print('r2:', np.sqrt(metrics.r2_score(y_test, lr_pred)))


# In[156]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lr_pred)))


# In[157]:


plt.scatter(y_test,lr_pred)


# In[158]:


from sklearn.model_selection import cross_val_score,KFold
cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[159]:


import scikitplot as skplt
skplt.estimators.plot_learning_curve(lr,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='LinearRegression')


# In[160]:


from sklearn.linear_model import Lasso
lass=Lasso(alpha=1.0)
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('lass',Lasso(alpha=1.0))
])


# In[161]:


pipe.fit(X_train,y_train)


# In[162]:


lasso_pred=pipe.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': lasso_pred})


# In[163]:


print('r2:', np.sqrt(metrics.r2_score(y_test, lasso_pred)))


# In[164]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lasso_pred)))


# In[165]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[166]:


skplt.estimators.plot_learning_curve(lass,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='Lasso')


# In[167]:


import xgboost as xgb
xg=Lasso(alpha=1.0)
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('xgb',xgb.XGBRegressor())
])


# In[168]:


pipe.fit(X_train,y_train)


# In[169]:


xgb_pred=pipe.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': xgb_pred})


# In[170]:


print('r2:', np.sqrt(metrics.r2_score(y_test, xgb_pred)))


# In[171]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, xgb_pred)))


# In[172]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[173]:


skplt.estimators.plot_learning_curve(xg,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='XGB')


# In[174]:


from sklearn.svm import SVR
sv=SVR()
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('svr',SVR(kernel='rbf'))
])


# In[175]:


pipe.fit(X_train,y_train)


# In[176]:


svr_pred=pipe.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': svr_pred})


# In[177]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svr_pred)))


# In[178]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[179]:


skplt.estimators.plot_learning_curve(sv,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='SVR')


# In[180]:


from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('dtr',DecisionTreeRegressor())
])


# In[181]:


pipe.fit(X_train,y_train)


# In[182]:


dtr_pred=pipe.predict(X_test)


# In[183]:


pd.DataFrame({'original test set':y_test, 'predictions': dtr_pred})


# In[184]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dtr_pred)))


# In[185]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[186]:


skplt.estimators.plot_learning_curve(dtr,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='Decision Tree')


# In[187]:


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor()
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('rfr',RandomForestRegressor())
])


# In[188]:


pipe.fit(X_train,y_train)


# In[189]:


rfr_pred=pipe.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': rfr_pred})


# In[190]:


print('r2:', np.sqrt(metrics.r2_score(y_test, rfr_pred)))


# In[191]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfr_pred)))


# In[192]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[193]:


plt.scatter(y_test,rfr_pred)


# In[194]:


skplt.estimators.plot_learning_curve(rfr,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='Random Forest')


# In[195]:


from sklearn.model_selection import GridSearchCV
param_grid = {  'bootstrap': [True,False], 'max_depth': [5, 10, 15], 'n_estimators': [5, 6, 7, 8, 9, 10, 11, 12, 13, 15],'min_samples_split' : [2, 5, 10, 15, 100],'min_samples_leaf' : [1, 2, 5, 10]}
cv=KFold(n_splits=10,random_state=1,shuffle=True)
search = GridSearchCV(estimator = rfr, param_grid = param_grid,cv = cv,scoring='neg_mean_squared_error', n_jobs = -1, verbose = 0, return_train_score=True)
search.fit(X_train,y_train)


# In[196]:


print(search.best_params_)


# In[197]:


print(search.best_score_)


# In[198]:


rfr_search=search.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': rfr_search})


# In[199]:


print('r2:', np.sqrt(metrics.r2_score(y_test, rfr_search)))


# In[200]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfr_search)))


# In[201]:


params = {'alpha': [1e-10,1e-6,1e-2,3,5,7,15,100,200,300,500,700]} # It will check from 1e-08 to 1e+08
lasso = Lasso()
cv=KFold(n_splits=10,random_state=1,shuffle=True)
lasso_model = GridSearchCV(lasso, params, cv = cv,scoring='neg_mean_squared_error')
lasso_model.fit(X_train, y_train)
print(lasso_model.best_params_)
print(lasso_model.best_score_)


# In[214]:


lass_modelpred=lasso_model.predict(X_test)
result=pd.DataFrame({'original test set':y_test, 'predictions': lass_modelpred})
result


# In[203]:


plt.scatter(y_test,lass_modelpred)


# In[204]:


print('r2:', np.sqrt(metrics.r2_score(y_test, lass_modelpred)))


# In[205]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lass_modelpred)))


# In[206]:


lasso_model.best_params_


# In[207]:


lasso_model.best_score_


# In[208]:


xgb1 = xgb.XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}
cv=KFold(n_splits=10,random_state=1,shuffle=True)
xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = cv,
                        n_jobs = -1,
                        verbose=True,scoring='neg_mean_squared_error')

xgb_grid.fit(X_train,y_train)


# In[209]:


print(xgb_grid.best_score_)
print(xgb_grid.best_params_)


# In[210]:


xgb_gridpred=xgb_grid.predict(X_test)


# In[211]:


pd.DataFrame({'original test set':y_test, 'predictions': xgb_gridpred})


# In[212]:


print('r2:', np.sqrt(metrics.r2_score(y_test, xgb_gridpred)))


# In[213]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, xgb_gridpred)))


# Out of all regression machine learning algorithms Lasso Regressor had best r2 score for strategy3 so we consider it as best performed algorithm and stored it in a data frame named "result"

# Calculating the mean of the final prediction of sales inorder to obtain the percentage of sales

# In[215]:


mean_of_sales_after_mid_2015=result['predictions'].mean()
mean_of_sales_after_mid_2015


# Calculating the individual percentages for the original and predictions of sales after mid 2015

# In[216]:


result['Percenatge_of_sales_after_mid_2015']=(result['predictions']/result['predictions'].sum())*100
result


# Calculating the average/mean value of all the final percenatge prediction of sales for Strategy1

# In[217]:


Percenatge_of_sales_after_mid_2015=result['Percenatge_of_sales_after_mid_2015'].mean()
Percenatge_of_sales_after_mid_2015


# Calculating the mean of the final prediction of sales inorder to obtain the percentage of sales after mid 2015.

# In[220]:


mean_of_sales_after_mid_2015


# In[221]:


mean_of_sales_before_mid_2015


# The extent of loss of potential sales due to a new competitor drug entrance in mid 2015

# In[218]:


Percenatge_of_sales_before_mid_2015


# In[219]:


Percenatge_of_sales_after_mid_2015


# Clearly the percentage of Sales of new drug is more than 52% more than NZT-48 after mid 2015. So there is a huge impact of sales on NZT-48 due to new competitor drug in market in 2015.
