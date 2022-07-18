#!/usr/bin/env python
# coding: utf-8

# # Prediction of Sales for all strategy1,strategy2,strategy3

# In[317]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[318]:


df=pd.read_csv("data.csv")


# In[319]:


df.head()


# In[320]:


df.shape


# In[321]:


df.info()


# Check for the existence of null values

# In[322]:


df.isnull().sum()


# Drop the columns which does not describe much about the sales 

# In[323]:


df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.head()


# Check for Cardinality of the features existed in the data set

# In[324]:


df['accID'].nunique()


# In[325]:


df['accID'].unique()


# In[326]:


df['accType'].nunique()


# In[327]:


df['accType'].unique()


# In[328]:


df1=df.copy()


# Furthur extension of month column inorder to get much more insights of sales for each of individual year/month/day features

# In[329]:


df1['year'] = pd.DatetimeIndex(df['month']).year
df1['month'] = pd.DatetimeIndex(df['month']).month
df1['day'] = pd.DatetimeIndex(df['month']).day
df1.head()


# In[330]:


df1.shape


# In[331]:


categorical=[var for var in df.columns if df1[var].dtype=='O']
print('There are {} categorical variables'.format(len(categorical)))
df1[categorical].head()


# In[332]:


#numerical variables
numerical=[var for var in df.columns if df1[var].dtype!='O']
print('There are {} numerical variables'.format(len(numerical)))
df1[numerical].head(5)


# In[333]:


continuous=[var for var in numerical if var not in['sales']]
print("Continuous feature Count {}".format(len(continuous)))
continuous
df1[continuous].head(5)


# In[334]:


import scipy.stats as stats


# Plotting various kinds of plots for better visualization of all continuos features

# In[335]:


import seaborn as sns
def dig_plot(d,var):
    plt.figure(figsize=(10,6))
    
    #hist
    plt.subplot(1,3,1)
    sns.distplot(d[var],bins=30)
    plt.title("Histogram")
    
    #Q-Q plot
    plt.subplot(1,3,2)
    stats.probplot(d[var],dist="norm",plot=plt)
    plt.ylabel("RM Quantiles")
    
    #box plot
    plt.subplot(1,3,3)
    sns.boxplot(y=d[var])
    plt.title("Boxplot")
for var in continuous:
    dig_plot(df1,var)


# Plotting individual featutres with respect to Sales for better information insights

# We observe a sudden rise in sales from the year 2014

# In[336]:


df1.groupby('year')['sales'].median().plot()
plt.xlabel('Year')
plt.ylabel('Sales')
plt.title("Sales Prediction by Year")


# The middle months are having more rise in sales than the start and end months of the year

# In[337]:


df1.groupby('month')['sales'].median().plot()
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title("Sales Prediction by Month")


# In[338]:


df1['day'].nunique()


# In[339]:


df1['month'].nunique()


# In[340]:


df1['year'].nunique()


# As the cardinality for feature 'day' is one we tend to drop the feature

# In[341]:


df1.drop(['day'],axis=1,inplace=True)
df1.head()


# In[342]:


df1.shape


# As the acc size increases the sales are increasing

# In[343]:


df1.groupby('accSize')['sales'].median().plot()
plt.xlabel('accSize')
plt.ylabel('Sales')
plt.title("Sales Prediction by accSize")


# Adequate rise in the Sales as the accTargets increases

# In[344]:


df1.groupby('accTargets')['sales'].median().plot()
plt.xlabel('accTargets')
plt.ylabel('Sales')
plt.title("Sales Prediction by accTargets")


# Sales have been on the decreasing part at the end as per the Strategy1 

# As there was only one district in the dataset there is nothing much we can use it to predict the Sales

# In[345]:


df1.groupby('district')['sales'].median().plot()
plt.xlabel('district')
plt.ylabel('Sales')
plt.title("Sales Prediction by district")


# In[346]:


df1['district'].nunique()


# The quantity had an uniform impact on Sales

# In[347]:


df1.groupby('qty')['sales'].median().plot()
plt.xlabel('Quantity')
plt.ylabel('Sales')
plt.title("Sales Prediction by Quantity")


# In[348]:


df1.groupby('strategy1')['sales'].median().plot()
plt.xlabel('Strategy1')
plt.ylabel('Sales')
plt.title("Sales Prediction by Strategy1")


# Sales have been on the increasing part at the end as per the Strategy2

# In[349]:


df1.groupby('strategy2')['sales'].median().plot()
plt.xlabel('Strategy2')
plt.ylabel('Sales')
plt.title("Sales Prediction by Strategy2")


# Sales have been on the decreasing part at the end as per the Strategy3

# In[350]:


df1.groupby('strategy3')['sales'].median().plot()
plt.xlabel('Strategy3')
plt.ylabel('Sales')
plt.title("Sales Prediction by Strategy3")


# Sales have been on the increasing part at the end as per the salesVisit1

# In[351]:


df1.groupby('salesVisit1')['sales'].median().plot()
plt.xlabel('salesVisit1')
plt.ylabel('Sales')
plt.title("Sales impacted by salesVisit1")


# Sales have been on the increasing part at the end as per the salesVisit2

# In[352]:


df1.groupby('salesVisit2')['sales'].median().plot()
plt.xlabel('salesVisit2')
plt.ylabel('Sales')
plt.title("Sales impacted by salesVisit2")


# Sales have been on the decreasing part at the end as per the salesVisit3

# In[353]:


df1.groupby('salesVisit3')['sales'].median().plot()
plt.xlabel('salesVisit3')
plt.ylabel('Sales')
plt.title("Sales impacted by salesVisit3")


# Sales have been on the decreasing part at the end as per the salesVisit4

# In[354]:


df1.groupby('salesVisit4')['sales'].median().plot()
plt.xlabel('salesVisit4')
plt.ylabel('Sales')
plt.title("Sales impacted by salesVisit4")


# Sales have been on the increasing part at the end as per the salesVisit5 

# In[355]:


df1.groupby('salesVisit5')['sales'].median().plot()
plt.xlabel('salesVisit5')
plt.ylabel('Sales')
plt.title("Sales impacted by salesVisit5")


# In[356]:


df1['compBrand'].unique()


# Uniform growth in sales when compared with the compBrand

# In[357]:


df1.groupby('compBrand')['sales'].median().plot()
plt.xlabel('compBrand')
plt.ylabel('Sales')
plt.title("Sales impacted by compBrand")


# Plotting Heat map for the purpose of checking features if Multi-Collinearity exists and for better visual representation of data set

# In[358]:


import seaborn as sns
corr = df1.corr(method='kendall')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
df1.columns


# In[359]:


df1['accType'].unique()


# Count plot inorder to understand which acctype has maostly been utilized

# In[360]:


sns.countplot(df1['accType'], palette="plasma")
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.title('Account Types')


# In[361]:


sns.countplot(df1['accID'], palette="plasma")
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.title('Account ID')


# Pharmacy had more amount of sales than rest

# In[362]:


plt.figure(figsize=(10,10))
ax = sns.boxplot(data=df1, x='accType',y='sales',palette='plasma')


# compBrand 5 had more amount of sales than compBrand 4

# In[363]:


plt.figure(figsize=(10,6))
ax = sns.boxplot(data=df1, x='compBrand',y='sales',palette='plasma')


# The sales have been more in the year 2015 then in 2013,2014

# In[364]:


plt.figure(figsize=(10,6))
ax = sns.boxplot(data=df1, x='year',y='sales',palette='plasma')


# In[365]:


plt.figure(figsize=(10,6))
ax = sns.boxplot(data=df1, x='month',y='sales',palette='plasma')


# Been evident that compBrand 5 was introduced into the market in 2015

# In[366]:


plt.figure(figsize=(10,6))
ax = sns.boxplot(data=df1, x='year',y='compBrand',palette='plasma')


# In[367]:


df2=df1.copy()


# Plotting the Sales individually for all 3 strategies for the year 2015

# In[368]:


df2 = df1[df1.year == 2015][["year",'month',"sales",'strategy1','strategy2','strategy3','compBrand']]
df2.head()


# In[369]:


df2.groupby('strategy1')['sales'].median().plot()
plt.xlabel('Strategy1')
plt.ylabel('Sales')
plt.title("Sales Prediction by Strategy1")


# In[370]:


df2.groupby('strategy2')['sales'].median().plot()
plt.xlabel('Strategy2')
plt.ylabel('Sales')
plt.title("Sales Prediction by Strategy2")


# In[371]:


df2.groupby('strategy3')['sales'].median().plot()
plt.xlabel('Strategy3')
plt.ylabel('Sales')
plt.title("Sales Prediction by Strategy3")


# In[372]:


df2.groupby('month')['sales'].median().plot()
plt.xlabel('month')
plt.ylabel('Sales')
plt.title("Sales Prediction by month")


# In[373]:


plt.figure(figsize=(10,6))
ax = sns.boxplot(data=df2, x='month',y='sales',palette='plasma')


# In[374]:


df1['month'].unique()


# In[375]:


df2['month'].unique()


# Plotting the Sales individually for all 3 strategies for the years 2013,2014

# In[376]:


df3=df1.copy()
df3 = df1[df1.year != 2015 ][["year",'qty','month',"sales",'strategy1','strategy2','strategy3','compBrand','salesVisit1','salesVisit2','salesVisit3','salesVisit4','salesVisit5']]
df3.head()


# In[377]:


df3.groupby('strategy1')['sales'].median().plot()
plt.xlabel('Strategy1')
plt.ylabel('Sales')
plt.title("Sales Prediction by Strategy1")


# In[378]:


df3.groupby('strategy2')['sales'].median().plot()
plt.xlabel('Strategy2')
plt.ylabel('Sales')
plt.title("Sales Prediction by Strategy2")


# In[379]:


df3.groupby('strategy3')['sales'].median().plot()
plt.xlabel('Strategy3')
plt.ylabel('Sales')
plt.title("Sales Prediction by Strategy3")


# In[380]:


df3.groupby('month')['sales'].median().plot()
plt.xlabel('month')
plt.ylabel('Sales')
plt.title("Sales Prediction by month")


# In[381]:


plt.figure(figsize=(10,6))
ax = sns.boxplot(data=df3, x='month',y='sales',palette='plasma')


# In[382]:


plt.figure(figsize=(10,6))
ax = sns.boxplot(data=df3, x='year',y='sales',palette='plasma')


# Dropping of the compBrand feature as its cardinality is one

# In[383]:


df3.drop(['compBrand'],axis=1,inplace=True)
df3.head()


# Plotting the Pairplot to get an over all visualised view of the data set

# In[384]:


sns.pairplot(df3)


# In[385]:


df3.shape


# In[386]:


import seaborn as sns
corr = df3.corr(method='kendall')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
df2.columns


# # Prediction of Sales for strategy1

# As per the above Heat map few features such as year,month are ignored inorder to avoid high multi-collinearity

# In[387]:


for_strategy1=df3[['year','qty','strategy1','salesVisit1','salesVisit2','salesVisit3','salesVisit4','salesVisit5','sales']]


# In[388]:


for_strategy1.head(5)


# Dependent feature Sales is stored in a variable

# In[389]:


y=for_strategy1['sales']
y


# Removing the dependent feature so we can obtain all the essestial independent features

# In[390]:


X=for_strategy1.copy()
X.drop(['sales'],axis=1,inplace=True)
X.head()


# Built a Pipeline consisting of the machine learning model and standard scaler

# StandardScaler is used inorder to scale all the features values present in data set in the range -3 to +3 to achieve better model accuracy

# In[391]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[392]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# In[393]:


X_train.shape


# In[394]:


y_test.shape


# In[395]:


X_test.shape


# In[396]:


y_train.shape


# LinearRegression machine learning model

# In[397]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[398]:


pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('lr',LinearRegression())
])


# In[399]:


pipe.fit(X_train,y_train)


# In[400]:


lr_pred=pipe.predict(X_test)


# In[401]:


pd.DataFrame({'original test set':y_test, 'predictions': lr_pred})


# In[402]:


from sklearn import metrics
print('r2:', np.sqrt(metrics.r2_score(y_test, lr_pred)))


# In[403]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lr_pred)))


# In[404]:


plt.scatter(y_test,lr_pred)


# KFold Cross Valiadtion

# In[405]:


from sklearn.model_selection import cross_val_score,KFold
cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# Plotting the learning curves

# In[406]:


import scikitplot as skplt
skplt.estimators.plot_learning_curve(lr,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='LinearRegression')


# Lasso machine learning model

# In[407]:


from sklearn.linear_model import Lasso
lass=Lasso(alpha=1.0)
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('lass',Lasso(alpha=1.0))
])


# In[408]:


pipe.fit(X_train,y_train)


# In[409]:


lasso_pred=pipe.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': lasso_pred})


# In[410]:


print('r2:', np.sqrt(metrics.r2_score(y_test, lasso_pred)))


# In[411]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lasso_pred)))


# In[412]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[413]:


skplt.estimators.plot_learning_curve(lass,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='Lasso')


# XGboost machine learning model

# In[414]:


import xgboost as xgb
xg=Lasso(alpha=1.0)
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('xgb',xgb.XGBRegressor())
])


# In[415]:


pipe.fit(X_train,y_train)


# In[416]:


xgb_pred=pipe.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': xgb_pred})


# In[417]:


print('r2:', np.sqrt(metrics.r2_score(y_test, xgb_pred)))


# In[418]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, xgb_pred)))


# In[419]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[420]:


skplt.estimators.plot_learning_curve(xg,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='XGB')


# Support Vector Regressor machine learning model

# In[421]:


from sklearn.svm import SVR
sv=SVR()
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('svr',SVR(kernel='rbf'))
])


# In[422]:


pipe.fit(X_train,y_train)


# In[423]:


svr_pred=pipe.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': svr_pred})


# In[424]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svr_pred)))


# In[425]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[426]:


skplt.estimators.plot_learning_curve(sv,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='SVR')


# Decision Tree Regressor machine learning model

# In[427]:


from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('dtr',DecisionTreeRegressor())
])


# In[428]:


pipe.fit(X_train,y_train)


# In[429]:


dtr_pred=pipe.predict(X_test)


# In[430]:


pd.DataFrame({'original test set':y_test, 'predictions': dtr_pred})


# In[431]:


print('r2:', np.sqrt(metrics.r2_score(y_test, dtr_pred)))


# In[432]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dtr_pred)))


# In[433]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[434]:


skplt.estimators.plot_learning_curve(dtr,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='Decision Tree')


# Random Forest Regressor machine learning model

# In[435]:


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor()
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('rfr',RandomForestRegressor())
])


# In[436]:


pipe.fit(X_train,y_train)


# In[437]:


rfr_pred=pipe.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': rfr_pred})


# In[438]:


print('r2:', np.sqrt(metrics.r2_score(y_test, rfr_pred)))


# In[439]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfr_pred)))


# In[440]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[441]:


plt.scatter(y_test,rfr_pred)


# In[442]:


skplt.estimators.plot_learning_curve(rfr,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='Random Forest')


# Perform the machine learning model hyper parameter tuning by GridSearchCv for better model accuracy score

# Hyper parameter tuning for Random Forest Regressor

# In[443]:


from sklearn.model_selection import GridSearchCV
param_grid = {  'bootstrap': [True,False], 'max_depth': [5, 10, 15], 'n_estimators': [5, 6, 7, 8, 9, 10, 11, 12, 13, 15],'min_samples_split' : [2, 5, 10, 15, 100],'min_samples_leaf' : [1, 2, 5, 10]}
cv=KFold(n_splits=10,random_state=1,shuffle=True)
search = GridSearchCV(estimator = rfr, param_grid = param_grid,cv = cv,scoring='neg_mean_squared_error', n_jobs = -1, verbose = 0, return_train_score=True)
search.fit(X_train,y_train)


# In[444]:


print(search.best_params_)


# In[445]:


print(search.best_score_)


# In[446]:


rfr_search=search.predict(X_test)
result=pd.DataFrame({'original test set':y_test, 'predictions': rfr_search})
result


# In[447]:


print('r2:', np.sqrt(metrics.r2_score(y_test, rfr_search)))


# In[448]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfr_search)))


# Hyper parameter tuning for Lasso

# In[449]:


params = {'alpha': [1e-10,1e-6,1e-2,3,5,7,15,100,200,300,500,700]} # It will check from 1e-08 to 1e+08
lasso = Lasso()
cv=KFold(n_splits=10,random_state=1,shuffle=True)
lasso_model = GridSearchCV(lasso, params, cv = cv,scoring='neg_mean_squared_error')
lasso_model.fit(X_train, y_train)
print(lasso_model.best_params_)
print(lasso_model.best_score_)


# In[450]:


lass_modelpred=lasso_model.predict(X_test)
result=pd.DataFrame({'original test set':y_test, 'predictions': lass_modelpred})
result


# In[451]:


plt.scatter(y_test,lass_modelpred)


# In[452]:


print('r2:', np.sqrt(metrics.r2_score(y_test, lass_modelpred)))


# In[453]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lass_modelpred)))


# In[454]:


lasso_model.best_params_


# In[455]:


lasso_model.best_score_


# Hyper parameter tuning for XGBoostRegressor

# In[456]:


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


# In[457]:


print(xgb_grid.best_score_)
print(xgb_grid.best_params_)


# In[458]:


xgb_gridpred=xgb_grid.predict(X_test)


# In[459]:


pd.DataFrame({'original test set':y_test, 'predictions': xgb_gridpred})


# In[460]:


print('r2:', np.sqrt(metrics.r2_score(y_test, xgb_gridpred)))


# In[461]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, xgb_gridpred)))


# Out of all regression machine learning algorithms Lasso Regressor with GridSearchCV had best r2 score for strategy1 so we consider it as best performed algorithm and stored it in a data frame named "result"

# Calculating the mean of the final prediction of sales inorder to obtain the percentage of sales obtained for strategy1

# In[462]:


mean_of_sales_strategy1=result['predictions'].mean()
mean_of_sales_strategy1


# Calculating the individual percentages for the original and predicted predictions for Srategy1

# In[463]:


result['Percentage_of_sales_strategy1']=(result['predictions']/result['predictions'].sum())*100
result


# Calculating the average/mean value of all the final percenatge prediction of sales for Strategy1

# In[464]:


Percentage_of_sales_strategy1=result['Percentage_of_sales_strategy1'].mean()
Percentage_of_sales_strategy1


# # Prediction of Sales for strategy2

# In[465]:


for_strategy2=df3[['year','qty','strategy2','salesVisit1','salesVisit2','salesVisit3','salesVisit4','salesVisit5','sales']]


# In[466]:


for_strategy2.head(5)


# In[467]:


y=for_strategy2['sales']
y


# In[468]:


X=for_strategy2.copy()
X.drop(['sales'],axis=1,inplace=True)
X.head()


# In[469]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[470]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# In[471]:


X_train.shape


# In[472]:


y_test.shape


# In[473]:


X_test.shape


# In[474]:


y_train.shape


# In[475]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[476]:


pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('lr',LinearRegression())
])


# In[477]:


pipe.fit(X_train,y_train)


# In[478]:


lr_pred=pipe.predict(X_test)


# In[479]:


pd.DataFrame({'original test set':y_test, 'predictions': lr_pred})


# In[480]:


from sklearn import metrics
print('r2:', np.sqrt(metrics.r2_score(y_test, lr_pred)))


# In[481]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lr_pred)))


# In[482]:


plt.scatter(y_test,lr_pred)


# In[483]:


from sklearn.model_selection import cross_val_score,KFold
cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[484]:


import scikitplot as skplt
skplt.estimators.plot_learning_curve(lr,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='LinearRegression')


# In[485]:


from sklearn.linear_model import Lasso
lass=Lasso(alpha=1.0)
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('lass',Lasso(alpha=1.0))
])


# In[486]:


pipe.fit(X_train,y_train)


# In[487]:


lasso_pred=pipe.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': lasso_pred})


# In[488]:


print('r2:', np.sqrt(metrics.r2_score(y_test, lasso_pred)))


# In[489]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lasso_pred)))


# In[490]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[491]:


skplt.estimators.plot_learning_curve(lass,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='Lasso')


# In[492]:


import xgboost as xgb
xg=Lasso(alpha=1.0)
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('xgb',xgb.XGBRegressor())
])


# In[493]:


pipe.fit(X_train,y_train)


# In[494]:


xgb_pred=pipe.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': xgb_pred})


# In[495]:


print('r2:', np.sqrt(metrics.r2_score(y_test, xgb_pred)))


# In[496]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, xgb_pred)))


# In[497]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[498]:


skplt.estimators.plot_learning_curve(xg,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='XGB')


# In[499]:


from sklearn.svm import SVR
sv=SVR()
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('svr',SVR(kernel='rbf'))
])


# In[500]:


pipe.fit(X_train,y_train)


# In[501]:


svr_pred=pipe.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': svr_pred})


# In[502]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svr_pred)))


# In[503]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[504]:


skplt.estimators.plot_learning_curve(sv,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='SVR')


# In[505]:


from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('dtr',DecisionTreeRegressor())
])


# In[506]:


pipe.fit(X_train,y_train)


# In[507]:


dtr_pred=pipe.predict(X_test)


# In[508]:


pd.DataFrame({'original test set':y_test, 'predictions': dtr_pred})


# In[509]:


print('r2:', np.sqrt(metrics.r2_score(y_test, dtr_pred)))


# In[510]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dtr_pred)))


# In[511]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[512]:


skplt.estimators.plot_learning_curve(dtr,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='Decision Tree')


# In[513]:


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor()
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('rfr',RandomForestRegressor())
])


# In[514]:


pipe.fit(X_train,y_train)


# In[515]:


rfr_pred=pipe.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': rfr_pred})


# In[516]:


print('r2:', np.sqrt(metrics.r2_score(y_test, rfr_pred)))


# In[517]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfr_pred)))


# In[518]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[519]:


plt.scatter(y_test,rfr_pred)


# In[520]:


skplt.estimators.plot_learning_curve(rfr,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='Random Forest')


# In[521]:


from sklearn.model_selection import GridSearchCV
param_grid = {  'bootstrap': [True,False], 'max_depth': [5, 10, 15], 'n_estimators': [5, 6, 7, 8, 9, 10, 11, 12, 13, 15],'min_samples_split' : [2, 5, 10, 15, 100],'min_samples_leaf' : [1, 2, 5, 10]}
cv=KFold(n_splits=10,random_state=1,shuffle=True)
search = GridSearchCV(estimator = rfr, param_grid = param_grid,cv = cv,scoring='neg_mean_squared_error', n_jobs = -1, verbose = 0, return_train_score=True)
search.fit(X_train,y_train)


# In[522]:


print(search.best_params_)


# In[523]:


print(search.best_score_)


# In[524]:


rfr_search=search.predict(X_test)
result=pd.DataFrame({'original test set':y_test, 'predictions': rfr_search})
result


# In[525]:


print('r2:', np.sqrt(metrics.r2_score(y_test, rfr_search)))


# In[526]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfr_search)))


# In[527]:


params = {'alpha': [1e-10,1e-6,1e-2,3,5,7,15,100,200,300,500,700]}
lasso = Lasso()
cv=KFold(n_splits=10,random_state=1,shuffle=True)
lasso_model = GridSearchCV(lasso, params, cv = cv,scoring='neg_mean_squared_error')
lasso_model.fit(X_train, y_train)
print(lasso_model.best_params_)
print(lasso_model.best_score_)


# In[528]:


lass_modelpred=lasso_model.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': lass_modelpred})


# In[529]:


plt.scatter(y_test,lass_modelpred)


# In[530]:


print('r2:', np.sqrt(metrics.r2_score(y_test, lass_modelpred)))


# In[531]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lass_modelpred)))


# In[532]:


lasso_model.best_params_


# In[533]:


lasso_model.best_score_


# In[534]:


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


# In[535]:


print(xgb_grid.best_score_)
print(xgb_grid.best_params_)


# In[536]:


xgb_gridpred=xgb_grid.predict(X_test)


# In[537]:


pd.DataFrame({'original test set':y_test, 'predictions': xgb_gridpred})


# In[538]:


print('r2:', np.sqrt(metrics.r2_score(y_test, xgb_gridpred)))


# In[539]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, xgb_gridpred)))


# Out of all regression machine learning algorithms Random Forest Regressor with GridSearchCV had best r2 score for strategy2 so we consider it as best performed algorithm and stored it in a data frame named "result"

# Calculating the mean of the final prediction of sales inorder to obtain the percentage of sales obtained for strategy2

# In[540]:


mean_of_sales_strategy2=result['predictions'].mean()
mean_of_sales_strategy2


# Calculating the individual percentages for the original and predicted predictions for Srategy2

# In[541]:


result['Percentage_of_sales_strategy2']=(result['predictions']/result['predictions'].sum())*100
result


# Calculating the average/mean value of all the final percentage prediction of sales for Strategy2

# In[542]:


Percentage_of_sales_strategy2=result['Percentage_of_sales_strategy2'].mean()
Percentage_of_sales_strategy2


# # Prediction of Sales for strategy3

# In[543]:


for_strategy3=df3[['year','qty','strategy3','salesVisit1','salesVisit2','salesVisit3','salesVisit4','salesVisit5','sales']]


# In[544]:


for_strategy3.head(5)


# In[545]:


y=for_strategy3['sales']
y


# In[546]:


X=for_strategy3.copy()
X.drop(['sales'],axis=1,inplace=True)
X.head()


# In[547]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[548]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# In[549]:


X_train.shape


# In[550]:


y_test.shape


# In[551]:


X_test.shape


# In[552]:


y_train.shape


# In[553]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[554]:


pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('lr',LinearRegression())
])


# In[555]:


pipe.fit(X_train,y_train)


# In[556]:


lr_pred=pipe.predict(X_test)


# In[557]:


pd.DataFrame({'original test set':y_test, 'predictions': lr_pred})


# In[558]:


from sklearn import metrics
print('r2:', np.sqrt(metrics.r2_score(y_test, lr_pred)))


# In[559]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lr_pred)))


# In[560]:


plt.scatter(y_test,lr_pred)


# In[561]:


from sklearn.model_selection import cross_val_score,KFold
cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[562]:


import scikitplot as skplt
skplt.estimators.plot_learning_curve(lr,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='LinearRegression')


# In[563]:


from sklearn.linear_model import Lasso
lass=Lasso(alpha=1.0)
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('lass',Lasso(alpha=1.0))
])


# In[564]:


pipe.fit(X_train,y_train)


# In[565]:


lasso_pred=pipe.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': lasso_pred})


# In[566]:


print('r2:', np.sqrt(metrics.r2_score(y_test, lasso_pred)))


# In[567]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lasso_pred)))


# In[568]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[569]:


skplt.estimators.plot_learning_curve(lass,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='Lasso')


# In[570]:


import xgboost as xgb
xg=Lasso(alpha=1.0)
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('xgb',xgb.XGBRegressor())
])


# In[571]:


pipe.fit(X_train,y_train)


# In[572]:


xgb_pred=pipe.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': xgb_pred})


# In[573]:


print('r2:', np.sqrt(metrics.r2_score(y_test, xgb_pred)))


# In[574]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, xgb_pred)))


# In[575]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[576]:


skplt.estimators.plot_learning_curve(xg,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='XGB')


# In[577]:


from sklearn.svm import SVR
sv=SVR()
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('svr',SVR(kernel='rbf'))
])


# In[578]:


pipe.fit(X_train,y_train)


# In[579]:


svr_pred=pipe.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': svr_pred})


# In[580]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svr_pred)))


# In[581]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[582]:


skplt.estimators.plot_learning_curve(sv,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='SVR')


# In[583]:


from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('dtr',DecisionTreeRegressor())
])


# In[584]:


pipe.fit(X_train,y_train)


# In[585]:


dtr_pred=pipe.predict(X_test)


# In[586]:


pd.DataFrame({'original test set':y_test, 'predictions': dtr_pred})


# In[587]:


print('r2:', np.sqrt(metrics.r2_score(y_test, dtr_pred)))


# In[588]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dtr_pred)))


# In[589]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[590]:


skplt.estimators.plot_learning_curve(dtr,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='Decision Tree')


# In[591]:


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor()
pipe = Pipeline(steps =[
    ('scaler', StandardScaler()), 
    ('rfr',RandomForestRegressor())
])


# In[592]:


pipe.fit(X_train,y_train)


# In[593]:


rfr_pred=pipe.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': rfr_pred})


# In[594]:


print('r2:', np.sqrt(metrics.r2_score(y_test, rfr_pred)))


# In[595]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfr_pred)))


# In[596]:


cv=KFold(n_splits=10,random_state=1,shuffle=True)
cvs = cross_val_score(pipe, X_train, y_train, cv = cv)
print("All cross val scores:", cvs)
print("Mean of all scores: ", cvs.mean())


# In[597]:


plt.scatter(y_test,rfr_pred)


# In[598]:


skplt.estimators.plot_learning_curve(rfr,X_train,y_train,cv=7,figsize=(6,4),title_fontsize='large',title='Random Forest')


# Utlizing GridSearchCV to perform hyper parameter tuning

# In[599]:


from sklearn.model_selection import GridSearchCV
param_grid = {  'bootstrap': [True,False], 'max_depth': [5, 10, 15], 'n_estimators': [5, 6, 7, 8, 9, 10, 11, 12, 13, 15],'min_samples_split' : [2, 5, 10, 15, 100],'min_samples_leaf' : [1, 2, 5, 10]}
cv=KFold(n_splits=10,random_state=1,shuffle=True)
search = GridSearchCV(estimator = rfr, param_grid = param_grid,cv = cv,scoring='neg_mean_squared_error', n_jobs = -1, verbose = 0, return_train_score=True)
search.fit(X_train,y_train)


# In[600]:


print(search.best_params_)


# In[601]:


print(search.best_score_)


# In[602]:


rfr_search=search.predict(X_test)
pd.DataFrame({'original test set':y_test, 'predictions': rfr_search})


# In[603]:


print('r2:', np.sqrt(metrics.r2_score(y_test, rfr_search)))


# In[604]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfr_search)))


# Hyper parameter tuning for Lasso regressor

# In[605]:


params = {'alpha': [1e-10,1e-6,1e-2,3,5,7,15,100,200,300,500,700]} # It will check from 1e-08 to 1e+08
lasso = Lasso()
cv=KFold(n_splits=10,random_state=1,shuffle=True)
lasso_model = GridSearchCV(lasso, params, cv = cv,scoring='neg_mean_squared_error')
lasso_model.fit(X_train, y_train)
print(lasso_model.best_params_)
print(lasso_model.best_score_)


# In[606]:


lass_modelpred=lasso_model.predict(X_test)
result=pd.DataFrame({'original test set':y_test, 'predictions': lass_modelpred})
result


# In[607]:


plt.scatter(y_test,lass_modelpred)


# In[608]:


print('r2:', np.sqrt(metrics.r2_score(y_test, lass_modelpred)))


# In[609]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lass_modelpred)))


# In[610]:


lasso_model.best_params_


# In[611]:


lasso_model.best_score_


# Hyper parameter tuning for XGBRegressor

# In[612]:


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


# In[613]:


print(xgb_grid.best_score_)
print(xgb_grid.best_params_)


# In[614]:


xgb_gridpred=xgb_grid.predict(X_test)


# In[615]:


pd.DataFrame({'original test set':y_test, 'predictions': xgb_gridpred})


# In[616]:


print('r2:', np.sqrt(metrics.r2_score(y_test, xgb_gridpred)))


# In[617]:


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, xgb_gridpred)))


# Out of all regression machine learning algorithms Lasso Regressor had best r2 score for strategy3 so we consider it as best performed algorithm and stored it in a data frame named "result"

# Calculating the mean of the final prediction of sales inorder to obtain the percentage of sales obtained for strategy3

# In[618]:


mean_of_sales_strategy3=result['predictions'].mean()
mean_of_sales_strategy3


# Calculating the individual percentages for the original and predicted predictions for Srategy3

# In[619]:


result['Percentage_of_sales_strategy3']=(result['predictions']/result['predictions'].sum())*100
result


# Calculating the average/mean value of all the final percenatge prediction of sales for Strategy3

# In[620]:


Percentage_of_sales_strategy3=result['Percentage_of_sales_strategy3'].mean()
Percentage_of_sales_strategy3


# Summary of all above observation of all 3 Strategies for prediction of Sales

# In[621]:


Percentage_of_sales_strategy1


# In[622]:


Percentage_of_sales_strategy2


# In[623]:


Percentage_of_sales_strategy3


# Clearly the Strategy1 had slightly better impact on the in terms of percentage of prediction of Sales than Strategy2,Strategy3.
# Order:
# Strategy1,Strategy2,Strategy2

# In[624]:


mean_of_sales_strategy1


# In[625]:


mean_of_sales_strategy2


# In[626]:


mean_of_sales_strategy3


# Clearly the Strategy1 had slightly better impact on the in terms of mean of prediction of Sales than Strategy2,Strategy3.
# Order:
# Strategy1,Strategy2,Strategy3
