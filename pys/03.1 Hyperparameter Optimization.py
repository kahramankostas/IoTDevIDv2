
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from scipy.stats import uniform as sp_randFloat
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from time import time
import numpy as np
import pandas as pd
import sklearn
import warnings
warnings.filterwarnings('ignore')


# In[2]:


from scipy.stats import randint as sp_randInt

from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer
from scipy import sparse


# In[3]:


features= ['pck_size', 'Ether_type', 'LLC_ctrl', 'EAPOL_version', 'EAPOL_type', 'IP_ihl', 'IP_tos', 'IP_len', 'IP_flags', 'IP_DF', 'IP_ttl', 'IP_options', 'ICMP_code', 'TCP_dataofs', 'TCP_FIN', 'TCP_ACK', 'TCP_window', 'UDP_len', 'DHCP_options', 'BOOTP_hlen', 'BOOTP_flags', 'BOOTP_sname', 'BOOTP_file', 'BOOTP_options', 'DNS_qr', 'DNS_rd', 'DNS_qdcount', 'dport_class', 'payload_bytes', 'entropy', 'Label']

df=pd.read_csv("Aalto_train_IoTDevID.csv",usecols=features) 
X_train = df.iloc[:,0:-1]
df['Label'] = df['Label'].astype('category')
y_train=df['Label'].cat.codes  


df=pd.read_csv("Aalto_test_IoTDevID.csv",usecols=features) 
X_test = df.iloc[:,0:-1]
df['Label'] = df['Label'].astype('category')
y_test=df['Label'].cat.codes  


# In[4]:


print(X_train.shape,X_test.shape)


# # Aalto

# In[5]:




X= np.concatenate([X_train, X_test])
test_fold = [-1 for _ in range(X_train.shape[0])] + [0 for _ in range(X_test.shape[0])]
y = np.concatenate([y_train, y_test])
ps = PredefinedSplit(test_fold)

def run_random_search(model, params, x_train, y_train):
    #grid = GridSearchCV(model, params, cv = ps, n_jobs = -1, scoring = score, verbose = 0, refit = False)
    grid =RandomizedSearchCV(model, param_grid, cv=ps,scoring = 'f1_macro')
    grid.fit(x_train, y_train)
    return (grid.best_params_, round(grid.best_score_,8),grid.best_estimator_)


# # RandomizedSearchCV  DT

# In[11]:


print ('%-90s %-20s %-8s %-8s' % ("HYPERPARAMETERS","F1 Score", "Time", "No"))
  


nfolds=10
param_grid = { 'criterion':['gini','entropy'],
                  "max_depth":np.linspace(1, 32, 32, endpoint=True),
                 "min_samples_split": sp_randint(2,10),#uniform(0.1,1 ),
                    # "min_samples_leafs" : np.linspace(0.1, 0.5, 5, endpoint=True)
                    "max_features" : sp_randint(1,X_train.shape[1])}

second=time()
f1=[]
clf=DecisionTreeClassifier()
for ii in range(25):
    clf.fit(X_train, y_train)
    predict =clf.predict(X_test)
    f1.append(sklearn.metrics.f1_score(y_test, predict,average= "macro") )
f1=sum(f1)/len(f1)   
#if f1>0.76:
print('%-90s %-20s %-8s %-8s' % ("default",f1,round(time()-second,3),ii))

for i in range(100):
    second=time()
    a,b,clf=run_random_search(DecisionTreeClassifier(),param_grid,X,y)
    f1=[]
    for ii in range(25):
        clf.fit(X_train, y_train)
        predict =clf.predict(X_test)
        f1.append(sklearn.metrics.f1_score(y_test, predict,average= "macro") )
    f1=sum(f1)/len(f1)   
    #if f1>0.76:
    print('%-90s %-20s %-8s %-8s' % (a,f1,round(time()-second,3),i))
    
    
    
    


# # GridSearchCV DT

# In[21]:


param_grid = { 'criterion':['gini','entropy'],
                  "max_depth":list(range(1,32)),
                 "min_samples_split":list(range(2,10)),#uniform(0.1,1 ),
                    # "min_samples_leafs" : np.linspace(0.1, 0.5, 5, endpoint=True)
                    "max_features" :list(range(1,X_train.shape[1]))}



nbModel_grid = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid, verbose=1, cv=ps, n_jobs=-1)
nbModel_grid.fit(X, y)
print(nbModel_grid.best_estimator_)


# # RandomizedSearchCV RF

# In[12]:


# use a full grid over all parameters
param_grid = {"max_depth":np.linspace(1, 32, 32, endpoint=True),
              "n_estimators" : sp_randint(1, 200),
              "max_features": sp_randint(1, 11),
              "min_samples_split":sp_randint(2, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
second=time()
f1=[]
clf=RandomForestClassifier()
for ii in range(1):
    clf.fit(X_train, y_train)
    predict =clf.predict(X_test)
    f1.append(sklearn.metrics.f1_score(y_test, predict,average= "macro") )
f1=sum(f1)/len(f1)   
#if f1>0.76:
print('%-90s %-20s %-8s %-8s' % ("default",f1,round(time()-second,3),ii))

for i in range(50):
    second=time()
    a,b,clf=run_random_search(RandomForestClassifier(),param_grid,X,y)
    clf.fit(X_train, y_train)
    predict =clf.predict(X_test)
    f1=sklearn.metrics.f1_score(y_test, predict,average= "macro") 
    print('%-90s %-20s %-8s %-8s' % (a,f1,round(time()-second,3),i))


# # RandomizedSearchCV  KNeighborsClassifier

# In[13]:


# use a full grid over all parameters
param_grid = {"n_neighbors" : sp_randint(1,64) ,  
             "leaf_size": sp_randint(1,50) , 
              "algorithm" : ["auto", "ball_tree", "kd_tree", "brute"],
              "weights" : ["uniform", "distance"]}
second=time()
f1=[]
clf=KNeighborsClassifier()
for ii in range(1):
    clf.fit(X_train, y_train)
    predict =clf.predict(X_test)
    f1.append(sklearn.metrics.f1_score(y_test, predict,average= "macro") )
f1=sum(f1)/len(f1)   
#if f1>0.76:
print('%-90s %-20s %-8s %-8s' % ("default",f1,round(time()-second,3),i))


for i in range(50):
    second=time()
    a,b,clf=run_random_search(KNeighborsClassifier(),param_grid,X,y)

    clf.fit(X_train, y_train)
    predict =clf.predict(X_test)
    f1=sklearn.metrics.f1_score(y_test, predict,average= "macro") 
    print('%-90s %-20s %-8s %-8s' % (a,f1,round(time()-second,3),i))
    


# # RandomizedSearchCV  GradientBoostingClassifier

# In[14]:


# use a full grid over all parameters
param_grid =  {"learning_rate": sp_randFloat(),
          "subsample"    : sp_randFloat(),
          "n_estimators" : sp_randInt(100, 1000),
          "max_depth"    : sp_randInt(4, 10)
         }

second=time()
f1=[]
clf=GradientBoostingClassifier()
for ii in range(1):
    clf.fit(X_train, y_train)
    predict =clf.predict(X_test)
    f1.append(sklearn.metrics.f1_score(y_test, predict,average= "macro") )
f1=sum(f1)/len(f1)   
#if f1>0.76:
print('%-90s %-20s %-8s %-8s' % ("default",f1,round(time()-second,3),ii))


for i in range(1):
    second=time()
    a,b,clf=run_random_search(GradientBoostingClassifier(),param_grid,X,y)
    clf.fit(X_train, y_train)
    predict =clf.predict(X_test)
    f1=sklearn.metrics.f1_score(y_test, predict,average= "macro") 
    print('%-90s %-20s %-8s %-8s' % (a,f1,round(time()-second,3),i))
    


# #  SVM

# In[15]:


param_grid =  {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma' : [0.001, 0.01, 0.1, 1]}    
nbModel_grid = GridSearchCV(estimator=svm.SVC(), param_grid=param_grid, verbose=1, cv=ps, n_jobs=-1)
nbModel_grid.fit(X, y)
print(nbModel_grid.best_estimator_)    


# #   NB

# In[10]:


from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import GridSearchCV

param_grid_nb = {
    'alpha': np.logspace(0,-9, num=100),
    "fit_prior":["True","False"]
}
nbModel_grid = GridSearchCV(estimator=CategoricalNB(), param_grid=param_grid_nb, verbose=1, cv=ps, n_jobs=-1)
nbModel_grid.fit(X, y)
print(nbModel_grid.best_estimator_)


# -------------

# In[2]:


#  IoTSense- IoTsentinel 


# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from scipy.stats import uniform as sp_randFloat
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from time import time
import numpy as np
import pandas as pd
import sklearn
import warnings
warnings.filterwarnings('ignore')


# In[2]:


from scipy.stats import randint as sp_randInt

from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer
from scipy import sparse


# # IoTSentinel

# In[3]:


df=pd.read_csv("Aalto_IoTSentinel_Train.csv")
df


# In[4]:


df.columns


# In[5]:


features= ['ARP', 'LLC', 'EAPOL', 'IP', 'ICMP', 'ICMP6', 'TCP', 'UDP', 'HTTP',
       'HTTPS', 'DHCP', 'BOOTP', 'SSDP', 'DNS', 'MDNS', 'NTP', 'IP_padding',
       'IP_add_count', 'IP_ralert', 'Portcl_src', 'Portcl_dst', 'Pck_size',
       'Pck_rawdata',  'Label']
df=pd.read_csv("Aalto_IoTSentinel_Train.csv",usecols=features) 
X_train = df.iloc[:,0:-1]
df['Label'] = df['Label'].astype('category')
y_train=df['Label'].cat.codes  


df=pd.read_csv("Aalto_IoTSentinel_Test.csv",usecols=features) 
X_test = df.iloc[:,0:-1]
df['Label'] = df['Label'].astype('category')
y_test=df['Label'].cat.codes  


# In[6]:


print(X_train.shape,X_test.shape)


# In[7]:


X= np.concatenate([X_train, X_test])
test_fold = [-1 for _ in range(X_train.shape[0])] + [0 for _ in range(X_test.shape[0])]
y = np.concatenate([y_train, y_test])
ps = PredefinedSplit(test_fold)

def run_random_search(model, params, x_train, y_train):
    #grid = GridSearchCV(model, params, cv = ps, n_jobs = -1, scoring = score, verbose = 0, refit = False)
    grid =RandomizedSearchCV(model, param_grid, cv=ps,scoring = 'f1_macro')
    grid.fit(x_train, y_train)
    return (grid.best_params_, round(grid.best_score_,8),grid.best_estimator_)


# # RandomizedSearchCV  DT

# In[8]:


print ('%-90s %-20s %-8s %-8s' % ("HYPERPARAMETERS","F1 Score", "Time", "No"))
  


nfolds=10
param_grid = { 'criterion':['gini','entropy'],
                  "max_depth":np.linspace(1, 32, 32, endpoint=True),
                 "min_samples_split": sp_randint(2,10),#uniform(0.1,1 ),
                    # "min_samples_leafs" : np.linspace(0.1, 0.5, 5, endpoint=True)
                    "max_features" : sp_randint(1,X_train.shape[1])}

second=time()
f1=[]
clf=DecisionTreeClassifier()
for ii in range(25):
    clf.fit(X_train, y_train)
    predict =clf.predict(X_test)
    f1.append(sklearn.metrics.f1_score(y_test, predict,average= "macro") )
f1=sum(f1)/len(f1)   
#if f1>0.76:
print('%-90s %-20s %-8s %-8s' % ("default",f1,round(time()-second,3),ii))

for i in range(100):
    second=time()
    a,b,clf=run_random_search(DecisionTreeClassifier(),param_grid,X,y)
    f1=[]
    for ii in range(25):
        clf.fit(X_train, y_train)
        predict =clf.predict(X_test)
        f1.append(sklearn.metrics.f1_score(y_test, predict,average= "macro") )
    f1=sum(f1)/len(f1)   
    #if f1>0.76:
    print('%-90s %-20s %-8s %-8s' % (a,f1,round(time()-second,3),i))
    
    
    
    


# # IoT Sense

# In[13]:


df=pd.read_csv("Aalto_IoTSense_Train.csv")
df


# In[14]:


df.columns


# In[15]:


features= ['ARP', 'EAPOL', 'IP', 'ICMP', 'ICMP6', 'TCP', 'UDP', 'TCP_w_size',
       'HTTP', 'HTTPS', 'DHCP', 'BOOTP', 'SSDP', 'DNS', 'MDNS', 'NTP',
       'IP_padding', 'IP_ralert', 'payload_l', 'Entropy', 'Label']
df=pd.read_csv("Aalto_IoTSense_Train.csv",usecols=features) 
X_train = df.iloc[:,0:-1]
df['Label'] = df['Label'].astype('category')
y_train=df['Label'].cat.codes  


df=pd.read_csv("Aalto_IoTSense_Test.csv",usecols=features) 
X_test = df.iloc[:,0:-1]
df['Label'] = df['Label'].astype('category')
y_test=df['Label'].cat.codes  


# In[16]:


print(X_train.shape,X_test.shape)


# In[17]:


X= np.concatenate([X_train, X_test])
test_fold = [-1 for _ in range(X_train.shape[0])] + [0 for _ in range(X_test.shape[0])]
y = np.concatenate([y_train, y_test])
ps = PredefinedSplit(test_fold)

def run_random_search(model, params, x_train, y_train):
    #grid = GridSearchCV(model, params, cv = ps, n_jobs = -1, scoring = score, verbose = 0, refit = False)
    grid =RandomizedSearchCV(model, param_grid, cv=ps,scoring = 'f1_macro')
    grid.fit(x_train, y_train)
    return (grid.best_params_, round(grid.best_score_,8),grid.best_estimator_)


# # RandomizedSearchCV  DT

# In[18]:


print ('%-90s %-20s %-8s %-8s' % ("HYPERPARAMETERS","F1 Score", "Time", "No"))
  


nfolds=10
param_grid = { 'criterion':['gini','entropy'],
                  "max_depth":np.linspace(1, 32, 32, endpoint=True),
                 "min_samples_split": sp_randint(2,10),#uniform(0.1,1 ),
                    # "min_samples_leafs" : np.linspace(0.1, 0.5, 5, endpoint=True)
                    "max_features" : sp_randint(1,X_train.shape[1])}

second=time()
f1=[]
clf=DecisionTreeClassifier()
for ii in range(25):
    clf.fit(X_train, y_train)
    predict =clf.predict(X_test)
    f1.append(sklearn.metrics.f1_score(y_test, predict,average= "macro") )
f1=sum(f1)/len(f1)   
#if f1>0.76:
print('%-90s %-20s %-8s %-8s' % ("default",f1,round(time()-second,3),ii))

for i in range(100):
    second=time()
    a,b,clf=run_random_search(DecisionTreeClassifier(),param_grid,X,y)
    f1=[]
    for ii in range(25):
        clf.fit(X_train, y_train)
        predict =clf.predict(X_test)
        f1.append(sklearn.metrics.f1_score(y_test, predict,average= "macro") )
    f1=sum(f1)/len(f1)   
    #if f1>0.76:
    print('%-90s %-20s %-8s %-8s' % (a,f1,round(time()-second,3),i))
    
    
    
    


# ________________

# # UNSW

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from scipy.stats import uniform as sp_randFloat
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from time import time
import numpy as np
import pandas as pd
import sklearn
import warnings
warnings.filterwarnings('ignore')


# In[4]:


from scipy.stats import randint as sp_randInt

from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer
from scipy import sparse


# # IoTSentinel

# In[5]:


df=pd.read_csv("UNSW_IoTSentinel_Train.csv")
df


# In[6]:


df.columns


# In[7]:


features= ['ARP', 'LLC', 'EAPOL', 'IP', 'ICMP', 'ICMP6', 'TCP', 'UDP', 'HTTP',
       'HTTPS', 'DHCP', 'BOOTP', 'SSDP', 'DNS', 'MDNS', 'NTP', 'IP_padding',
       'IP_add_count', 'IP_ralert', 'Portcl_src', 'Portcl_dst', 'Pck_size',
       'Pck_rawdata',  'Label']
df=pd.read_csv("UNSW_IoTSentinel_Train.csv",usecols=features) 
X_train = df.iloc[:,0:-1]
df['Label'] = df['Label'].astype('category')
y_train=df['Label'].cat.codes  


df=pd.read_csv("UNSW_IoTSentinel_Test.csv",usecols=features) 
X_test = df.iloc[:,0:-1]
df['Label'] = df['Label'].astype('category')
y_test=df['Label'].cat.codes  


# In[8]:


print(X_train.shape,X_test.shape)


# In[9]:


X= np.concatenate([X_train, X_test])
test_fold = [-1 for _ in range(X_train.shape[0])] + [0 for _ in range(X_test.shape[0])]
y = np.concatenate([y_train, y_test])
ps = PredefinedSplit(test_fold)

def run_random_search(model, params, x_train, y_train):
    #grid = GridSearchCV(model, params, cv = ps, n_jobs = -1, scoring = score, verbose = 0, refit = False)
    grid =RandomizedSearchCV(model, param_grid, cv=ps,scoring = 'f1_macro')
    grid.fit(x_train, y_train)
    return (grid.best_params_, round(grid.best_score_,8),grid.best_estimator_)


# # RandomizedSearchCV  DT

# In[10]:


print ('%-90s %-20s %-8s %-8s' % ("HYPERPARAMETERS","F1 Score", "Time", "No"))
  


nfolds=10
param_grid = { 'criterion':['gini','entropy'],
                  "max_depth":np.linspace(1, 32, 32, endpoint=True),
                 "min_samples_split": sp_randint(2,10),#uniform(0.1,1 ),
                    # "min_samples_leafs" : np.linspace(0.1, 0.5, 5, endpoint=True)
                    "max_features" : sp_randint(1,X_train.shape[1])}

second=time()
f1=[]
clf=DecisionTreeClassifier()
for ii in range(25):
    clf.fit(X_train, y_train)
    predict =clf.predict(X_test)
    f1.append(sklearn.metrics.f1_score(y_test, predict,average= "macro") )
f1=sum(f1)/len(f1)   
#if f1>0.76:
print('%-90s %-20s %-8s %-8s' % ("default",f1,round(time()-second,3),ii))

for i in range(100):
    second=time()
    a,b,clf=run_random_search(DecisionTreeClassifier(),param_grid,X,y)
    f1=[]
    for ii in range(25):
        clf.fit(X_train, y_train)
        predict =clf.predict(X_test)
        f1.append(sklearn.metrics.f1_score(y_test, predict,average= "macro") )
    f1=sum(f1)/len(f1)   
    #if f1>0.76:
    print('%-90s %-20s %-8s %-8s' % (a,f1,round(time()-second,3),i))
    
    
    
    


# # IoT Sense

# In[11]:


df=pd.read_csv("UNSW_IoTSense_Train.csv")
df


# In[12]:


df.columns


# In[17]:


features= ['ARP', 'EAPOL', 'IP', 'ICMP', 'ICMP6', 'TCP', 'UDP', 'TCP_w_size',
       'HTTP', 'HTTPS', 'DHCP', 'BOOTP', 'SSDP', 'DNS', 'MDNS', 'NTP',
       'IP_padding', 'IP_ralert', 'payload_l', 'Entropy', 'Label']
df=pd.read_csv("UNSW_IoTSense_Train.csv",usecols=features) 
X_train = df.iloc[:,0:-1]
df['Label'] = df['Label'].astype('category')
y_train=df['Label'].cat.codes  


df=pd.read_csv("UNSW_IoTSense_Test.csv",usecols=features) 
X_test = df.iloc[:,0:-1]
df['Label'] = df['Label'].astype('category')
y_test=df['Label'].cat.codes  


# In[18]:


print(X_train.shape,X_test.shape)


# In[19]:


X= np.concatenate([X_train, X_test])
test_fold = [-1 for _ in range(X_train.shape[0])] + [0 for _ in range(X_test.shape[0])]
y = np.concatenate([y_train, y_test])
ps = PredefinedSplit(test_fold)

def run_random_search(model, params, x_train, y_train):
    #grid = GridSearchCV(model, params, cv = ps, n_jobs = -1, scoring = score, verbose = 0, refit = False)
    grid =RandomizedSearchCV(model, param_grid, cv=ps,scoring = 'f1_macro')
    grid.fit(x_train, y_train)
    return (grid.best_params_, round(grid.best_score_,8),grid.best_estimator_)


# # RandomizedSearchCV  DT

# In[20]:


print ('%-90s %-20s %-8s %-8s' % ("HYPERPARAMETERS","F1 Score", "Time", "No"))
  


nfolds=10
param_grid = { 'criterion':['gini','entropy'],
                  "max_depth":np.linspace(1, 32, 32, endpoint=True),
                 "min_samples_split": sp_randint(2,10),#uniform(0.1,1 ),
                    # "min_samples_leafs" : np.linspace(0.1, 0.5, 5, endpoint=True)
                    "max_features" : sp_randint(1,X_train.shape[1])}

second=time()
f1=[]
clf=DecisionTreeClassifier()
for ii in range(25):
    clf.fit(X_train, y_train)
    predict =clf.predict(X_test)
    f1.append(sklearn.metrics.f1_score(y_test, predict,average= "macro") )
f1=sum(f1)/len(f1)   
#if f1>0.76:
print('%-90s %-20s %-8s %-8s' % ("default",f1,round(time()-second,3),ii))

for i in range(100):
    second=time()
    a,b,clf=run_random_search(DecisionTreeClassifier(),param_grid,X,y)
    f1=[]
    for ii in range(25):
        clf.fit(X_train, y_train)
        predict =clf.predict(X_test)
        f1.append(sklearn.metrics.f1_score(y_test, predict,average= "macro") )
    f1=sum(f1)/len(f1)   
    #if f1>0.76:
    print('%-90s %-20s %-8s %-8s' % (a,f1,round(time()-second,3),i))
    
    
    
    


# # IoTDevID

# In[21]:


df=pd.read_csv("UNSW_train_IoTDevID.csv")
df


# In[22]:


df.columns


# In[23]:


features= ['pck_size', 'Ether_type', 'LLC_ctrl', 'EAPOL_version', 'EAPOL_type', 'IP_ihl', 'IP_tos', 'IP_len', 'IP_flags', 'IP_DF', 'IP_ttl', 'IP_options', 'ICMP_code', 'TCP_dataofs', 'TCP_FIN', 'TCP_ACK', 'TCP_window', 'UDP_len', 'DHCP_options', 'BOOTP_hlen', 'BOOTP_flags', 'BOOTP_sname', 'BOOTP_file', 'BOOTP_options', 'DNS_qr', 'DNS_rd', 'DNS_qdcount', 'dport_class', 'payload_bytes', 'entropy',
'Label']
df=pd.read_csv("UNSW_train_IoTDevID.csv",usecols=features) 
X_train = df.iloc[:,0:-1]
df['Label'] = df['Label'].astype('category')
y_train=df['Label'].cat.codes  


df=pd.read_csv("UNSW_test_IoTDevID.csv",usecols=features) 
X_test = df.iloc[:,0:-1]
df['Label'] = df['Label'].astype('category')
y_test=df['Label'].cat.codes  


# In[24]:


print(X_train.shape,X_test.shape)


# In[25]:


X= np.concatenate([X_train, X_test])
test_fold = [-1 for _ in range(X_train.shape[0])] + [0 for _ in range(X_test.shape[0])]
y = np.concatenate([y_train, y_test])
ps = PredefinedSplit(test_fold)

def run_random_search(model, params, x_train, y_train):
    #grid = GridSearchCV(model, params, cv = ps, n_jobs = -1, scoring = score, verbose = 0, refit = False)
    grid =RandomizedSearchCV(model, param_grid, cv=ps,scoring = 'f1_macro')
    grid.fit(x_train, y_train)
    return (grid.best_params_, round(grid.best_score_,8),grid.best_estimator_)


# # RandomizedSearchCV  DT

# In[26]:


print ('%-90s %-20s %-8s %-8s' % ("HYPERPARAMETERS","F1 Score", "Time", "No"))
  


nfolds=10
param_grid = { 'criterion':['gini','entropy'],
                  "max_depth":np.linspace(1, 32, 32, endpoint=True),
                 "min_samples_split": sp_randint(2,10),#uniform(0.1,1 ),
                    # "min_samples_leafs" : np.linspace(0.1, 0.5, 5, endpoint=True)
                    "max_features" : sp_randint(1,X_train.shape[1])}

second=time()
f1=[]
clf=DecisionTreeClassifier()
for ii in range(25):
    clf.fit(X_train, y_train)
    predict =clf.predict(X_test)
    f1.append(sklearn.metrics.f1_score(y_test, predict,average= "macro") )
f1=sum(f1)/len(f1)   
#if f1>0.76:
print('%-90s %-20s %-8s %-8s' % ("default",f1,round(time()-second,3),ii))

for i in range(100):
    second=time()
    a,b,clf=run_random_search(DecisionTreeClassifier(),param_grid,X,y)
    f1=[]
    for ii in range(25):
        clf.fit(X_train, y_train)
        predict =clf.predict(X_test)
        f1.append(sklearn.metrics.f1_score(y_test, predict,average= "macro") )
    f1=sum(f1)/len(f1)   
    #if f1>0.76:
    print('%-90s %-20s %-8s %-8s' % (a,f1,round(time()-second,3),i))
    
    
    
    

