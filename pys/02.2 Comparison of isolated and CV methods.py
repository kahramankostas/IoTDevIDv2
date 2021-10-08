
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
from numpy import array
from random import random
from sklearn import metrics

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


from sklearn.metrics import balanced_accuracy_score



import csv
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn
import time

from sklearn.metrics import classification_report


from sklearn.utils import shuffle


# In[3]:


def folder(f_name): #this function creates a folder named "attacks" in the program directory.
    try:
        if not os.path.exists(f_name):
            os.makedirs(f_name)
    except OSError:
        print ("The folder could not be created!")


# In[4]:


def find_the_way(path,file_format):
    files_add = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                files_add.append(os.path.join(r, file))  
    return files_add



# In[5]:


ml_list={"DT":DecisionTreeClassifier()}


# In[6]:


def target_name(name):
    df = pd.read_csv(name,usecols=["Label"])
    target_names=sorted(list(df["Label"].unique()))
    return target_names


# ## Calculation of evaluations

# In[7]:


def score(altime,train_time,test_time,predict,y_test,class_based_results,i,cv,dname,ii):
    precision=[]
    recall=[]
    f1=[]
    accuracy=[]
    total_time=[]
    kappa=[]
    accuracy_b=[]
    
    rc=sklearn.metrics.recall_score(y_test, predict,average= "macro")
    pr=sklearn.metrics.precision_score(y_test, predict,average= "macro")
    f_1=sklearn.metrics.f1_score(y_test, predict,average= "macro")        
    report = classification_report(y_test, predict, target_names=target_names,output_dict=True)
    cr = pd.DataFrame(report).transpose()
    if class_based_results.empty:
        class_based_results =cr
    else:
        class_based_results = class_based_results.add(cr, fill_value=0)
    precision.append(float(pr))
    recall.append(float(rc))
    f1.append(float(f_1))
    accuracy_b.append(balanced_accuracy_score( y_test,predict))
    accuracy.append(accuracy_score(y_test, predict))
    #clf.score(X_test, y_test))
    #print(balanced_accuracy_score( y_test,predict))
    #t_time.append(float((time.time()-second)) )
    kappa.append(round(float(sklearn.metrics.cohen_kappa_score(y_test, predict, 
    labels=None, weights=None, sample_weight=None)),15))
    print ('%-15s %-3s %-3s %-6s  %-5s %-5s %-5s %-5s %-8s %-5s %-8s %-8s%-8s%-8s' % (dname,i,cv,ii[0:6],str(round(np.mean(accuracy),2)),str(round(np.mean(accuracy_b),2)),
        str(round(np.mean(precision),2)), str(round(np.mean(recall),2)),str(round(np.mean(f1),4)), 
        str(round(np.mean(kappa),2)),str(round(np.mean(train_time),2)),str(round(np.mean(test_time),2)),str(round(np.mean(test_time)+np.mean(train_time),2)),str(round(np.mean(altime),2))))
    lines=(str(dname)+","+str(i)+","+str(cv)+","+str(ii)+","+str(round(np.mean(accuracy),15))+","+str(round(np.mean(accuracy_b),15))+","+str(round(np.mean(precision),15))+","+ str(round(np.mean(recall),15))+","+str(round(np.mean(f1),15))+","+str(round(np.mean(kappa),15))+","+str(round(np.mean(train_time),15))+","+str(round(np.mean(test_time),15))+","+str(altime)+"\n")
    return lines,class_based_results


# # isolated training and test data

# In[8]:


def ML_isolated(loop1,loop2,output_csv,cols,step,x,dname):
    graph_on_off=False
    #graph_on_off=False
    print ('%-15s %-3s %-3s %-6s  %-5s %-5s %-5s %-5s %-8s %-5s %-8s %-8s%-8s%-8s'%
               ("Dataset","T","CV","ML alg","Acc","b_Acc","Prec", "Rec" , "F1", "kap" ,"tra-T","test-T","total","alg-time"))
    ths = open(output_csv, "w")
    ths.write("Dataset,T,CV,ML algorithm,Acc,b_Acc,Precision, Recall , F1-score, kappa ,tra-Time,test-Time,Alg-Time\n")
    repetition=10
    fold=1

    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import Normalizer
    
    for ii in ml_list:
        class_based_results=pd.DataFrame()#"" #pd.DataFrame(0, index=np.arange((len(target_names)+3)), columns=["f1-score","precision","recall","support"])
        cm=pd.DataFrame()
        cv=0
        for i in range(repetition):



            #TRAIN
            df = pd.read_csv(loop1,usecols=cols)
            #m_train=df["MAC"]
            #del df["MAC"]
            X_train =df[df.columns[0:-1]]
            X_train=np.array(X_train)
            df[df.columns[-1]] = df[df.columns[-1]].astype('category')
            y_train=df[df.columns[-1]].cat.codes  

            #TEST
            df = pd.read_csv(loop2,usecols=cols)
            df = shuffle(df)
            #m_test=df["MAC"]
            #del df["MAC"]
            X_test =df[df.columns[0:-1]]
            X_test=np.array(X_test)
            df[df.columns[-1]] = df[df.columns[-1]].astype('category')
            y_test=df[df.columns[-1]].cat.codes



            #dname=loop1  [10:-17]  
            results_y=[]
            cv+=1
            results_y.append(y_test)


            precision=[]
            recall=[]
            f1=[]
            accuracy=[]
            train_time=[]
            test_time=[]
            total_time=[]
            kappa=[]
            accuracy_b=[]

                #machine learning algorithm is applied in this section
            clf = ml_list[ii]#choose algorithm from ml_list dictionary
            second=time.time()
            clf.fit(X_train, y_train)
            train_time.append(float((time.time()-second)) )
            second=time.time()
            predict =clf.predict(X_test)
            test_time.append(float((time.time()-second)) )

            altime=-0
                
            lines,class_based_results=score(altime,train_time,test_time,predict,y_test,class_based_results,i,cv,dname,ii)

            
    
            df_cm = pd.DataFrame(confusion_matrix(y_test, predict))
            results_y.append(predict)
            if cm.empty:
                cm =df_cm
            else:
                cm = cm.add(df_cm, fill_value=0)

            ths.write (lines)
            
        if graph_on_off:
            print(class_based_results/(repetition*fold)) 
            graph_name=output_csv[:-4]+".pdf"     
            plt.figure(figsize = (20,14))
            sns.heatmap(cm,xticklabels=target_names, yticklabels=target_names,fmt='', annot=True)
            plt.savefig(graph_name,bbox_inches='tight')#, dpi=400)
            plt.show()
            #print(cm)
            print("\n\n\n") 



    ths.close()  
    print()


# In[12]:


def feature_names():
    features=['pck_size', 'Ether_type', 'LLC_dsap', 'LLC_ssap', 'LLC_ctrl',
           'EAPOL_version', 'EAPOL_type', 'EAPOL_len', 'IP_version', 'IP_ihl',
           'IP_tos', 'IP_len', 'IP_flags', 'IP_Z', 'IP_MF', 
           'IP_DF', 'IP_frag', 'IP_ttl', 'IP_proto', 'IP_options', 
           'ICMP_type', 'ICMP_code',  'ICMP_seq',
           'ICMP_ts_ori', 'ICMP_ts_rx', 'ICMP_ts_tx', 'ICMP_ptr', 'ICMP_reserved',
           'ICMP_length', 'ICMP_nexthopmtu', 'ICMP_unused',
            'TCP_dataofs', 'TCP_reserved', 'TCP_flags',
           'TCP_FIN', 'TCP_SYN', 'TCP_RST', 'TCP_PSH', 'TCP_ACK', 'TCP_URG',
           'TCP_ECE', 'TCP_CWR', 'TCP_window',  'TCP_urgptr',
           'TCP_options', 'UDP_len', 'DHCP_options', 'BOOTP_op',
           'BOOTP_htype', 'BOOTP_hlen', 'BOOTP_hops',  'BOOTP_secs',
           'BOOTP_flags', 'BOOTP_sname', 'BOOTP_file', 'BOOTP_options',
           'DNS_length',  'DNS_qr', 'DNS_opcode', 'DNS_aa', 'DNS_tc',
           'DNS_rd', 'DNS_ra', 'DNS_z', 'DNS_ad', 'DNS_cd', 'DNS_rcode',
           'DNS_qdcount', 'DNS_ancount', 'DNS_nscount', 'DNS_arcount','entropy', 'Protocol',
           "Label"]
    iden=[ 'IP_id','ICMP_chksum', 'ICMP_id','TCP_seq', 'TCP_ack','TCP_chksum', 'UDP_chksum','DNS_id','BOOTP_xid', 'sport','dport','TCP_sport', 'TCP_dport', 'UDP_sport', 'UDP_dport','sport_class', 'dport_class']
    return iden,features


    


# In[13]:


folder("isolated")


# In[14]:


test='Aalto_test_IoTDevID.csv'
train='Aalto_train_IoTDevID.csv'

iden,features=feature_names()

ml_list={"DT":DecisionTreeClassifier()}
step=1
flexible=0
i="0 empty"
output_csv="./isolated/DT_"+i+".csv"
target_names=target_name(test)
ML_isolated(train,test,output_csv,features,step,flexible,i)   


for ii,i in enumerate(iden):
    temp=[]
    temp=features
    temp.insert(0,i)
    output_csv="./isolated/DT_"+i+".csv"
    target_names=target_name(test)
    ML_isolated(train,test,output_csv,temp,step,flexible,i)   
    temp.remove(i)
i="z all"
output_csv="./isolated/DT_"+i+".csv"
target_names=target_name(test)
ML_isolated(train,test,output_csv,iden+features,step,flexible,i)   



# # Crossvalidated data
# ### to equalize the size by making 2 times 5 folds to strengthen comparability

# In[15]:


folder("crossval")


# In[16]:


test='Aalto_test_IoTDevID.csv'
train='Aalto_train_IoTDevID.csv'


# In[17]:


df1 = pd.read_csv(test)
df2 = pd.read_csv(train)
frames = [df1, df2]
df = pd.concat(frames)
df.to_csv('Temp.csv', index=False)


# In[18]:


def ML_CV(loop1,loop2,output_csv,cols,step,x,dname):
    fold=5
    ths = open(output_csv, "w")
    ths.write("Dataset,T,CV,ML_algorithm,Acc,b_Acc,Precision, Recall , F1-score, kappa ,tra-Time,test-Time,total-Time,Al-Time\n")
    repetition=2


    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import Normalizer
    print ('%-15s %-3s %-3s %-6s  %-5s %-5s %-5s %-5s %-5s %-5s %-8s %-8s%-8s%-8s'%
               ("Dataset","T","CV","ML_alg","Acc","b_Acc","Prec", "Rec" , "F1", "kap" ,"tra-T","test-T","total","al-time"))

    for ii in ml_list:
        class_based_results=pd.DataFrame()#"" #pd.DataFrame(0, index=np.arange((len(target_names)+3)), columns=["f1-score","precision","recall","support"])
        cm=pd.DataFrame()
        cv=0
        for i in range(repetition):
            rnd = random()
            kfold = KFold(fold, True, int(rnd*100))  
            cv=0
            df = pd.read_csv(loop1,usecols=cols)#,header=None )
            #del df["MAC"] # if dataset has MAC colomn please uncomment this line
            X =df[df.columns[0:-1]]
            X=np.array(X)
            df[df.columns[-1]] = df[df.columns[-1]].astype('category')
            y=df[df.columns[-1]].cat.codes  
            #scaler = Normalizer().fit(X)
            #X = scaler.transform(X)
            # summarize transformed data
            #dname=loop1[7:-4]
            X.shape
            for train_index, test_index in kfold.split(X):

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]  

      


                #dname=loop1  [6:-13]  
                results_y=[]
                cv+=1
                results_y.append(y_test)


                precision=[]
                recall=[]
                f1=[]
                accuracy=[]
                train_time=[]
                test_time=[]
                total_time=[]
                kappa=[]
                accuracy_b=[]

                    #machine learning algorithm is applied in this section
                clf = ml_list[ii]#choose algorithm from ml_list dictionary
                second=time.time()
                clf.fit(X_train, y_train)
                train_time.append(float((time.time()-second)) )
                second=time.time()
                predict =clf.predict(X_test)
                test_time.append(float((time.time()-second)) )
    
                altime=0
                lines,class_based_results=score(altime,train_time,test_time,predict,y_test,class_based_results,i,cv,dname,ii)

                ths.write (lines)
    print("")






    ths.close()  


# In[19]:


iden


# In[20]:


test=''
train='Temp.csv'

iden,features=feature_names()

ml_list={"DT":DecisionTreeClassifier()}
step=1
flexible=0
i="0 empty"
output_csv="./crossval/DT_"+i+".csv"
target_names=target_name(train)
ML_CV(train,test,output_csv,features,step,flexible,i)   


for ii,i in enumerate(iden):
    temp=[]
    temp=features
    temp.insert(0,i)
    output_csv="./crossval/DT_"+i+".csv"
    target_names=target_name(train)
    ML_CV(train,test,output_csv,temp,step,flexible,i)   
    temp.remove(i)
i="z all"
output_csv="./crossval/DT_"+i+".csv"
target_names=target_name(train)
ML_CV(train,test,output_csv,iden+features,step,flexible,i)   



# # Taking the average for comparison and displaying the results on the graph.

# In[21]:


import matplotlib.pylab as pylab
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[22]:


def average_values(name_list):
    flag=1
    for i in name_list:
        df = pd.read_csv(i) 
        col=i[14:-4]
        temp=pd.DataFrame(df.mean(),columns=[col])
        if flag:
            std=temp
            flag=0
        else:
            std[col]=temp[col]
    tt=std.T
    return tt        


# ## isolated

# In[23]:


name_list=find_the_way('./isolated/','.csv')
iso=average_values(name_list)

iso


# ## crossvalidated

# In[24]:


name_list=find_the_way('./crossval/','.csv')
cv=average_values(name_list)

cv


# In[27]:


etiket=['Primary',
 'BOOTP_xid',
 'DNS_id',
 'Dest. ports',
 "dport_class",
 'ICMP_chksum',
 'ICMP_id',
 'IP_id',
 'Source ports',
 "sport_class",
 'TCP_ack',
 'TCP_chksum',
 'TCP_dport',
 'TCP_seq',
 'TCP_sport',
 'UDP_chksum',
 'UDP_dport',
 'UDP_sport',
 'All together']


# In[29]:


graph_name="Comparison of isolated and CV methods.pdf"
my_xticks=etiket#list(iso.index)
import matplotlib.pylab as pylab



sns.set_style("whitegrid")
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
#plt.figure(figsize=(10,10))
#plt.plot(my_xticks,iso['Acc'], linestyle='--', marker='.', color='b',label= "Separate Train & Test acc")
#plt.plot(my_xticks,cv['Acc'], linestyle='--', marker='.', color='r',label= "10-Fold CV acc")
plt.plot(my_xticks,iso[' F1-score'], linestyle='-', marker='o', color='g',label= "Isolated train & test")
plt.plot(my_xticks,cv[' F1-score'], linestyle='-', marker='o', color='b',label= "5-Fold CV")
plt.axhline(0.673052, color='r',label= "Primary feature list")
plt.title("Comparison of isolated data and merged-cross-validated data result according to features")
plt.legend(numpoints=1)
#plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.ylabel("F1 Score")
plt.xticks(rotation=90) 
#plt.ylim([0.69, 0.71]) 
plt.savefig(graph_name,bbox_inches='tight',format="pdf")#, dpi=400)

