
# coding: utf-8

# In[30]:


import csv
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn
import time


# In[31]:


import warnings
warnings.filterwarnings("ignore")


# # create binary datasets

# In[32]:


def folder(f_name): #this function creates a folder.
    try:
        if not os.path.exists(f_name):
            os.makedirs(f_name)
    except OSError:
        print ("The folder could not be created!")


# In[33]:


def target_name(name):
    df = pd.read_csv(name,usecols=["Label"])
    target_names=sorted(list(df["Label"].unique()))
    target_names = list(map(str, target_names))
    #print(target_names)
    return target_names
train="./Aalto_train_IoTDevID.csv"
device_names=target_name(train)


# In[34]:


folder("devicebasedcsvs")
folder("tree")
folder("results")


# In[35]:



for device in device_names:
    print(device)
    df=pd.read_csv(train)
    #print(df.groupby("Label").size())
    k= df['Label'].values==device
    del df["MAC"]
    k=list(k*1)
    size=k.count(1)
    df['Label']=k
    if size>1000:
        size=1000
    dev=df[df["Label"]==1]
    notdev=df[df["Label"]==0]
    dev=dev.sample(n=size, random_state=1)
    notdev=notdev.sample(n=size*5, random_state=1)
    df = pd.concat([dev, notdev])
    #df=df.groupby('Label').apply(lambda x: x.sample(n=size)).reset_index(drop = True)
    df.to_csv("./devicebasedcsvs/"+device+".csv",  index=False)


# # create decision tree diagrams

# In[36]:


from sklearn import tree
import graphviz
from graphviz import render

def ciz(name,model,feature_names,target_names):

    dot_data = tree.export_graphviz(model, out_file=None, 
                                feature_names=feature_names,  
                                class_names=target_names,
                                filled=True)

    # Draw graph
    #graph = graphviz.Source(dot_data) 
    graph = graphviz.Source(dot_data,format='pdf')    
    name="./tree/"+name[18:-4]
    graph.render(name, view=True)  


# # ML Application

# In[37]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from random import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier


# In[38]:


def find_the_way(path,file_format):
    files_add = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                files_add.append(os.path.join(r, file))  
    return files_add
name_list=find_the_way('./devicebasedcsvs','.csv')
name_list


# In[39]:


ml_list={"DT":DecisionTreeClassifier()}


# In[40]:


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
    print ('%-15s %-3s %-3s %-6s  %-5s %-5s %-5s %-5s %-5s %-5s %-8s %-8s%-8s%-8s' % (dname,i,cv,ii[0:6],str(round(np.mean(accuracy),2)),str(round(np.mean(accuracy_b),2)),
        str(round(np.mean(precision),2)), str(round(np.mean(recall),2)),str(round(np.mean(f1),2)), 
        str(round(np.mean(kappa),2)),str(round(np.mean(train_time),2)),str(round(np.mean(test_time),2)),str(round(np.mean(test_time)+np.mean(train_time),2)),str(round(np.mean(altime),2))))
    lines=(str(dname)+","+str(i)+","+str(cv)+","+str(ii)+","+str(round(np.mean(accuracy),15))+","+str(round(np.mean(accuracy_b),15))+","+str(round(np.mean(precision),15))+","+ str(round(np.mean(recall),15))+","+str(round(np.mean(f1),15))+","+str(round(np.mean(kappa),15))+","+str(round(np.mean(train_time),15))+","+str(round(np.mean(test_time),15))+","+str(round(np.mean(test_time)+np.mean(train_time),15))+","+str(altime)+"\n")
    return lines,class_based_results


# In[41]:


def ML(loop1,loop2,output_csv,cols,step,Tree):
    fold=10
    ths = open(output_csv, "w")
    ths.write("Dataset,T,CV,ML_algorithm,Acc,b_Acc,Precision, Recall , F1-score, kappa ,tra-Time,test-Time,total-Time,Al-Time\n")
    repetition=1


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
            feature_names=df.columns
            feature_names=feature_names[0:-1]
            X =df[df.columns[0:-1]]
            X=np.array(X)
            df[df.columns[-1]] = df[df.columns[-1]].astype('category')
            y=df[df.columns[-1]].cat.codes  
            #scaler = Normalizer().fit(X)
            #X = scaler.transform(X)
            # summarize transformed data
            dname=loop1[18:-4]
            X.shape
            for train_index, test_index in kfold.split(X):

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]  

      


                #dname=loop1  [6:-13]  
                results_y=[]
                
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
                
                
                if Tree:
                    if cv==1:# creates a decision tree for the first step of cross-validation.
                        ciz(loop1,clf,feature_names,target_names)
                cv+=1
         
                altime=0
                lines,class_based_results=score(altime,train_time,test_time,predict,y_test,class_based_results,i,cv,dname,ii)

                ths.write (lines)






                df_cm = pd.DataFrame(confusion_matrix(y_test, predict))
                results_y.append(predict)
                if cm.empty:
                    cm =df_cm
                else:
                    cm = cm.add(df_cm, fill_value=0)


        print(class_based_results/(repetition*fold)) 
        graph_name=output_csv[:-4]+".pdf"     
        plt.figure(figsize = (5,3.5))
        sns.heatmap(cm,xticklabels=target_names, yticklabels=target_names,fmt='', annot=True)
        #plt.savefig(graph_name,bbox_inches='tight')#, dpi=400)
        plt.show()
        #print(cm)
        print("\n\n\n") 
    ths.close()  


# In[42]:


features=['pck_size', 'Ether_type', 'LLC_dsap', 'LLC_ssap', 'LLC_ctrl',
       'EAPOL_version', 'EAPOL_type', 'EAPOL_len', 'IP_version', 'IP_ihl',
       'IP_tos', 'IP_len', 'IP_flags', 'IP_Z', 'IP_MF', 'IP_DF', 'IP_frag',
       'IP_ttl', 'IP_proto', 'IP_options', 'ICMP_type',
       'ICMP_code', 'ICMP_chksum', 'ICMP_id', 'ICMP_seq', 'ICMP_ts_ori',
       'ICMP_ts_rx', 'ICMP_ts_tx', 'ICMP_ptr', 'ICMP_reserved', 'ICMP_length',
        'ICMP_nexthopmtu', 'ICMP_unused', 'TCP_seq',"IP_id",
       'TCP_ack', 'TCP_dataofs', 'TCP_reserved', 'TCP_flags', 'TCP_FIN',
       'TCP_SYN', 'TCP_RST', 'TCP_PSH', 'TCP_ACK', 'TCP_URG', 'TCP_ECE',
       'TCP_CWR', 'TCP_window', 'TCP_chksum', 'TCP_urgptr', 'TCP_options',
       'UDP_len', 'UDP_chksum', 'DHCP_options', 'BOOTP_op', 'BOOTP_htype',
       'BOOTP_hlen', 'BOOTP_hops', 'BOOTP_xid', 'BOOTP_secs', 'BOOTP_flags',
       'BOOTP_sname', 'BOOTP_file', 'BOOTP_options', 'DNS_length', 'DNS_id',
       'DNS_qr', 'DNS_opcode', 'DNS_aa', 'DNS_tc', 'DNS_rd', 'DNS_ra', 'DNS_z',
       'DNS_ad', 'DNS_cd', 'DNS_rcode', 'DNS_qdcount', 'DNS_ancount',
       'DNS_nscount', 'DNS_arcount', 'sport_class', 'dport_class',
       'sport', 'dport', 'TCP_sport', 'TCP_dport', 'UDP_sport',
       'UDP_dport', 'payload_bytes', 'entropy', 'Protocol', 'Label']


# In[43]:


len(features)


# In[44]:


for i in name_list:
    step=1
    Tree=True # It uses the graphviz library.
    output_csv="./results/"+i[18:-4]+"_"+str(step)+"_"+str(Tree)+".csv"
    target_names=target_name(i)
    ML(i,i,output_csv,features,step,Tree)   


# # voting process for device csv files
# # each voting action is saved in a separate csv file

# In[45]:


from xverse.ensemble import VotingSelector


# In[46]:


for i in name_list:
    df=pd.read_csv(i,usecols=features) 
    X =df[df.columns[0:-1]]
    target_names=sorted(list(df[df.columns[-1]].unique()))
    y=df[df.columns[-1]]

    feature_names=df.columns
    
    
    clf = VotingSelector()
    clf.fit(X, y)
    #Selected features
    temp="./results/"+i[18:-4]+"_FI_.csv"
    fi=clf.feature_importances_
    fi.to_csv(temp, index=None)

    temp="./results/"+i[18:-4]+"_VETO_.csv"
    votes=clf.feature_votes_
    votes.to_csv(temp, index=None)
    print(clf.feature_votes_)


# In[47]:


df.columns


# # calculate the average of the votes

# In[48]:


name_list=find_the_way('./results/','_VETO_.csv')
name_list


# In[49]:


df_add = pd.DataFrame(columns=[ 'Information_Value', 'Random_Forest',
       'Recursive_Feature_Elimination', 'Extra_Trees', 'Chi_Square', 'L_One',
       'Votes'])

flag=1
for i in name_list:
    df = pd.read_csv(i, index_col=0)
    df=df.sort_index()    
    df_add= df_add.add(df, fill_value=0)
    


# In[50]:


df=df_add/27
df=df.sort_values(['Votes'], ascending=[False])
df.to_csv("veto_average_results.csv")


# In[51]:


df


# # Creating the voting process result graph

# In[52]:


data = pd.read_csv("veto_average_results.csv")
new = data[['Variable_Name', 'Votes']].copy()

new


# In[53]:


graph_name="Feature Selection with Voting.PDF"
import seaborn as sns
sns.set_theme(style="whitegrid")

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
plt.figure(figsize=(18,8))
plt.title("Feature Selection with Voting")

plt.ylabel('Voting Average')
plt.xticks(rotation=90) 
ax = sns.barplot(x="Variable_Name",color='b', y="Votes", data=new)
plt.xlabel('Feature Name')
plt.savefig(graph_name,bbox_inches='tight',format="pdf")#, dpi=400)
plt.show()

