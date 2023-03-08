#!/usr/bin/env python
# coding: utf-8

# # automates the tshark filter rules within a CSV file using python.

# In[1]:


import os
import pandas as pd


# In[3]:


import os
def find_the_way(path,file_format):
    files_add = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                files_add.append(os.path.join(r, file))  
    return files_add
files_add=find_the_way('./','.ipynb')
files_add


# In[4]:


from tqdm import tqdm


# In[6]:


ara1="KF"
for i in (files_add):
    try:
        with open(i, "r", encoding="utf8") as file:
      
            line=file.read()
            if ara1 in line:
                print(i)
    except:pass

files_add=find_the_way('./','.py')
files_add


# In[4]:


from tqdm import tqdm


# In[6]:


ara1="tshark"

for i in (files_add):
    try:
        with open(i, "r", encoding="utf8") as file:
      
            line=file.read()
            if ara1 in line:
                print(i)
    except:pass


    

