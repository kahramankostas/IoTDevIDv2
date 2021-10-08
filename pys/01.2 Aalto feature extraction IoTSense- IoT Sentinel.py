
# coding: utf-8

# # This file extracts feature sets from pcap files.

# ### You can access the  Aalto University IoT devices captures  data used in our study from the link below.
# 
# 
# 
# 
# [ Aalto University IoT devices captures ](https://research.aalto.fi/en/datasets/iot-devices-captures)

# ------------------

# ###  importing relevant libraries

# In[1]:



from scapy.all import*
import math
import pandas as pd
import os
import numpy as np
import zipfile


# In[2]:


def folder(f_name): #this function creates a folder named "attacks" in the program directory.
    try:
        if not os.path.exists(f_name):
            os.makedirs(f_name)
    except OSError:
        print ("The folder could not be created!")


# In[3]:


path="captures_IoT_Sentinel.zip"
with zipfile.ZipFile(path, 'r') as zip_ref:
    zip_ref.extractall("./")
path="./captures_IoT-Sentinel\\"


# ### Discovering pcap extension files under "pcaps" folder.

# In[4]:


def find_the_way(path,file_format):
    files_add = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                files_add.append(os.path.join(r, file))  
    return files_add
files_add=find_the_way(path,'.pcap')
files_add


# In[5]:


train=[]
test=[]

for ii, i in enumerate(files_add):
    print(ii,i)
    if ii%5==0:
        test.append(i)
    else:
        train.append(i)


# ### Port numbers are classified in this part as:
# 
# | Port Numbers | Equivalents |
# | :------ | :------ |
# |No port| 0|
# |Well known ports (between 0 and 1023) |1|
# |Rregistered ports (between 1024 and 49151)  |2|
# |Dynamic ports (between  49152 and 65535) |3|
# # ↓ 

# In[6]:


def port_class(port):
    if 0 <= port <= 1023:
        return 1
    elif  1024 <= port <= 49151 :
        return 2
    elif 49152 <=port <= 65535 :
        return 3
    else:
        return 0


# ### List of pcap files to be processed

# In[7]:


files_add


# ### The dictionary to be used for MAC address and device matching.
# #### Datasets, their MAC addresses and Devices are given separately.

# In[8]:


MAC_list={
# UNSW IEEE TMC 2018 Data MAC and Device names
"d0:52:a8:00:67:5e":"Smart Things",
"44:65:0d:56:cc:d3":"Amazon Echo",
"70:ee:50:18:34:43":"Netatmo Welcome",
"f4:f2:6d:93:51:f1":"TP-Link Day Night Cloud camera",
"00:16:6c:ab:6b:88":"Samsung SmartCam",
"30:8c:fb:2f:e4:b2":"Dropcam",
"00:62:6e:51:27:2e":"Insteon Camera",
"e8:ab:fa:19:de:4f":"unknown maybe cam",
"00:24:e4:11:18:a8":"Withings Smart Baby Monitor",
"ec:1a:59:79:f4:89":"Belkin Wemo switch",
"50:c7:bf:00:56:39":"TP-Link Smart plug",
"74:c6:3b:29:d7:1d":"iHome",
"ec:1a:59:83:28:11":"Belkin wemo motion sensor",
"18:b4:30:25:be:e4":"NEST Protect smoke alarm",
"70:ee:50:03:b8:ac":"Netatmo weather station",
"00:24:e4:1b:6f:96":"Withings Smart scale",
"74:6a:89:00:2e:25":"Blipcare Blood Pressure meter",
"00:24:e4:20:28:c6":"Withings Aura smart sleep sensor",
"d0:73:d5:01:83:08":"Light Bulbs LiFX Smart Bulb",
"18:b7:9e:02:20:44":"Triby Speaker",
"e0:76:d0:33:bb:85":"PIX-STAR Photo-frame",
"70:5a:0f:e4:9b:c0":"HP Printer",
"08:21:ef:3b:fc:e3":"Samsung Galaxy Tab",
"30:8c:fb:b6:ea:45":"Nest Dropcam",
"40:f3:08:ff:1e:da":"Android Phone",
"74:2f:68:81:69:42":"Laptop",
"ac:bc:32:d4:6f:2f":"MacBook",
"b4:ce:f6:a7:a3:c2":"Android Phone",
"d0:a6:37:df:a1:e1":"IPhone",
"f4:5c:89:93:cc:85":"MacBook/Iphone",
"14:cc:20:51:33:ea":"TPLink Router Bridge LAN (Gateway)",
# Yourthings Data MAC and Device names
'00:01:c0:18:7f:9b': 'Gateway',
 '00:04:4b:55:f6:4f': 'nVidiaShield',
 '00:12:16:ab:c0:22': 'ChineseWebcam',
 '00:17:88:21:f7:e4': 'PhilipsHUEHub',
 '00:1d:c9:23:f6:00': 'RingDoorbell',
 '00:21:cc:4d:59:35': 'Wink2Hub',
 '00:24:e4:2b:a5:34': 'WithingsHome',
 '00:7e:56:77:35:4d': 'KoogeekLightbulb',
 '08:05:81:ee:06:46': 'Roku4',
 '08:86:3b:6f:7a:15': 'BelkinWeMoMotionSensor',
 '08:86:3b:70:d7:39': 'BelkinWeMoSwitch',
 '0c:47:c9:4e:fe:5b': 'AmazonFireTV',
 '10:ce:a9:eb:5a:8a': 'BoseSoundTouch10',
 '18:b4:30:31:04:b9': 'NestProtect',
 '18:b4:30:40:1e:c5': 'NestGuard',
 '18:b4:30:58:3d:6c': 'NestCamera',
 '18:b4:30:8c:03:e4': 'NestCamIQ',
 '20:df:b9:20:87:39': 'GoogleHomeMini',
 '30:52:cb:a3:4f:5f': 'RokuTV',
 '3c:f7:a4:f2:15:87': 'iPhone',
 '40:9f:38:92:40:13': 'Roomba',
 '44:73:d6:01:3d:fd': 'LogitechLogiCircle',
 '48:d6:d5:98:53:84': 'GoogleHome',
 '50:c7:bf:92:a6:4a': 'TP-LinkSmartWiFiLEDBulb',
 '54:4a:16:f9:54:18': 'InsteonHub',
 '5c:aa:fd:6c:e0:d4': 'Sonos',
 '64:52:99:97:f8:40': 'ChamberlainmyQGarageOpener',
 '74:c2:46:1b:8e:e2': 'AmazonEchoGen1',
 '7c:64:56:60:71:74': 'SamsungSmartTV',
 '7c:70:bc:5d:09:d1': 'Canary',
 '94:10:3e:5c:2e:31': 'BelkinWeMoCrockpot',
 '94:10:3e:cc:67:95': 'BelkinWeMoLink',
 '94:4a:0c:08:7e:72': 'MiCasaVerdeVeraLite',
 'a4:f1:e8:8d:b0:9e': 'AndroidTablet',
 'ac:3f:a4:70:4a:d6': 'PiperNV',
 'b0:4e:26:20:15:8a': 'TP-LinkWiFiPlug',
 'b0:7f:b9:a6:47:4d': 'NetgearArloCamera',
 'b0:c5:54:03:c7:09': 'D-LinkDCS-5009LCamera',
 'b4:79:a7:22:f9:fc': 'WinkHub',
 'c0:56:27:53:09:6d': 'BelkinNetcam',
 'c8:db:26:02:bb:bb': 'LogitechHarmonyHub',
 'cc:b8:a8:ad:4d:04': 'AugustDoorbellCam',
 'd0:03:4b:39:12:e3': 'AppleTV(4thGen)',
 'd0:52:a8:63:47:9e': 'SamsungSmartThingsHub',
 'd0:73:d5:12:84:d1': 'LIFXVirtualBulb',
 'd4:90:9c:cc:62:42': 'AppleHomePod',
 'd8:f7:10:c2:29:be': 'HarmonKardonInvoke',
 'e4:71:85:25:ce:ec': 'SecurifiAlmond',
 'e8:b2:ac:af:62:0f': 'iPad',
 'f4:5e:ab:5e:c0:23': 'CasetaWirelessHub',
 'f4:f2:6d:ce:9a:5d': 'GoogleOnHub',
# IoT devices captures MAC and Device names
'00:17:88:24:76:ff': 'Hue-Device',
 '00:1a:22:03:cb:be': 'MAXGateway',
 '00:1a:22:05:c4:2e': 'HomeMaticPlug',
 '00:24:e4:24:80:2a': 'Withings',
 '00:b5:6d:06:08:ba': 'unknown',
 '1c:5f:2b:aa:fd:4e': 'D-LinkDevice',
 '20:f8:5e:ca:91:52': 'Aria',
 '24:77:03:7c:ea:dc': 'unknown',
 '28:b2:bd:c3:41:79': 'unknown',
 '38:0b:40:ef:85:41': 'unknown',

 '50:c7:bf:00:c7:03': 'TP-LinkPlugHS110',
 '50:c7:bf:00:fc:a3': 'TP-LinkPlugHS100',
 '3c:49:37:03:17:db': 'EdnetCam',
 '3c:49:37:03:17:f0': 'EdnetCam',
 '5c:cf:7f:06:d9:02': 'iKettle2',
 '5c:cf:7f:07:ae:fb': 'SmarterCoffee',
 '6c:72:20:c5:17:5a': 'D-LinkWaterSensor',
 '74:da:38:23:22:7b': 'EdimaxPlug2101W',
 '74:da:38:4a:76:49': 'EdimaxPlug1101W',
 '74:da:38:80:79:fc': 'EdimaxCam',
 '74:da:38:80:7a:08': 'EdimaxCam',
 '84:18:26:7b:5f:6b': 'Lightify',
 '90:8d:78:a8:e1:43': 'D-LinkSensor',
 '90:8d:78:a9:3d:6f': 'D-LinkSwitch',
 '90:8d:78:dd:0d:60': 'D-LinkSiren',
 '94:10:3e:34:0c:b5': 'WeMoSwitch',
 '94:10:3e:35:01:c1': 'WeMoSwitch',
 '94:10:3e:41:c2:05': 'WeMoInsightSwitch',
 '94:10:3e:42:80:69': 'WeMoInsightSwitch',
 '94:10:3e:cd:37:65': 'WeMoLink',
 'ac:cf:23:62:3c:6e': 'EdnetGateway',
 'b0:c5:54:1c:71:85': 'D-LinkDayCam',
 'b0:c5:54:25:5b:0e': 'D-LinkCam',
 'bc:f5:ac:f4:c0:9d': 'unknown'}


# ### Calculating the payload entropy value.
# 
# 
# # ↓ 

# In[9]:


def pre_entropy(payload):
    
    characters=[]
    for i in payload:
            characters.append(i)
    return shannon(characters)


def shannon(data):
    freq_dict={} 
    for i in data:
        if i in freq_dict:
            freq_dict[i] += 1
        else:
            freq_dict[i] = 1    
    entropy = 0.0
    logarithm_base = 2
    payload_size = len(data) #
    for key in freq_dict.keys():
        frequency = float(freq_dict[key])/payload_size
        if frequency > 0: 
            entropy = entropy + frequency * math.log(frequency, logarithm_base)
    return -entropy


# In[10]:


import time 
dataset_name=["Train.csv", "Test.csv"]
for numero,dataset in enumerate ([train,test]):
    count=0
    ths = open(dataset_name[numero], "w")
    header="ARP,LLC,EAPOL,IP,ICMP,ICMP6,TCP,UDP,TCP_w_size,HTTP,HTTPS,DHCP,BOOTP,SSDP,DNS,MDNS,NTP,IP_padding,IP_add_count,IP_ralert,Portcl_src,Portcl_dst,Pck_size,Pck_rawdata,payload_l,Entropy,Label,MAC,Folder,Session\n"
    ths.write(header)  
    dst_ip_list={}


    files_add=dataset
    for i in files_add:
        filename=str(i)
        filename=filename.replace("\\","/")
        #x = filename.rfind("/")
        filename=filename.split("/")
        print(filename)


        for ii in MAC_list:
            dst_ip_list[ii]=[]
       


        pkt = rdpcap(i)
        print("\n\n"+"========"+ i[8:]+"========"+"\n" )
        print(pkt)
        for jj,j in enumerate(pkt):       
            ip_add_count=0
            layer_2_arp = 0
            layer_2_llc = 0

            layer_3_eapol = 0        
            layer_3_ip = 0
            layer_3_icmp = 0
            layer_3_icmp6 = 0



            layer_4_tcp = 0
            layer_4_udp = 0
            layer_4_tcp_ws=0


            layer_7_http = 0
            layer_7_https = 0
            layer_7_dhcp = 0
            layer_7_bootp = 0
            layer_7_ssdp = 0
            layer_7_dns = 0
            layer_7_mdns = 0
            layer_7_ntp = 0

            ip_padding = 0
            ip_ralert = 0


            port_class_src = 0
            port_class_dst = 0

            pck_size = 0
            pck_rawdata = 0
            entropy=0


            layer_4_payload_l=0




            try:
                pck_size=j.len
            except:pass

            try:
                if j[IP]:
                    layer_3_ip = 1     
                temp=str(j[IP].dst)
                if temp not in dst_ip_list[j.src]:
                    dst_ip_list[j.src].append(temp)
                ip_add_count=len(dst_ip_list[j.src])
                port_class_src = port_class(j[IP].sport)
                port_class_dst = port_class(j[IP].dport)

            except:pass 

            temp=str(j.show)
            if "ICMPv6" in temp:
                layer_3_icmp6 = 1

            try:
                if j[IP].ihl >5:
                    if IPOption_Router_Alert(j):
                        pad=str(IPOption_Router_Alert(j).show)
                        if "Padding" in pad:
                            ip_padding=1
                        ip_ralert = 1     
            except:pass 

            if j.haslayer(ICMP):
                layer_3_icmp = 1  


            if j.haslayer(Raw):
                pck_rawdata = 1   

            if j.haslayer(UDP):

                layer_4_udp = 1
                if j[UDP].sport==68 or j[UDP].sport==67:
                    layer_7_dhcp = 1
                    layer_7_bootp = 1
                if j[UDP].sport==53 or j[UDP].dport==53:
                    layer_7_dns = 1      
                if j[UDP].sport==5353 or j[UDP].dport==5353:
                    layer_7_mdns = 1                    
                if j[UDP].sport==1900 or j[UDP].dport==1900:
                    layer_7_ssdp = 1                    
                if j[UDP].sport==123 or j[UDP].dport==123:
                    layer_7_ntp = 1                    

            try:
                if j[UDP].payload:
                    layer_4_payload_l=len(j[UDP].payload)
            except:pass  



            if j.haslayer(TCP):
                layer_4_tcp = 1
                layer_4_tcp_ws=j[TCP].window
                if j[TCP].sport==80 or j[TCP].dport==80:
                    layer_7_http = 1      
                if j[TCP].sport==443 or j[TCP].dport==443:
                    layer_7_https = 1  
                try:
                    if j[TCP].payload:
                        layer_4_payload_l=len(j[TCP].payload)
                except:pass  

            if j.haslayer(ARP):
                layer_2_arp = 1                                 

            if j.haslayer(LLC):
                layer_2_llc = 1                             

            if j.haslayer(EAPOL):        
                layer_3_eapol = 1                                 
            try:
                entropy=pre_entropy(j[Raw].original)
            except:pass
            label=MAC_list[j.src]
            line=[layer_2_arp, layer_2_llc, layer_3_eapol, layer_3_ip, layer_3_icmp, layer_3_icmp6, layer_4_tcp, layer_4_udp, layer_4_tcp_ws, layer_7_http, layer_7_https, layer_7_dhcp, layer_7_bootp, layer_7_ssdp, layer_7_dns, layer_7_mdns, layer_7_ntp, ip_padding, ip_add_count, ip_ralert, port_class_src, port_class_dst, pck_size, pck_rawdata,layer_4_payload_l,entropy, label,j.src,filename[2],filename[3][:-5]]  
            line=str(line).replace("[","")
            line=str(line).replace("]","")
            line=str(line).replace(", ",",")
            line=str(line).replace("\'","")
            Mac=j.src


            if label=="unknown" and j.dst in ['3c:49:37:03:17:db', '3c:49:37:03:17:f0', '5c:cf:7f:06:d9:02', '5c:cf:7f:07:ae:fb'] :
                label=MAC_list[j.dst] # çift yön eklemesi
                Mac=j.dst


            if label!="unknown":
                ths.write(str(line)+"\n")  

    ths.close()          


# # Creation of three different feature sets

# ### Generates  IoTSentinel, IoTSense feature sets

# In[11]:


Folder= {'Aria': 'Aria', 'D-LinkCam': 'D-LinkCam', 'D-LinkDayCam': 'D-LinkDayCam', 'D-LinkDoorSensor': 'D-LinkDoorSensor', 'D-LinkHomeHub': 'D-LinkHomeHub', 'D-LinkSensor': 'D-LinkSensor', 'D-LinkSiren': 'D-LinkSiren', 'D-LinkSwitch': 'D-LinkSwitch', 'D-LinkWaterSensor': 'D-LinkWaterSensor', 'EdimaxCam1': 'EdimaxCam', 'EdimaxCam2': 'EdimaxCam', 'EdimaxPlug1101W': 'EdimaxPlug1101W', 'EdimaxPlug2101W': 'EdimaxPlug2101W', 'EdnetCam1': 'EdnetCam', 'EdnetCam2': 'EdnetCam', 'EdnetGateway': 'EdnetGateway', 'HomeMaticPlug': 'HomeMaticPlug', 'HueBridge': 'HueBridge', 'HueSwitch': 'HueSwitch', 'iKettle2': 'iKettle2', 'Lightify': 'Lightify', 'MAXGateway': 'MAXGateway', 'SmarterCoffee': 'SmarterCoffee', 'TP-LinkPlugHS100': 'TP-LinkPlugHS100', 'TP-LinkPlugHS110': 'TP-LinkPlugHS110', 'WeMoInsightSwitch': 'WeMoInsightSwitch', 'WeMoInsightSwitch2': 'WeMoInsightSwitch', 'WeMoLink': 'WeMoLink', 'WeMoSwitch': 'WeMoSwitch', 'WeMoSwitch2': 'WeMoSwitch', 'Withings': 'Withings'}


# In[13]:


for i in dataset_name:
    df=pd.read_csv(i)
    df=df.replace({"Folder": Folder})
    del df["Label"]
    df["Label"]=df["Folder"]
    del df["Folder"]
    del df["Session"]
    
    
    
    deleted=["payload_l","TCP_w_size","Entropy"]
    temp=df
    name="IoTSentinel_"+i
    temp=temp.drop(columns=deleted)
    temp.to_csv(name, index=False)


    deleted=["LLC","Pck_rawdata","Pck_size","Portcl_src","Portcl_dst","IP_add_count"]
    temp=df
    name="IoTSense_"+i
    temp=temp.drop(columns=deleted)
    temp.to_csv(name, index=False)   
    
os.remove("Train.csv")
os.remove("Test.csv")

