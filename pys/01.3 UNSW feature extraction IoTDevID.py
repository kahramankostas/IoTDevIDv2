
# coding: utf-8

# ### You can access the UNSW datasets used in our study from the links below.
# 
# [ UNSW - IOT TRAFFIC TRACES - IEEE TMC 2018](https://iotanalytics.unsw.edu.au/iottraces)
# 
# [ UNSW - IOT BENIGN AND ATTACK TRACES - ACM SOSR 2019](https://iotanalytics.unsw.edu.au/attack-data)
# 

# ### Since these data are very large, we filter the data on a device and session basis. You can access the Pcap files obtained from this filtering process from [ this link (Used Pcap Files)](https://drive.google.com/file/d/1RSnQJNTHj8FoS1KvBxbCGaS4sYmX3sPF/view).
# 
# 
# <img src="unsw.jpg" alt="unsw" width="600"/>
# 

# In[1]:


import pandas as pd
import zipfile
import os
import shutil


# In[75]:


path="UNSW_PCAP.zip"
with zipfile.ZipFile(path, 'r') as zip_ref:
    zip_ref.extractall("./")
path="./UNSW_PCAP/"


# In[10]:


def folder(f_name): #this function creates a folder named "attacks" in the program directory.
    try:
        if not os.path.exists(f_name):
            os.makedirs(f_name)
    except OSError:
        print ("The folder could not be created!")


# In[11]:


def find_the_way(path,file_format):
    files_add = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                files_add.append(os.path.join(r, file))  
    return files_add
#files_add=find_the_way("./",'.pcap')
files_add=find_the_way("./UNSW_PCAP",'.pcap')
files_add


# # PCAP2CSV

# In[12]:


from scapy.all import*


# In[13]:


import math
import pandas as pd
import os
import numpy as np


# In[14]:


def folder(f_name): #this function creates a folder.
    try:
        if not os.path.exists(f_name):
            os.makedirs(f_name)
    except OSError:
        print ("The folder could not be created!")


# In[15]:


MAC_list={
# UNSW IEEE TMC 2018 Data MAC and Device names
"d0:52:a8:00:67:5e":"Smart Things",
"44:65:0d:56:cc:d3":"Amazon Echo",
"70:ee:50:18:34:43":"Netatmo Welcome",
"f4:f2:6d:93:51:f1":"TP-Link Day Night Cloud camera",
"00:16:6c:ab:6b:88":"Samsung SmartCam",
"30:8c:fb:2f:e4:b2":"Dropcam",
"00:62:6e:51:27:2e":"Insteon Camera",
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
"e8:ab:fa:19:de:4f":"unknown maybe cam",
"30:8c:fb:b6:ea:45":"Nest Dropcam",
"40:f3:08:ff:1e:da":"Android Phone 1",
"74:2f:68:81:69:42":"Laptop",
"ac:bc:32:d4:6f:2f":"MacBook",
"b4:ce:f6:a7:a3:c2":"Android Phone 2",
"d0:a6:37:df:a1:e1":"IPhone",
"f4:5c:89:93:cc:85":"MacBook-Iphone",
"14:cc:20:51:33:ea":"TPLink Router Bridge LAN",
"00:24:e4:10:ee:4c":"Withings Baby Monitor 2",
"88:4a:ea:31:66:9d":"Ring Door Bell",
"00:17:88:2b:9a:25":"Phillip Hue Lightbulb",
"7c:70:bc:5d:5e:dc":"Canary Camera",
"6c:ad:f8:5e:e4:61":"Google Chromecast",
"28:c2:dd:ff:a5:2d":"Hello Barbie",
"70:88:6b:10:0f:c6":"Awair air quality monitor",
"b4:75:0e:ec:e5:a9":"Belkin Camera",
"e0:76:d0:3f:00:ae":"August Doorbell Cam",
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
 '3c:49:37:03:17:db': 'EdnetCam',
 '3c:49:37:03:17:f0': 'EdnetCam',
 '50:c7:bf:00:c7:03': 'TP-LinkPlugHS110',
 '50:c7:bf:00:fc:a3': 'TP-LinkPlugHS100',
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


# In[16]:


def shannon(data):
    LOG_BASE = 2
   # We determine the frequency of each byte
   # in the dataset and if this frequency is not null we use it for the
   # entropy calculation
    dataSize = len(data)
    ent = 0.0
    freq={} 
    for c in data:
        if c in freq:
            freq[c] += 1
        else:
            freq[c] = 1
   # to determine if each possible value of a byte is in the list
    for key in freq.keys():
        f = float(freq[key])/dataSize
        if f > 0: # to avoid an error for log(0)
            ent = ent + f * math.log(f, LOG_BASE)
    return -ent

def pre_entropy(payload):
    
    characters=[]
    for i in payload:
            characters.append(i)
    return shannon(characters)
            


# In[17]:


def port_class(port):
    port_list=[0,53,67,68,80,123,443,1900,5353,49153]# private port list (0-Reserved,53-DNS, 67-BOOTP server, 68-BOOTP client...)
    if port in port_list: #Is the port number in the list?
        return port_list.index(port)+1 # return the port's index number in the list (actually with index+1)
    elif 0 <= port <= 1023: # return 11 if the port number is in the range 0 :1023
        return 11
    elif  1024 <= port <= 49151 : # return 12 if the port number is in the range 1024:49151
        return 12
    elif 49152 <=port <= 65535 :# return 13 if the port number is in the range 49152:65535
        return 13
    else:# return 0 if no previous conditions are met
        return 0
    
    
def port_1023(port):
    if 0 <= port <= 1023:
        return port
    elif  1024 <= port <= 49151 :
        return 2
    elif 49152 <=port <= 65535 :
        return 3
    else:
        return 0


# In[18]:


header="pck_size,Ether_type,LLC_dsap,LLC_ssap,LLC_ctrl,EAPOL_version,EAPOL_type,EAPOL_len,IP_version,IP_ihl,IP_tos,IP_len,IP_flags,IP_Z,IP_MF,IP_id,IP_chksum,IP_DF,IP_frag,IP_ttl,IP_proto,IP_options,IP_add_count,ICMP_type,ICMP_code,ICMP_chksum,ICMP_id,ICMP_seq,ICMP_ts_ori,ICMP_ts_rx,ICMP_ts_tx,ICMP_ptr,ICMP_reserved,ICMP_length,ICMP_nexthopmtu,ICMP_unused,TCP_seq,TCP_ack,TCP_dataofs,TCP_reserved,TCP_flags,TCP_FIN,TCP_SYN,TCP_RST,TCP_PSH,TCP_ACK,TCP_URG,TCP_ECE,TCP_CWR,TCP_window,TCP_chksum,TCP_urgptr,TCP_options,UDP_len,UDP_chksum,DHCP_options,BOOTP_op,BOOTP_htype,BOOTP_hlen,BOOTP_hops,BOOTP_xid,BOOTP_secs,BOOTP_flags,BOOTP_sname,BOOTP_file,BOOTP_options,DNS_length,DNS_id,DNS_qr,DNS_opcode,DNS_aa,DNS_tc,DNS_rd,DNS_ra,DNS_z,DNS_ad,DNS_cd,DNS_rcode,DNS_qdcount,DNS_ancount,DNS_nscount,DNS_arcount,sport_class,dport_class,sport23,dport23,sport_bare,dport_bare,TCP_sport,TCP_dport,UDP_sport,UDP_dport,payload_bytes,entropy,MAC,Label\n"

#header="pck_size,Ether_type,LLC_dsap,LLC_ssap,LLC_ctrl,EAPOL_version,EAPOL_type,EAPOL_len,IP_version,IP_ihl,IP_tos,IP_len,IP_flags,IP_frag,IP_ttl,IP_proto,IP_options,IP_add_count,ICMP_type,ICMP_code,ICMP_seq,ICMP_ts_ori,ICMP_ts_rx,ICMP_ts_tx,ICMP_gw,ICMP_ptr,ICMP_reserved,ICMP_length,ICMP_nexthopmtu,ICMP_unused,TCP_dataofs,TCP_reserved,TCP_flags,TCP_window,TCP_urgptr,UDP_len,BOOTP_op,BOOTP_htype,BOOTP_hlen,BOOTP_hops,BOOTP_secs,BOOTP_flags,DNS_length,DNS_qr,DNS_opcode,DNS_aa,DNS_tc,DNS_rd,DNS_ra,DNS_z,DNS_ad,DNS_cd,DNS_rcode,DNS_qdcount,DNS_ancount,DNS_nscount,DNS_arcount,sport,dport,entropy,Label,MAC\n"


# In[19]:


#flags
#TCP
FIN = 0x01
SYN = 0x02
RST = 0x04
PSH = 0x08
ACK = 0x10
URG = 0x20
ECE = 0x40
CWR = 0x80
#IP
Z = 0x00
MF= 0x01
DF= 0x02


# In[20]:




ipf=[]
tcpf=[]
import time
degistir=""
dst_ip_list={}
Ether_adresses=[]
IP_adresses=[]
label_count=0



for numero,i in enumerate (files_add):


        
    filename=i[:-4]+".csv"
    ths = open(filename, "w")
    ths.write(header)  
    

    #header=header
    #ths.write(header)  
    filename=str(i)
    filename=filename.replace("\\","/")
    #x = filename.rfind("/")
    filename=filename.split("/")
    
    #break
    pkt = rdpcap(i)
    #print("\n",numero,"/",len(files_add),"========"+ i[8:]+"========\n" )
    #print("\n",numero,"/",len(files_add))
    sayaç=len(pkt)//20
    for jj, j in enumerate (pkt):
        
        try:        
            if jj%sayaç==0:

                    sys.stdout.write("\r[" + "=" * int(jj//sayaç) +  " " * int((sayaç*20 - jj)// sayaç) + "]" +  str(5*jj//sayaç) + "%")
                    sys.stdout.flush()
        except:pass
        if j.haslayer(ARP):
            continue
        else:
            
            ts=j.time
            try:pck_size=j.len
            except:pck_size=0
            if j.haslayer(Ether):

                if j[Ether].dst not in Ether_adresses:
                    Ether_adresses.append(j[Ether].dst)
                if j[Ether].src not in Ether_adresses:
                    Ether_adresses.append(j[Ether].src)


                Ether_dst=j[Ether].dst#Ether_adresses.index(j[Ether].dst)+1
                Ether_src=j[Ether].src#Ether_adj[Ether].dstresses.index(j[Ether].src)+1


                Ether_type=j[Ether].type
            else:
                Ether_dst=0
                Ether_src=0
                Ether_type=0


            if j.haslayer(ARP):
                ARP_hwtype=j[ARP].hwtype
                ARP_ptype=j[ARP].ptype
                ARP_hwlen=j[ARP].hwlen
                ARP_plen=j[ARP].plen
                ARP_op=j[ARP].op


                ARP_hwsrc=j[ARP].hwsrc
                ARP_psrc=j[ARP].psrc
                ARP_hwdst=j[ARP].hwdst
                ARP_pdst=j[ARP].pdst

                if j[ARP].hwsrc not in Ether_adresses:
                    Ether_adresses.append(j[ARP].hwsrc)
                if j[ARP].psrc not in IP_adresses:
                    IP_adresses.append(j[ARP].psrc)           
                if j[ARP].hwdst not in Ether_adresses:
                    Ether_adresses.append(j[ARP].hwdst)
                if j[ARP].pdst not in IP_adresses:
                    IP_adresses.append(j[ARP].pdst)


                ARP_hwsrc=j[ARP].hwsrc#Ether_adresses.index(j[ARP].hwsrc)+1
                ARP_psrc=j[ARP].psrc#IP_adresses.index(j[ARP].psrc)+1
                ARP_hwdst=j[ARP].hwdst#Ether_adresses.index(j[ARP].hwdst)+1
                ARP_pdst=j[ARP].pdst#IP_adresses.index(j[ARP].pdst)+1


            else:
                ARP_hwtype=0
                ARP_ptype=0
                ARP_hwlen=0
                ARP_plen=0
                ARP_op=0
                ARP_hwsrc=0
                ARP_psrc=0
                ARP_hwdst=0
                ARP_pdst=0            


            if j.haslayer(LLC):
                LLC_dsap=j[LLC].dsap
                LLC_ssap=j[LLC].ssap
                LLC_ctrl=j[LLC].ctrl
            else:
                LLC_dsap=0
                LLC_ssap=0
                LLC_ctrl=0            



            if j.haslayer(EAPOL):
                EAPOL_version=j[EAPOL].version
                EAPOL_type=j[EAPOL].type
                EAPOL_len=j[EAPOL].len

            else:
                EAPOL_version=0
                EAPOL_type=0
                EAPOL_len=0            


            if j.haslayer(IP):

                IP_Z = 0
                IP_MF= 0
                IP_DF= 0

                IP_version=j[IP].version
                IP_ihl=j[IP].ihl
                IP_tos=j[IP].tos
                IP_len=j[IP].len
                IP_id=j[IP].id
                IP_flags=j[IP].flags

                IP_frag=j[IP].frag
                IP_ttl=j[IP].ttl
                IP_proto=j[IP].proto
                IP_chksum=j[IP].chksum


                #if j[IP].options!=0:
                IP_options=j[IP].options
                if "IPOption_Router_Alert"   in str(IP_options):
                    IP_options=1
                else:IP_options=0
                
                
                if j[Ether].src not in dst_ip_list:
                    dst_ip_list[j[Ether].src]=[]
                    dst_ip_list[j[Ether].src].append(j[IP].dst)
                elif j[IP].dst not in dst_ip_list[j[Ether].src]:
                    dst_ip_list[j[Ether].src].append(j[IP].dst)
                IP_add_count=len(dst_ip_list[j.src])

                #if IP_flags not in ipf: ipf.append(IP_flags)

                if IP_flags & Z:IP_Z = 1
                if IP_flags & MF:IP_MF = 1
                if IP_flags & DF:IP_DF = 1
                #if "Flag" in str(IP_flags):
                    #IP_flags=str(IP_flags)
                    #temp=IP_flags.find("(")
                    #IP_flags=int(IP_flags[6:temp-1])


                if j[IP].src not in IP_adresses:
                    IP_adresses.append(j[IP].src)
                if j[IP].dst  not in IP_adresses:
                    IP_adresses.append(j[IP].dst)           

                IP_src=j[IP].src#IP_adresses.index(j[IP].src)+1
                IP_dst=j[IP].dst#IP_adresses.index(j[IP].dst)+1                



            else:
                IP_Z = 0
                IP_MF= 0
                IP_DF= 0

                IP_version=0
                IP_ihl=0
                IP_tos=0
                IP_len=0
                IP_id=0
                IP_flags=0
                IP_frag=0
                IP_ttl=0
                IP_proto=0
                IP_chksum=0
                IP_src=0
                IP_dst=0
                IP_options=0
                IP_add_count=0            

            if j.haslayer(ICMP):
                ICMP_type=j[ICMP].type
                ICMP_code=j[ICMP].code
                ICMP_chksum=j[ICMP].chksum
                ICMP_id=j[ICMP].id
                ICMP_seq=j[ICMP].seq
                ICMP_ts_ori=j[ICMP].ts_ori
                ICMP_ts_rx=j[ICMP].ts_rx
                ICMP_ts_tx=j[ICMP].ts_tx
                ICMP_gw=j[ICMP].gw
                ICMP_ptr=j[ICMP].ptr
                ICMP_reserved=j[ICMP].reserved
                ICMP_length=j[ICMP].length
                ICMP_addr_mask=j[ICMP].addr_mask
                ICMP_nexthopmtu=j[ICMP].nexthopmtu
                ICMP_unused=j[ICMP].unused
            else:
                ICMP_type=0
                ICMP_code=0
                ICMP_chksum=0
                ICMP_id=0
                ICMP_seq=0
                ICMP_ts_ori=0
                ICMP_ts_rx=0
                ICMP_ts_tx=0
                ICMP_gw=0
                ICMP_ptr=0
                ICMP_reserved=0
                ICMP_length=0
                ICMP_addr_mask=0
                ICMP_nexthopmtu=0
                ICMP_unused=0




            if j.haslayer(TCP):
                TCP_FIN = 0
                TCP_SYN = 0
                TCP_RST = 0
                TCP_PSH = 0
                TCP_ACK = 0
                TCP_URG = 0
                TCP_ECE = 0
                TCP_CWR = 0
                TCP_sport=j[TCP].sport
                TCP_dport=j[TCP].dport
                TCP_seq=j[TCP].seq
                TCP_ack=j[TCP].ack
                TCP_dataofs=j[TCP].dataofs
                TCP_reserved=j[TCP].reserved
                TCP_flags=j[TCP].flags

                TCP_window=j[TCP].window
                TCP_chksum=j[TCP].chksum
                TCP_urgptr=j[TCP].urgptr
                TCP_options=j[TCP].options
                TCP_options= str(TCP_options).replace(",","-")
                if TCP_options!="0":
                    TCP_options=1
                else:
                    TCP_options=0
                
                
                

                #if TCP_flags not in tcpf:
                    #tcpf.append(TCP_flags)
                #print(TCP_options)
                if TCP_flags & FIN:TCP_FIN = 1
                if TCP_flags & SYN:TCP_SYN = 1
                if TCP_flags & RST:TCP_RST = 1
                if TCP_flags & PSH:TCP_PSH = 1
                if TCP_flags & ACK:TCP_ACK = 1
                if TCP_flags & URG:TCP_URG = 1
                if TCP_flags & ECE:TCP_ECE = 1
                if TCP_flags & CWR:TCP_CWR = 1   
                #print(TCP_flags)
                #if "Flag" in str(TCP_flags):
                    #TCP_flags=str(TCP_flags)
                    #temp=TCP_flags.find("(")
                    #TCP_flags=int(TCP_flags[6:temp-1])
                    



            else:
                TCP_sport=0
                TCP_dport=0
                TCP_seq=0
                TCP_ack=0
                TCP_dataofs=0
                TCP_reserved=0
                TCP_flags=0
                TCP_window=0
                TCP_chksum=0
                TCP_urgptr=0
                TCP_options=0
                TCP_options=0
                TCP_FIN = 0
                TCP_SYN = 0
                TCP_RST = 0
                TCP_PSH = 0
                TCP_ACK = 0
                TCP_URG = 0
                TCP_ECE = 0
                TCP_CWR = 0


            if j.haslayer(UDP):
                UDP_sport=j[UDP].sport
                UDP_dport=j[UDP].dport
                UDP_len=j[UDP].len
                UDP_chksum=j[UDP].chksum
            else:
                UDP_sport=0
                UDP_dport=0
                UDP_len=0
                UDP_chksum=0





            if j.haslayer(DHCP):
                DHCP_options=str(j[DHCP].options)
                DHCP_options=DHCP_options.replace(",","-")
                if "message" in DHCP_options:
                    x = DHCP_options.find(")")
                    DHCP_options=int(DHCP_options[x-1])
                    
            else:
                DHCP_options=0            


            if j.haslayer(BOOTP):
                BOOTP_op=j[BOOTP].op
                BOOTP_htype=j[BOOTP].htype
                BOOTP_hlen=j[BOOTP].hlen
                BOOTP_hops=j[BOOTP].hops
                BOOTP_xid=j[BOOTP].xid
                BOOTP_secs=j[BOOTP].secs
                BOOTP_flags=j[BOOTP].flags
                #if "Flag" in str(BOOTP_flags):BOOTP_flags=str(BOOTP_flags)temp=BOOTP_flags.find("(") BOOTP_flags=int(BOOTP_flags[6:temp-1])
                BOOTP_ciaddr=j[BOOTP].ciaddr
                BOOTP_yiaddr=j[BOOTP].yiaddr
                BOOTP_siaddr=j[BOOTP].siaddr
                BOOTP_giaddr=j[BOOTP].giaddr
                BOOTP_chaddr=j[BOOTP].chaddr
                BOOTP_sname=str(j[BOOTP].sname)
                if BOOTP_sname!="0":
                    BOOTP_sname=1
                else:
                    BOOTP_sname=0
                BOOTP_file=str(j[BOOTP].file)
                if BOOTP_file!="0":
                    BOOTP_file=1
                else:
                    BOOTP_file=0
                
                BOOTP_options=str(j[BOOTP].options)
                BOOTP_options=BOOTP_options.replace(",","-")
                if BOOTP_options!="0":
                    BOOTP_options=1
                else:
                    BOOTP_options=0
            else:
                BOOTP_op=0
                BOOTP_htype=0
                BOOTP_hlen=0
                BOOTP_hops=0
                BOOTP_xid=0
                BOOTP_secs=0
                BOOTP_flags=0
                BOOTP_ciaddr=0
                BOOTP_yiaddr=0
                BOOTP_siaddr=0
                BOOTP_giaddr=0
                BOOTP_chaddr=0
                BOOTP_sname=0
                BOOTP_file=0
                BOOTP_options=0






            if j.haslayer(DNS):
                DNS_length=j[DNS].length
                DNS_id=j[DNS].id
                DNS_qr=j[DNS].qr
                DNS_opcode=j[DNS].opcode
                DNS_aa=j[DNS].aa
                DNS_tc=j[DNS].tc
                DNS_rd=j[DNS].rd
                DNS_ra=j[DNS].ra
                DNS_z=j[DNS].z
                DNS_ad=j[DNS].ad
                DNS_cd=j[DNS].cd
                DNS_rcode=j[DNS].rcode
                DNS_qdcount=j[DNS].qdcount
                DNS_ancount=j[DNS].ancount
                DNS_nscount=j[DNS].nscount
                DNS_arcount=j[DNS].arcount
                DNS_qd=str(j[DNS].qd).replace(",","-")
                if DNS_qd!="0":
                    DNS_qd=1
                else:
                    DNS_qd=0
                DNS_an=str(j[DNS].an).replace(",","-")
                if DNS_an!="0":
                    DNS_an=1
                else:
                    DNS_an=0
                DNS_ns=str(j[DNS].ns).replace(",","-")
                if DNS_ns!="0":
                    DNS_ns=1
                else:
                    DNS_ns=0
                DNS_ar=str(j[DNS].ar).replace(",","-")
                if DNS_ar!="0":
                    DNS_ar=1
                else:
                    DNS_ar=0
            else:
                DNS_length=0
                DNS_id=0
                DNS_qr=0
                DNS_opcode=0
                DNS_aa=0
                DNS_tc=0
                DNS_rd=0
                DNS_ra=0
                DNS_z=0
                DNS_ad=0
                DNS_cd=0
                DNS_rcode=0
                DNS_qdcount=0
                DNS_ancount=0
                DNS_nscount=0
                DNS_arcount=0
                DNS_qd=0
                DNS_an=0
                DNS_ns=0
                DNS_ar=0





            pdata=[]
            if "TCP" in j:            
                pdata = (j[TCP].payload)
            if "Raw" in j:
                pdata = (j[Raw].load)
            elif "UDP" in j:            
                pdata = (j[UDP].payload)
            elif "ICMP" in j:            
                pdata = (j[ICMP].payload)
            pdata=list(memoryview(bytes(pdata)))            
    
            if pdata!=[]:
                entropy=shannon(pdata)        
            else:
                entropy=0
            payload_bytes=len(pdata)

            sport_class=port_class(TCP_sport+UDP_sport)
            dport_class=port_class(TCP_dport+UDP_dport)
            sport23=port_1023(TCP_sport+UDP_sport)
            dport23=port_1023(TCP_dport+UDP_dport)
            sport_bare=TCP_sport+UDP_sport
            dport_bare=TCP_dport+UDP_dport#port_class(TCP_dport+UDP_dport)
            
            try:
                label=MAC_list[j.src]
            except:
                label=""
            Mac=j.src             
            
            
            
            
            line=[pck_size,
            Ether_type,
            LLC_dsap,
            LLC_ssap,
            LLC_ctrl,
            EAPOL_version,
            EAPOL_type,
            EAPOL_len,
            IP_version,
            IP_ihl,
            IP_tos,
            IP_len,
            IP_flags,
            IP_Z,
            IP_MF,
            IP_id,
            IP_chksum,
            IP_DF  ,
            IP_frag,
            IP_ttl,
            IP_proto,
            IP_options,
            IP_add_count,
            ICMP_type,
            ICMP_code,
            ICMP_chksum,
            ICMP_id,
            ICMP_seq,
            ICMP_ts_ori,
            ICMP_ts_rx,
            ICMP_ts_tx,
            ICMP_ptr,
            ICMP_reserved,
            ICMP_length,
            #ICMP_addr_mask,
            ICMP_nexthopmtu,
            ICMP_unused,
            TCP_seq,
            TCP_ack,
            TCP_dataofs,
            TCP_reserved,
            TCP_flags,
            TCP_FIN,
            TCP_SYN,
            TCP_RST,
            TCP_PSH,
            TCP_ACK,
            TCP_URG,
            TCP_ECE,
            TCP_CWR   ,
            TCP_window,
            TCP_chksum,
            TCP_urgptr,
            TCP_options,
            UDP_len,
            UDP_chksum,
            DHCP_options,
            BOOTP_op,
            BOOTP_htype,
            BOOTP_hlen,
            BOOTP_hops,
            BOOTP_xid,
            BOOTP_secs,
            BOOTP_flags,
            BOOTP_sname,
            BOOTP_file,
            BOOTP_options,
            DNS_length,
            DNS_id,
            DNS_qr,
            DNS_opcode,
            DNS_aa,
            DNS_tc,
            DNS_rd,
            DNS_ra,
            DNS_z,
            DNS_ad,
            DNS_cd,
            DNS_rcode,
            DNS_qdcount,
            DNS_ancount,
            DNS_nscount,
            DNS_arcount,
            sport_class,
            dport_class,
            sport23,
            dport23,
            sport_bare,
            dport_bare,
            TCP_sport,
            TCP_dport,
            UDP_sport,
            UDP_dport, 
            payload_bytes,
            entropy,
            Mac,
            label]

            #print(line)
            line=str(line).replace("[","")
            line=str(line).replace("]","")
            #line=str(line).replace("\',","-")
            line=str(line).replace(", ",",")
            line=str(line).replace("\'","")
            line=str(line).replace("None","0")
            if label!="":
                ths.write(str(line)+"\n")  
            #kk=line.split(",")
            #print(len(kk))
            #if len(kk)==112:
            #ths.write(line+"\n")
            
            #else:print(line)
    print("  - ",numero+1,"/",len(files_add))    
    ths.close()          
    


# In[21]:


for ii,i in enumerate(files_add):
    filename=i[:-4]+".csv"
    ths = open("Protocol.csv", "w")
    ths.write("Protocol\n")
    
    command="tshark -r "+i+" -T fields -e _ws.col.Protocol -E header=n -E separator=, -E quote=d -E occurrence=f > temp.csv"
    os.system(command)

    with open("temp.csv", "r") as file:
        while True:
            line=file.readline()
            if line=="":break
            if  "ARP" not in line:# this line eliminates the headers of CSV files and incomplete streams .
                ths.write(str(line))
            else:
                continue                       
    ths.close()  
    print("   {}  /  {}".format(ii+1,len(files_add)))    
    os.remove("temp.csv")
    df1=pd.read_csv(filename)
    df2=pd.read_csv("Protocol.csv")
    df1["Protocol"]=df2["Protocol"]        
    label=df1["Label"]
    del df1["Label"]
    df1["Label"]=label
    df1.to_csv(filename,index=None)
os.remove("Protocol.csv")


# ________________

# In[22]:


name_list=find_the_way('./UNSW_PCAP/shared','.csv')
name_list


# In[23]:


from sklearn.model_selection import train_test_split
for name in name_list:    
    df=pd.read_csv(name)#,header=None) 
    X =df[df.columns[0:-1]]
    df[df.columns[-1]] = df[df.columns[-1]].astype('category')
    y=df[df.columns[-1]]

    # setting up testing and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27,stratify=y)

    # concatenate our training data back together
    train = pd.concat([X_train, y_train], axis=1)


    file=name[0:-5]+"_"+"_TRAIN.csv"
    train.to_csv(file,index=False)


    test= pd.concat([X_test, y_test], axis=1)

    file=name[0:-5]+"_"+"_TEST.csv"
    test.to_csv(file,index=False)


# In[24]:


shared_train=find_the_way('./UNSW_PCAP/shared','TRAIN.csv')
shared_test=find_the_way('./UNSW_PCAP/shared','TEST.csv')


# In[25]:


shared_train


# In[26]:


IP_flags = {'0': 1, '<Flag 0 ()>': 2, '<Flag 2 (DF)>': 3, '<Flag 1 (MF)>': 4}
TCP_flags = {'0': 1, '<Flag 2 (S)>': 2, '<Flag 18 (SA)>': 3, '<Flag 16 (A)>': 4, '<Flag 24 (PA)>': 5, '<Flag 25 (FPA)>': 6, '<Flag 17 (FA)>': 7, '<Flag 4 (R)>': 8, '<Flag 20 (RA)>': 9, '<Flag 194 (SEC)>': 10, '<Flag 1 (F)>': 11, '<Flag 152 (PAC)>': 12, '<Flag 144 (AC)>': 13,'<Flag 82 (SAE)>':14,'<Flag 49 (FAU)>':15}
BOOTP_flags = {'0': 1, '<Flag 0 ()>': 2, '<Flag 32768 (B)>': 3, 0: 1}
Protocol = {'EAPOL': 1, 'DHCP': 2, 'DNS': 3, 'TCP': 4, 'HTTP': 5, 'ICMP': 6, 'MDNS': 7, 'IGMPv3': 8, 'SSDP': 9, 'NTP': 10, 'HTTP/XML': 11, 'UDP': 12, 'SSLv2': 13, 'TLSv1': 14, 'ADwin Config': 15, 'TLSv1.2': 16, 'ICMPv6': 17, 'HTTP/JSON': 18, 'XID': 19, 'TFTP': 20, 'NXP 802.15.4 SNIFFER': 21, 'IGMPv2': 22, 'A21': 23, 'STUN': 24, 'Gearman': 25, '? KNXnet/IP': 26, 'UDPENCAP': 27, 'ESP': 28, 'SSL': 29, 'NBNS': 30, 'SIP': 31, 'BROWSER': 32, 'SABP': 33, 'ISAKMP': 34, 'CLASSIC-STUN': 35, 'Omni-Path': 36, 'XMPP/XML': 37, 'ULP': 38, 'TFP over TCP': 39, 'AX4000': 40, 'MIH': 41, 'DHCPv6': 42, 'TDLS': 43, 'RTMP': 44, 'TCPCL': 45, 'IPA': 46, 'GQUIC': 47, '0x86dd': 48, 'DB-LSP-DISC': 49, 'SSLv3': 50, 'LLMNR': 51, 'FB_ZERO': 52, 'OCSP': 53, 'IPv4': 54, 'STP': 55, 'SSH': 56, 'TLSv1.1': 57, 'KINK': 58, 'MANOLITO': 59, 'PKTC': 60, 'TELNET': 61, 'RTSP': 62, 'HCrt': 63, 'MPTCP': 64, 'S101': 65, 'IRC': 66, 'AJP13': 67, 'PMPROXY': 68, 'PNIO': 69, 'AMS': 70, 'ECATF': 71, 'LLC': 72, 'TZSP': 73,'RSIP':74,'SSHv2':75
,'DIAMETER':76
,'BFD Control':77
,'ASAP':78
,'DISTCC':79 
,'DISTCC ':79       
,'LISP':80
,'WOW':81
,'DTLSv1.0':82
,'SNMP':83
,'SMB2':84
,'SMB':85
,'NBSS':86
,'UDT':87,'HiQnet':88
,'POWERLINK/UDP':89
,'RTP':90
,'WebSocket':91
,'NAT-PMP':92
,'RTCP':93,'Syslog':94
,'Portmap':95
,'OpenVPN':96
,'BJNP':97
,'RIPv1':98
,'MAC-Telnet':99
,'ECHO':100
,'ASF':101
,'DAYTIME':102
,'SRVLOC':103
,'KRB4':104
,'CAPWAP-Control':105
,'XDMCP':106
,'Chargen':107
,'RADIUS':108
,'L2TP':109
,'DCERPC':110
,'KPASSWD':111
,'H264':112
,'FTP':113
,'FTP-DATA':114
,'ENIP':115
,'RIPv2':116
,'ICP':117,
"BACnet-APDU":118,
"IAX2":119,
"RX":120,
"HTTP2":121,
"SIP/SDP":122,
"TIME":123,
"Elasticsearch":124,
"RSL":125,
"TPCP":126,
 "IPv6":  127 }
Folder= {'Aria': 'Aria', 'D-LinkCam': 'D-LinkCam', 'D-LinkDayCam': 'D-LinkDayCam', 'D-LinkDoorSensor': 'D-LinkDoorSensor', 'D-LinkHomeHub': 'D-LinkHomeHub', 'D-LinkSensor': 'D-LinkSensor', 'D-LinkSiren': 'D-LinkSiren', 'D-LinkSwitch': 'D-LinkSwitch', 'D-LinkWaterSensor': 'D-LinkWaterSensor', 'EdimaxCam1': 'EdimaxCam', 'EdimaxCam2': 'EdimaxCam', 'EdimaxPlug1101W': 'EdimaxPlug1101W', 'EdimaxPlug2101W': 'EdimaxPlug2101W', 'EdnetCam1': 'EdnetCam', 'EdnetCam2': 'EdnetCam', 'EdnetGateway': 'EdnetGateway', 'HomeMaticPlug': 'HomeMaticPlug', 'HueBridge': 'HueBridge', 'HueSwitch': 'HueSwitch', 'iKettle2': 'iKettle2', 'Lightify': 'Lightify', 'MAXGateway': 'MAXGateway', 'SmarterCoffee': 'SmarterCoffee', 'TP-LinkPlugHS100': 'TP-LinkPlugHS100', 'TP-LinkPlugHS110': 'TP-LinkPlugHS110', 'WeMoInsightSwitch': 'WeMoInsightSwitch', 'WeMoInsightSwitch2': 'WeMoInsightSwitch', 'WeMoLink': 'WeMoLink', 'WeMoSwitch': 'WeMoSwitch', 'WeMoSwitch2': 'WeMoSwitch', 'Withings': 'Withings'}

label= {'Android Phone 1':'Non-IoT', 'Android Phone 2': 'Non-IoT',
'Samsung Galaxy Tab':'Non-IoT',
"Laptop":'Non-IoT',
"IPhone":'Non-IoT',
"MacBook":'Non-IoT',                   
"MacBook/Iphone":'Non-IoT',
"Samsung Galaxy Tab":'Non-IoT',
"MacBook-Iphone":'Non-IoT' }



# In[27]:



name_list


# In[28]:


name_list=find_the_way('./UNSW_PCAP/TRAIN','.csv')
name_list=name_list+shared_train
name={"UNSW_train_IoTDevID.csv":8000}


df=pd.read_csv(name_list[0])
col_names=list(df.columns)

for i in name:
    
    empty = pd.DataFrame(columns=col_names)
    empty.to_csv(i, mode="a", index=False)#,header=False)

    for iii in name_list:

        df=pd.read_csv(iii)
        print(iii,df.shape)
        if len(df)>name[i]:
            df=df.sample(n=name[i], random_state=0)
        df.to_csv(i, mode="a", index=False,header=False)

    df=pd.read_csv(i)
    df=df.replace({"IP_flags": IP_flags})
    df=df.replace({"TCP_flags": TCP_flags})
    df=df.replace({"BOOTP_flags": BOOTP_flags})
    df=df.replace({"Protocol": Protocol})
    df=df.replace({"Label": label})
    df.to_csv(i,index=None)




# In[29]:


name_list=find_the_way('./UNSW_PCAP/TEST','.csv')
name_list=name_list+shared_test
name={"UNSW_test_IoTDevID.csv":2000}


df=pd.read_csv(name_list[0])
col_names=list(df.columns)

for i in name:
    
    empty = pd.DataFrame(columns=col_names)
    empty.to_csv(i, mode="a", index=False)#,header=False)

    for iii in name_list:

        df=pd.read_csv(iii)
        print(iii,df.shape)
        if len(df)>name[i]:
            df=df.sample(n=name[i], random_state=0)
        df.to_csv(i, mode="a", index=False,header=False)

    df=pd.read_csv(i)
    df=df.replace({"IP_flags": IP_flags})
    df=df.replace({"TCP_flags": TCP_flags})
    df=df.replace({"BOOTP_flags": BOOTP_flags})
    df=df.replace({"Protocol": Protocol})
    df=df.replace({"Label": label})
    df.to_csv(i,index=None)




# ## Delete unnecessary CSV files

# In[30]:


name_list=find_the_way('./UNSW_PCAP/','.csv')
for i in name_list:
    os.remove(i)

