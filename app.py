# ====================== IMPORT PACKAGES ==============

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing 

import streamlit as st
import base64

 # ------------ TITLE 

st.markdown(f'<h1 style="color:#8d1b92;text-align: center;font-size:36px;">{"Anomaly Detection Using Machine Learning"}</h1>', unsafe_allow_html=True)


# ================ Background image ===

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('1.jpg')



import pickle

with open('model.pickle', 'rb') as f:
    rf = pickle.load(f)
    
    
    
    
with open('finalpred.pickle', 'rb') as f:
     pred_rf = pickle.load(f)  
     

# pred_rf = pred_rf

# ================== PREDICTION  ====================

st.write("-----------------------------------")
st.write("          Prediction               ")
st.write("-----------------------------------")


inpp = st.text_input("Enter Prediction Index Number = ")

butt = st.button("Submit")

if butt:

        
    if pred_rf[int(inpp)] == 0:
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{"Identified = BRUTE FORCE"}</h1>', unsafe_allow_html=True)


        from faker import Faker
        ex = Faker()
        ip_rec = ex.ipv4()
        ip_sen = ex.ipv4()
        
        
        st.write("Sender's IP Address   = ",ip_sen )
        st.write("Reciever's IP Address = ",ip_rec )
            
    
    elif pred_rf[int(inpp)] == 1:
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{"Identified = HTTP DDoS"}</h1>', unsafe_allow_html=True)


        from faker import Faker
        ex = Faker()
        ip_rec = ex.ipv4()
        ip_sen = ex.ipv4()
        
        
        st.write("Sender's IP Address   = ",ip_sen )
        st.write("Reciever's IP Address = ",ip_rec )


    elif pred_rf[int(inpp)] == 2:
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{"Identified = INSIDER THREAT"}</h1>', unsafe_allow_html=True)


        from faker import Faker
        ex = Faker()
        ip_rec = ex.ipv4()
        ip_sen = ex.ipv4()
        
        
        st.write("Sender's IP Address   = ",ip_sen )
        st.write("Reciever's IP Address = ",ip_rec )        

    elif pred_rf[int(inpp)] == 3:
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{"Identified = MAN IN THE MIDDLE ATTACK"}</h1>', unsafe_allow_html=True)


        from faker import Faker
        ex = Faker()
        ip_rec = ex.ipv4()
        ip_sen = ex.ipv4()
        
        
        st.write("Sender's IP Address   = ",ip_sen )
        st.write("Reciever's IP Address = ",ip_rec )   
        
    elif pred_rf[int(inpp)] == 4:
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{"Identified = NORMAL "}</h1>', unsafe_allow_html=True)


    elif pred_rf[int(inpp)] == 5:
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{"Identified = PORT SCANNING "}</h1>', unsafe_allow_html=True)


        from faker import Faker
        ex = Faker()
        ip_rec = ex.ipv4()
        ip_sen = ex.ipv4()
        
        
        st.write("Sender's IP Address   = ",ip_sen )
        st.write("Reciever's IP Address = ",ip_rec )  