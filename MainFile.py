# ====================== IMPORT PACKAGES ==============

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing 


# ===-------------------------= INPUT DATA -------------------------------


    
dataframe=pd.read_csv("Dataset.csv")

print("--------------------------------")
print("Data Selection")
print("--------------------------------")
print()
print(dataframe.head(15))    
    
    
    
#-------------------------- PRE PROCESSING --------------------------------
   
#------ checking missing values --------
   
print("----------------------------------------------------")
print("              Handling Missing values               ")
print("----------------------------------------------------")
print()
print(dataframe.isnull().sum())




res = dataframe.isnull().sum().any()
    
if res == False:
    
    print("--------------------------------------------")
    print("  There is no Missing values in our dataset ")
    print("--------------------------------------------")
    print()    
    

    
else:

    print("--------------------------------------------")
    print(" Missing values is present in our dataset   ")
    print("--------------------------------------------")
    print()    

    
    dataframe = dataframe.fillna(0)
    
    resultt = dataframe.isnull().sum().any()
    
    if resultt == False:
        
        print("--------------------------------------------")
        print(" Data Cleaned !!!   ")
        print("--------------------------------------------")
        print()    
        print(dataframe.isnull().sum())



               
# ---- LABEL ENCODING
        
print("--------------------------------")
print("Before Label Encoding")
print("--------------------------------")   

df_class=dataframe['Label']

print(dataframe['Label'].head(15))

   
              
print("--------------------------------")
print("After Label Encoding")
print("--------------------------------")            
        
label_encoder = preprocessing.LabelEncoder() 

dataframe['Label']=label_encoder.fit_transform(dataframe['Label'])     


           
print(dataframe['Label'].head(15))       




#-------------------------- FEATURE SCALING --------------------------------
   
#------ MIN MAX SCALAR --------

dataframe1 = dataframe

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(dataframe1)


# ------------------------- DATA SPLITTING  -------------------------------

X1 = dataframe.drop(['Label'],axis=1)

y1 = dataframe['Label']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=0)


print("---------------------------------------------")
print("             Data Splitting                  ")
print("---------------------------------------------")

print()

print("Total no of input data   :",dataframe.shape[0])
print("Total no of test data    :",X_test1.shape[0])
print("Total no of train data   :",X_train1.shape[0])



# ------------------------- FEATURE EXTRACTION  -------------------------------


#------ PCA --------

from sklearn.decomposition import PCA

pca = PCA(n_components=15) 
principal_components = pca.fit_transform(dataframe)


print("---------------------------------------------")
print("   Feature Extraction ---> PCA               ")
print("---------------------------------------------")

print()

print(" Original Features     :",dataframe.shape[1])
print(" Reduced Features      :",principal_components.shape[1])



# Plot the results
plt.figure(figsize=(5, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c='blue', edgecolor='k', s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: First Two Principal Components')
plt.grid()
plt.savefig("pca.png")
plt.show()


# ------------------------- CLASSIFICATION  -------------------------------


#==========================================
# o LOGISTIC REGRESSION
#==========================================


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train1, y_train1)

pred_rf = rf.predict(X_test1)

acc_lr = metrics.accuracy_score(pred_rf,y_test1) * 100

loss_lr= 100 - acc_lr

print("---------------------------------------------")
print("  Performance Analysis - Random Forest")
print("---------------------------------------------")

print()

print("1) Accuracy = " ,acc_lr )
print()
print("2) Loss     = ",loss_lr )
print()
print("3) Classification Report")
print()
print(metrics.classification_report(y_test1, pred_rf))




#==========================================
# o CONVOLUTIONAL NEURAL NETWORK
#==========================================


print("---------------------------------------------")
print("   Convlotional Neural Network - CNN         ")
print("---------------------------------------------")


from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Reshape input for CNN
X_train_cnn = X_train1.values.reshape(X_train1.shape[0], X_train1.shape[1], 1)
X_test_cnn = X_test1.values.reshape(X_test1.shape[0], X_test1.shape[1], 1)

# CNN model
model_cnn = Sequential()
model_cnn.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train_cnn.shape[1], 1)))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(16, activation='relu'))
model_cnn.add(Dense(1, activation='sigmoid'))

model_cnn.compile(optimizer='adam', loss='mse')

# Train the model
model_cnn.fit(X_train_cnn, y_train1, epochs=10, batch_size=32, validation_data=(X_test_cnn, y_test1))


# Evaluate the CNN
loss_cnn = model_cnn.evaluate(X_test_cnn, y_test1)
# print(f"CNN Accuracy: {accuracy_cnn}")
print("---------------------------------------------")
print("  Performance Analysis - CNN ")
print("---------------------------------------------")

print()

accuracy_cnn =  100 - loss_cnn

from sklearn.metrics import classification_report

# Make predictions
y_pred_cnn_probs = model_cnn.predict(X_test_cnn)

y_pred_cnn = (y_pred_cnn_probs > 0.5).astype(int)  

print("1) Accuracy = " ,accuracy_cnn )
print()
print("2) Loss     = ",loss_cnn )
print()



#==========================================
# o HYBRID LR + CNN-1D
#==========================================


print("---------------------------------------------")
print("   Hybrid CNN + LR       ")
print("---------------------------------------------")

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train1, y_train1)

pred_lr = lr.predict(X_train1)


# CNN model
model_cnn = Sequential()
model_cnn.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train_cnn.shape[1], 1)))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(16, activation='relu'))
model_cnn.add(Dense(1, activation='sigmoid'))

model_cnn.compile(optimizer='adam', loss='mae')

# Train the model
model_cnn.fit(X_train_cnn, pred_lr, epochs=10, batch_size=32, validation_data=(X_test_cnn, y_test1))


# Evaluate the CNN
loss_hyb = model_cnn.evaluate(X_test_cnn, y_test1)
# print(f"CNN Accuracy: {accuracy_cnn}")
print("---------------------------------------------")
print("  Performance Analysis - Hybrid CNN + LR ")
print("---------------------------------------------")

print()

accuracy_hyb =  100 - loss_hyb

from sklearn.metrics import classification_report

# Make predictions
y_pred_cnn_probs = model_cnn.predict(X_test_cnn)

y_pred_cnn = (y_pred_cnn_probs > 0.5).astype(int)  

print("1) Accuracy = " ,accuracy_hyb )
print()
print("2) Loss     = ",loss_hyb )
print()



import pickle
with open('model.pickle', 'wb') as f:
    pickle.dump(rf, f)


import pickle
with open('finalpred.pickle', 'wb') as f:
    pickle.dump(pred_rf, f)








# --------------- COMPARISON GRAPH


import seaborn as sns
sns.barplot(x=["RF","CNN","Hybrid"],y=[acc_lr,accuracy_cnn,accuracy_hyb])
plt.title("Comparison Graph")
plt.savefig("com.png")
plt.show()





import seaborn as sns
plt.figure(figsize = (6,6))
counts = y1.value_counts()
plt.pie(counts, labels = counts.index, startangle = 90, counterclock = False, wedgeprops = {'width' : 0.6},autopct='%1.1f%%', pctdistance = 0.55, textprops = {'color': 'black', 'fontsize' : 15}, shadow = True,colors = sns.color_palette("Paired")[3:])
plt.text(x = -0.35, y = 0, s = 'Total Data: {}'.format(dataframe.shape[0]))
plt.title('Attack Analysis', fontsize = 14);
plt.show()

plt.savefig("graph.png")
plt.show()

