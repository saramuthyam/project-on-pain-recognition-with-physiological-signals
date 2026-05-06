from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers import Dense, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Activation, RepeatVector
from keras.layers import LSTM #class for LSTM training
import os
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional, GRU #class for bidirectional LSTM as BILSTM and GRU
from keras.utils.np_utils import to_categorical

from sklearn.metrics import accuracy_score
import pickle
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut #class to calculate LOSO
from sklearn.model_selection import cross_val_score




main = tkinter.Tk()
main.title("Pain Recognition") #designing main screen
main.geometry("1300x1200")

global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca
global accuracy, precision, recall, fscore, values,algorithm, predict,unique,extension_model
precision = []
recall = []
fscore = []
accuracy = []
loso = []

def uploadDataset():
    global filename, dataset, labels, values,unique
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,'Dataset loaded\n\n')
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    text.insert(END,str(dataset))
    data = dataset.values
    plt.plot(data) 
    plt.xlabel("Number of Records")
    plt.ylabel("Signals")
    plt.title("EEG Signal from all Subjects")
    plt.show()

def processDataset():
    global dataset, X, Y
    global X_train, X_test, y_train, y_test, pca, scaler,labels
    text.delete('1.0', END)
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1] #extract training features as X
    Y = dataset[:,dataset.shape[1]-1] #extract target pain label
    scaler = MinMaxScaler((0,1))
    X = scaler.fit_transform(X)#normalized features
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices) #shuffle the dataset values
    X = X[indices]
    Y = Y[indices]
    text.insert(END,"Normalized Training Features : \n\n "+str(X)+"\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    print()
    text.insert(END,"Dataset train & test split as 80% dataset for training and 20% for testing"+"\n")
    text.insert(END,"Training Size (80%): "+str(X_train.shape[0])+"\n") #print training and test size
    text.insert(END,"Testing Size (20%): "+str(X_test.shape[0])+"\n")
    print()
    labels, count = np.unique(dataset[:, -1], return_counts = True)
    labels = ["Pain0", "Pain1", "Pain2" ,"Pain3", "Pain4"]
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Dataset Class Label Graph")
    plt.ylabel("Count")
    plt.show()

def calculateMetrics(algorithm, predict, testY,labels):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100     
    print()
    text.insert(END,algorithm+' Accuracy  : '+str(a)+"\n")
    text.insert(END,algorithm+' Precision   : '+str(p)+"\n")
    text.insert(END,algorithm+' Recall      : '+str(r)+"\n")
    text.insert(END,algorithm+' FMeasure    : '+str(f)+"\n")  
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(5, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

def trainRF():
    global X_train, y_train, X_test, y_test
    global algorithm, predict, test_labels,labels
    text.delete('1.0', END)
    rf = RandomForestClassifier(n_estimators=40, criterion='gini', max_features="log2", min_weight_fraction_leaf=0.3)
    rf.fit(X_train, y_train)
    predict = rf.predict(X_test)#perform prediction on test data
    calculateMetrics("Existing Random Forest", predict, y_test,labels)#call function to calculate accuracy and other metrics

def trainCNNBILSTM():
    global X_train, y_train, X_test, y_test
    global algorithm, predict,labels
    text.delete('1.0', END)
    X_train = np.reshape(X_train, (X_train.shape[0], 34, 4))
    X_test = np.reshape(X_test, (X_test.shape[0], 34, 4))
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    #create CNN sequential object
    propose_model = Sequential()
    #create CNN1D layer with 32 neurons for data filteration and pool size as 3
    propose_model.add(Conv1D(filters=32, kernel_size = 3, activation = 'relu', input_shape = (X_train.shape[1], X_train.shape[2])))
    #defining another CNN layer with 64 neurons
    propose_model.add(Conv1D(filters=64, kernel_size = 2, activation = 'relu'))
    propose_model.add(Conv1D(filters=128, kernel_size = 2, activation = 'relu'))
    #max pooling layer to collect relevant features from CNN layer
    propose_model.add(MaxPooling1D(pool_size = 1))
    propose_model.add(Flatten())
    propose_model.add(RepeatVector(2))
    #defining BILSTM kayer with 32 neurons to optimize CNN features
    propose_model.add(Bidirectional(LSTM(32, activation = 'relu', return_sequences=True)))
    propose_model.add(Bidirectional(LSTM(64, activation = 'relu')))
    #adding dropout layer to remove irrelevant features
    propose_model.add(Dropout(0.2))
    #defining output an dprediction layer
    propose_model.add(Dense(units = 100, activation = 'softmax'))
    propose_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    #train and compile the model
    propose_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/propose_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/propose_weights.hdf5', verbose = 1, save_best_only = True)
        hist = propose_model.fit(X_train, y_train, batch_size = 32, epochs = 10, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/propose_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        propose_model = load_model("model/propose_weights.hdf5")
    #perform prediction on test data   
    predict = propose_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("Propose CNN + BILSTM", predict, y_test1, labels)#call function to calculate accuracy and other metrics

def trainCNNBILSTMBIGRU():
    global X_train, y_train, X_test, y_test
    global algorithm, predict,labels,extension_model
    text.delete('1.0', END)
    extension_model = Sequential()
    #create CNN1D layer with 32 neurons for data filteration and pool size as 3
    extension_model.add(Conv1D(filters=32, kernel_size = 3, activation = 'relu', input_shape = (X_train.shape[1], X_train.shape[2])))
    extension_model.add(Conv1D(filters=64, kernel_size = 2, activation = 'relu'))
    extension_model.add(Conv1D(filters=128, kernel_size = 2, activation = 'relu'))
    extension_model.add(MaxPooling1D(pool_size = 1))
    extension_model.add(Flatten())
    extension_model.add(RepeatVector(2))
    #adding LSTM Bidirectional layer to obtained optimized features from CNN
    extension_model.add(Bidirectional(LSTM(32, activation = 'relu', return_sequences=True)))
    #now bidirectional GRU will extract optimized fetaures from BI-LSTM and then train a model with below prediction layer
    extension_model.add(Bidirectional(GRU(64, activation = 'relu')))
    extension_model.add(Dropout(0.2))
    #Define output prediction layer
    extension_model.add(Dense(units = 100, activation = 'softmax'))
    extension_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    #compile and train the model
    extension_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/extension_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/extension_weights.hdf5', verbose = 1, save_best_only = True)
        hist = extension_model.fit(X_train, y_train, batch_size = 32, epochs = 10, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/extension_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        extension_model = load_model("model/extension_weights.hdf5")
    #perform prediction on test data   
    predict = extension_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("Etension CNN + BILSTM + BIGRU", predict, y_test1, labels)#call function to calculate accuracy and other metrics

def graph():
    global X_train, y_train, X_test, y_test
    global algorithm, predict,labels
    text.delete('1.0', END)
    df = pd.DataFrame([['Existing Random Forest','Precision',precision[0]],['Existing Random Forest','Recall',recall[0]],['Existing Random Forest','F1 Score',fscore[0]],['Existing Random Forest','Accuracy',accuracy[0]],
                       ['Propsoe CNN + BI-LSTM','Precision',precision[1]],['Propsoe CNN + BI-LSTM','Recall',recall[1]],['Propsoe CNN + BI-LSTM','F1 Score',fscore[1]],['Propsoe CNN + BI-LSTM','Accuracy',accuracy[1]],
                       ['Extension CNN + BI-LSTM + BI-GRU','Precision',precision[2]],['Extension CNN + BI-LSTM + BI-GRU','Recall',recall[2]],['Extension CNN + BI-LSTM + BI-GRU','F1 Score',fscore[2]],['Extension CNN + BI-LSTM + BI-GRU','Accuracy',accuracy[2]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.title("All Algorithms Performance Graph")
    plt.show()


def predict():
    global X_train, y_train, X_test, y_test
    global algorithm, predict,labels,extension_model
    text.delete('1.0', END)
    testData = pd.read_csv("Dataset/testData.csv")#reading test data
    testData.fillna(0, inplace = True)
    temp = testData.values
    testData = testData.values
    test = scaler.transform(testData)#normalizing values
    test = np.reshape(test, (test.shape[0], 34, 4))
    predict = extension_model.predict(test)#performing prediction on test data using extension model object
    for i in range(len(predict)):
        y_pred = np.argmax(predict[i])
        text.insert(END,"Test Data = "+str(temp[i])+" Predicted Pain Type ====> "+labels[y_pred]+"\n")

font = ('times', 16, 'bold')
title = Label(main, text='Pain Recognition With Physiological Signals Using Multi-Level Context Information')
title.config(bg='gray24', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=27,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Preprocess & Split Dataset", command=processDataset)
processButton.place(x=250,y=100)
processButton.config(font=font1)

rfButton = Button(main, text="Run Random Forest", command=trainRF)
rfButton.place(x=490,y=100)
rfButton.config(font=font1)

cnnButton = Button(main, text="Run CNN + BILSTM", command=trainCNNBILSTM)
cnnButton.place(x=730,y=100)
cnnButton.config(font=font1)

dtButton = Button(main, text="Run CNN+BILSTM+BIGRU", command=trainCNNBILSTMBIGRU)
dtButton.place(x=970,y=100)
dtButton.config(font=font1)

graphButton = Button(main, text="Comparision Graph", command=graph)
graphButton.place(x=10,y=150)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict", command=predict)
predictButton.place(x=250,y=150)
predictButton.config(font=font1)



main.config(bg='chartreuse2')
main.mainloop()



