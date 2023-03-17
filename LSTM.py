from Functions import inputstargets 
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential, optimizers, callbacks
from keras.layers import Dense, LSTM, InputLayer, Bidirectional,Dropout
from keras.utils import to_categorical
import time


#define subject datasets to evaluate 
subjects = list(range(1,28))

#define lists to store data accumulated across multiple subjects
all_acc = []
all_train_time = []
all_test_times = []


#Function that defines stepwise learning rate for DL models 
def scheduler(epoch):
  if epoch < 50:
    lr =0.001
  elif 50 <= epoch <100:
    lr =0.0005
  elif 100 <= epoch <150:
    lr =0.0001
  elif epoch >= 150:
    lr =0.00001

  return lr 

#iterate through subjects 
for subject in subjects:

    #retreive input sEMG feature data and target labels from CSV files
    inputs, targets = inputstargets(subject,"train") 
    val_inputs, val_targets = inputstargets(subject,"validation") 
    test_inputs, test_targets = inputstargets(subject,"test")

    #reshape data inputs to preferred LSTM configuration 
    inputs = np.array(inputs).transpose(0, 2, 1)
    val_inputs = np.array(val_inputs).transpose(0, 2, 1)
    test_inputs = np.array(test_inputs).transpose(0, 2, 1)

    #one hot encode target data 
    targets = to_categorical(targets, num_classes = 15)
    val_targets = to_categorical(val_targets, num_classes= 15)
    test_targets = to_categorical(test_targets, num_classes = 15)

    #define optimised LSTM architecture 
    model = Sequential()

    model.add(InputLayer(input_shape = (40,10)))
    model.add(Dense(400))
    model.add(Bidirectional(LSTM(1000, return_sequences = True)))
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(1000)))
    model.add(Dropout(0.4))
    model.add(Dense(15, activation='softmax'))

    
    #compile model and define loss function and otpimiser 
    model.compile(optimizer= optimizers.Adam(learning_rate= 0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    callback = callbacks.LearningRateScheduler(scheduler) #implement stepwise learning rate

    #start timer 
    start_time = time.time()

    #train and validate the model and define epochs and batch size  
    history = model.fit(inputs,targets, epochs=125,validation_data=(val_inputs,val_targets), verbose=1, callbacks=[callback], batch_size = 32)

    #calculate training time 
    train_time = (time.time() - start_time)

    #test the model
    loss, acc = model.evaluate(test_inputs,test_targets, verbose=1)

    #store accuracy for specific subject 
    all_acc.append(acc)

    #calculate test time 
    test_time = (time.time() - train_time -start_time)

    #store time values in appropriate lists 
    all_train_time.append(train_time)
    all_test_times.append(test_time) 


    #plot graph of loss against epoch for validation set 
    plt.figure(1)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel("Model loss")
    plt.legend()
    plt.show()

    #plot graph of accuracy against epoch for validation set
    plt.figure(2)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel("Classification accuracy/%")
    plt.legend()
    plt.show()

    
    #test LSTM again using model.predict function so that performance metrics can be generated 

    #create lists to store actual and predicted classes
    # to allow performance metrics to be generated 
    Actual_Class = []
    Predicted_Class = []

    for input, target in zip(test_inputs, test_targets):
        # predict gesture classes and append to list
        prediction = model.predict(np.asarray([input])) 
        Predicted_Class.append(np.argmax(prediction)) 
        # append actual gesture class of to list
        Actual_Class.append(np.argmax(target))
        
    #create and display confusion matrix
    cm = confusion_matrix(Actual_Class, Predicted_Class)
    display = "0", "1", "2","3","4","5","6","7","8","9","10","11","12","13","14"
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display)
    disp.plot(cmap="Blues")

    #create classification report 
    cr = classification_report(Actual_Class,Predicted_Class)
    print(cr)
    plt.show()


#calculate average classification accuracy     
all_acc = np.array(all_acc)
totalaccuracy = np.mean(all_acc)
print("avg:",totalaccuracy,"values:",all_acc)

#calculate average and cumulative train and test times 
all_train_time = np.array(all_train_time)
all_test_times = np.array(all_test_times)

avg_train = np.mean(all_train_time)
avg_test = np.mean(all_test_times)

cum_train = np.sum(all_train_time)
cum_test = np.sum(all_test_times)


print("train times", all_train_time)
print("test times", all_test_times)

print(" avg.train times", avg_train)
print("avg. test times", avg_test)

print(" cum.train times", cum_train)
print("cum. test times", cum_test)