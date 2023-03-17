from sklearn import svm
from Functions import featinputstargets
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import time 


#instantiate SVM classifier 
SVM = svm.SVC(kernel='rbf', C=10, gamma=0.1)

#define subject datasets to evaluate 
subjects = list(range(1,28))

#define lists to store data accumulated across multiple subjects
acc = []
all_train_time = []
all_test_times = []

#iterate through subjects 
for subject in subjects:
    
    #retreive input sEMG feature data and target labels from CSV files
    inputs, targets = featinputstargets(subject,"train") 
    val_inputs, val_targets = featinputstargets(subject,"validation") 
    test_inputs, test_targets = featinputstargets(subject,"test")

    #start timer 
    start_time = time.time()

    #fit SVM model to training data 
    SVM.fit(inputs, targets)

    #calculate training time 
    train_time = (time.time() - start_time)

    #use SVM to classify test data
    tpred = SVM.predict(test_inputs)  #this was changed to val_inputs during validation

    #calculate test time 
    test_time = (time.time() - train_time -start_time)

    #store time values in appropriate lists 
    all_train_time.append(train_time)
    all_test_times.append(test_time)

    #evaulate SVM performance on test data set
    tscore = []
    for i, sample in enumerate(test_inputs):
        #check if the SVM classification was correct
        if round(tpred[i]) == test_targets[i]:  #this was changed to val_targets during validation
            tscore.append(1)
        else:
            tscore.append(0)
    pass

    # calculate the accuracy of SVM on test set
    score = np.asarray(tscore)
    test_acc = score.sum() / score.size * 100
    print("test accuracy:", test_acc)
    
    #create and display confusion matrix
    cm = confusion_matrix(test_targets, tpred)
    display = "0", "1", "2","3","4","5","6","7","8","9","10","11","12","13","14"
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display)
    disp.plot(cmap="Blues")

    #create classification report 
    cr = classification_report(test_targets,tpred)
    print(cr)
    plt.show()
    
    #add specific subject accuracy to list
    acc.append(test_acc)

#calculate average classification accuracy     
acc = np.array(acc)
totalaccuracy = np.mean(acc)
print("avg:",totalaccuracy,acc)

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
