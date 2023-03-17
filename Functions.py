import math
from scipy.stats import skew, kurtosis
import numpy as np
import pywt 
import csv
from pyentrp import entropy as ent
import statsmodels.api as sm
import eeglib 

###### Functions ##########

# This script contains all the required functions to run the classifier model scripts and the Sort script


#Function to segment sEMG data into defined time windows 
#takes data to be segmented as input
def windowmaker(data):

        #define window parameters and sampling frequency
        window_length = 0.5 
        overlap = 0.7
        fs = 100 
        
        num_samp = int(fs*window_length) #calculate number of samples in each window 
        next_window = int(num_samp - num_samp*overlap) #calculate sample number at which next window starts
        windows = []
        window_start = 0
        while window_start + num_samp < len(data): #ensure window length is within data 
                window_end = window_start + num_samp  #set end of window
                subwindow = data[window_start:window_end] #generate data window
                windows.append(subwindow) #add subwindow to group of windows 
                window_start = window_start + next_window #set starting point of next window
        windows = np.array(windows).transpose(0, 2, 1)
        return windows

#Function to compute and extract features from input sEMG data 
#takes data to extract features from as input as well as the desired feature set 
def feature_extract(data,set):
        features = []
        length = len(data)

        #create TD feature set  
        if set == 1:
                mav = np.sum(np.absolute(data))/length #calculate mean absolute value
                cross = zc(data,mav)
                slope = slopechange(data)
                wavlen = np.sum(abs(np.diff(data))) #calculate waveform length 
                featvec = np.hstack((mav,cross,slope,wavlen)) #combine features into vector
                return featvec #len 4
        
        #create Enhanced TD feature set 
        if set == 2:
                mav = np.sum(np.absolute(data))/length  #calculate mean absolute value
                cross = zc(data,mav)
                slope = slopechange(data)
                wavlen = np.sum(abs(np.diff(data))) #calculate waveform length 
                rms = np.sqrt(np.mean(data **2)) #calculate rms of signal
                iemg = np.sum(abs(data))  
                skewness = skew(data)
                ar_coff = ARcoff(data)
                hjorth_param = Hjorth(data)
                featvec = np.hstack((mav,cross,slope,wavlen,rms,iemg,skewness,ar_coff,hjorth_param)) #combine features into vector
                return featvec 

        #create Ninapro feature set 
        if set == 3:
                mav = np.sum(np.absolute(data))/length  #calculate mean absolute value
                cross = zc(data,mav)
                slope = slopechange(data)
                wavlen = np.sum(abs(np.diff(data)))
                rms = np.sqrt(np.mean(data **2)) #calculate rms of signal
                hgram =  np.histogram(data, bins = 20)
                dwav = dwt(data)
                featvec = np.hstack((mav,cross,slope,wavlen,rms,hgram[0], dwav)) #combine features into vector
                return featvec  

        #create SampEn Pipeline feature set 
        if set == 4:
                sampentr = sampEn(data)
                wavlen = np.sum(abs(np.diff(data))) #calculate waveform length 
                rms = np.sqrt(np.mean(data **2)) #calculate rms of signal
                featvec = np.hstack((sampentr,wavlen,rms)) #combine features into vector
                return featvec 
        
        # extract SampEn feature 
        if set == 5:
                sampentr = sampEn(data)
                featvec = sampentr 
                return featvec 

        # extract SampEn and AR features 
        if set == 6:
                sampentr = sampEn(data)
                ar_coff = ARcoff(data)
                featvec = np.hstack((sampentr,ar_coff)) #combine features into vector
                return featvec 

        # extract AR feature 
        if set == 7:
                ar_coff = ARcoff(data)
                featvec = np.hstack((ar_coff)) 
                return featvec 



#Function to carry out marginal dsicrete wavelet transfrom 
#on input data and reduce number of coefficients by computing
#more features from returned dwt coefficients
def dwt(data):
        dwt = pywt.wavedec(data,'db7',level=3) #implement dwt

        #claculate mav of returned coefficients 
        mav0 = mav(dwt[0])
        mav1 = mav(dwt[1])
        mav2 = mav(dwt[2])
        mav3 = mav(dwt[3])

        #claculate signal power of returned coefficients 
        pwr1 = np.mean(dwt[1] **2)
        pwr2 = np.mean(dwt[2] **2)
        pwr3 = np.mean(dwt[3] **2)
        std1 = np.std(dwt[1])
        std2 = np.std(dwt[2])
        std3 = np.std(dwt[3])

        #claculate skew of returned coefficients 
        skw1 = skew(dwt[1])
        skw2 = skew(dwt[2])
        skw3 = skew(dwt[3])

        #claculate kurtosis of returned coefficients 
        kurt1 = kurtosis(dwt[1])
        kurt2 = kurtosis(dwt[2])
        kurt3 = kurtosis(dwt[3])
        vec = np.hstack((mav0,mav1,mav2,mav3,pwr1,pwr2,pwr3,std1,std2,std3,skw1,skw2,skw3,kurt1,kurt2,kurt3)) #combine features into vector
      
        return vec


#Function to calulate Hjorth parameters from input data
def Hjorth(data):
        result = all(element == data[0] for element in data)
        if (result):
                data[0] = data[0]+0.0000001  #add tiny value to data to prevent "nan" errors
        a = eeglib.features.hjorthActivity(data)
        c = eeglib.features.hjorthComplexity(data)
        m = eeglib.features.hjorthMobility(data)
        hjorth = [a,c,m]

        return hjorth

#Function to calulate autoregressive coefficients from input data
def ARcoff(data):
        result = all(element == data[0] for element in data)
        if (result):
                data[0] = data[0]+0.0000001    #add tiny value to data to prevent "nan" errors
        coff, sig = sm.regression.linear_model.burg(data, order=4 )
        for i in range(len(coff)):
                isnan = np.isnan(coff[i])  #check no "nan" values  produced and replace them with 0 if there are
                if isnan == True:
                        coff[i]= 0

        return coff

#Function to calulate zero crossings from input data, 
#takes data and mean aboslute value of data as input
def zc(data,mav):
        cross = 0
        for x,y in zip(data[::],data[1::]):
                if x > mav and y < mav:
                        cross +=1
                elif y > mav and x < mav:
                        cross +=1
        return int(cross)

#Function to calulate slope sign changes from input data
def slopechange(data):
        slope = 0
        for x,y,z in zip(data[::],data[1::],data[2::]):
                if y > x and y > z:
                        slope +=1
                elif y < x and y < z:
                        slope +=1
        return int(slope)

#Function to calulate mean absolute value from input data
def mav(data):
        length = len(data)
        mav = np.sum(np.absolute(data))/length
        return mav


#Function to calulate sample entropy from input data
def sampEn(data):
        result = all(element == data[0] for element in data)
        if (result):
                data[0] = data[0]+0.0000001
        std = np.std(data)
        sampEN = ent.sample_entropy(data,2,0.2*std)
        for i in range(len(sampEN)):
                isnan = np.isnan(sampEN[i])
                isinf = math.isinf(sampEN[i])
                if isnan == True:
                        sampEN[i]= 0
                elif isinf == True:
                        sampEN[i]= 0
       
        return sampEN


#Function used to set the length of the feaure vector according to
#feature set used in order to create feature CSV of the right size
def feat_size(featset):
        if featset == 1:
                row = 4
                vec = 10*row
                return row, vec
        
        if featset == 2:
                row = 14
                vec = 10*row
                return row, vec
        if featset == 3:
                row = 41
                vec = 10*row
                return row, vec
        if featset == 4:
                row = 4
                vec = 10*row
                return row, vec
        if featset == 5:
                row = 2
                vec = 10*row
                return row, vec

        if featset == 6:
                row = 6
                vec = 10*row
                return row, vec
        
        if featset == 7:
                row = 4
                vec = 10*row
                return row, vec


#Function to create training, validation and feature CSV files  
def makecsv(data,subject):

        #define gestures to add to CSV files
        gesturelist = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

        #create training set CSV file
        f_name = "subject{}_train_data.csv".format(subject)  #create
        f = open(f_name,"w") #open CSV file
        writer = csv.writer(f)
        trainrep = [1,3,5,6,8,9,10] 
        for i in gesturelist:  #iterate through gestures
                for k in trainrep:   #iterate through gesture reptitions
                        window_num = 0
                        for window in data["gesture{}".format(i)]["repitition{}".format(k)]:
                                if window_num <1000:  # conditional statement to cap number samples for each gesture if desired
                                        row = []
                                        wl = list(window) 
                                        gest = [i]
                                        dat = gest + wl
                                        writer.writerow(dat) #add data window and label to CSV 
                                        window_num = window_num +1
                print(i,"finished")                
        f.close


        #create validation set CSV file
        f_name = "subject{}_validation_data.csv".format(subject) 
        f = open(f_name,"w") #open CSV file
        writer = csv.writer(f)
        val_rep = [2,4]
        for i in gesturelist:   #iterate through gestures
                for k in val_rep:   #iterate through gesture reptitions
                        window_num = 0
                        for window in data["gesture{}".format(i)]["repitition{}".format(k)]:
                                if window_num <1000:
                                        row = []
                                        wl = list(window)
                                        gest = [i]
                                        dat = gest + wl
                                        writer.writerow(dat) #add data window and label to CSV 
                                        window_num = window_num +1
        f.close
                
        #create test set CSV file
        f_name = "subject{}_test_data.csv".format(subject) 
        f = open(f_name,"w") #open CSV file
        writer = csv.writer(f)
        for i in gesturelist:
                k = 7
                window_num = 0
                for window in data["gesture{}".format(i)]["repitition{}".format(k)]:
                        if window_num <1000:
                                row = []
                                wl = list(window)
                                gest = [i]
                                dat = gest + wl
                                writer.writerow(dat) #add data window and label to CSV 
                                window_num = window_num +1
        f.close


#Function to create training, validation and test CSV files
#using extracted feature set data 
def makefeaturecsv(data,subject,featset):

        #define gestures to add to CSV files
        gesturelist = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        
        row_size, featmat_size = feat_size(featset) #change dimensions of row vectors to fit different feature sets

        #create training set feature CSV file
        f_name = "subject{}_train_feature_data.csv".format(subject) 
        f = open(f_name,"w") 
        writer = csv.writer(f)
        trainrep = [1,3,5,6,8,9,10]
        for i in gesturelist: 
                for k in trainrep:
                        for window in data["gesture{}".format(i)]["repitition{}".format(k)]:
                                featmat = np.empty((0,row_size))  #create structure to append feature vectors to
                                for row in window:
                                        feat = feature_extract(row,featset)
                                        featmat = np.vstack((featmat,feat)) #combine feature vectors for each electrode channel
                                featmat = np.array(featmat)
                                xx = featmat.reshape(featmat_size)
                                row = []
                                row.append(i)    
                                row.append(xx)  
                                writer.writerow(row) #add feature vectors and label to CSV
        f.close
        
        #create validation set feature CSV file
        f_name = "subject{}_validation_feature_data.csv".format(subject) 
        f = open(f_name,"w") 
        writer = csv.writer(f)
        val_rep = [2,4]
        for i in gesturelist:
                for k in val_rep:
                        for window in data["gesture{}".format(i)]["repitition{}".format(k)]:
                                featmat = np.empty((0,row_size))  #create structure to append feature vectors to
                                for row in window:
                                        feat = feature_extract(row,featset)
                                        featmat = np.vstack((featmat,feat)) #combine feature vectors for each electrode channel
                                featmat = np.array(featmat)
                                xx = featmat.reshape(featmat_size)
                                row = []
                                row.append(i)  
                                row.append(xx) 
                                writer.writerow(row) #add feature vectors and label to CSV
        f.close
        
        #create test set feature CSV file
        f_name = "subject{}_test_feature_data.csv".format(subject) 
        f = open(f_name,"w") 
        writer = csv.writer(f)
        for i in gesturelist:
                k = 7
                for window in data["gesture{}".format(i)]["repitition{}".format(k)]:
                        featmat = np.empty((0,row_size))  #create structure to append feature vectors to
                        for row in window:
                                feat = feature_extract(row,featset) 
                                featmat = np.vstack((featmat,feat)) #combine feature vectors for each electrode channel
                        featmat = np.array(featmat)
                        xx = featmat.reshape(featmat_size)
                        row = []
                        row.append(i)  
                        row.append(xx)  
                        writer.writerow(row) #add feature vectors and label to CSV
                f.close
        
#Function to read CSV files and create input and target arrays 
#for direct input into DL classifier models, takes CSV file and the 
#the type of CSV file e.g. "train" as inputs 

def inputstargets(subject,type):

        #define input and target lists for classifier input
        inputs= []
        targets=[]

        #open csv file specific to subject 
        data_file = open("subject{}_{}_data.csv".format(subject,type), 'r') #open CSV file from stored location
        data_list = list(csv.reader(data_file)) #read csv file
        data_file.close()

        #extract and convert values from csv file into a list of float input values 
        #and integer target values

        for data in data_list: #iterate through data windows stored in CSV
                window = []
                for j in range(1,11):
                        res = data[j].strip('][').split(' ')     
                        res2 = []
                        for a in res:
                                if a != '':
                                        float(a)
                                        res2.append(a)
                        res2 = np.asfarray(res2)
                        window.append(res2)
                inputs.append(window)
                gesture = int(data[0]) #extract gesture label from CSV
                targets.append(gesture)
        return inputs, targets

#Function to read feature CSV files and create input and target arrays 
#for direct input into ML classifier models, takes CSV file and the 
#the type of CSV file e.g. "train" as inputs 

def featinputstargets(subject,type):
        #define input and target lists for classifier input
        inputs= []
        targets=[] 
        #open csv file specific to subject 
        data_file = open("subject{}_{}_feature_data.csv".format(subject,type), 'r') #open CSV file from stored location
        data_list = list(csv.reader(data_file)) #read csv file
        data_file.close()

        #extract and convert values from csv file into a list of float input values 
        #and integer target values
        for data in data_list:  #iterate through feature vector windows stored in CSV
                res = data[1].strip('][').split(' ')     
                res2 = []
                for a in res:
                        if a != '':
                                float(a)
                                res2.append(a)
                res2 = np.asfarray(res2)
                inputs.append(res2)
                gesture = int(data[0])  #extract gesture label from CSV
                targets.append(gesture)
        return inputs, targets  

#Function to renumber gestures from DB1 Ex2 to be in 0-9 range
def asign_ex2_gesture(gesture):
        if gesture == 5:
                assigned = 1
        elif gesture == 6:
                assigned = 2
        elif gesture == 7:
                assigned = 3
        elif gesture == 11:
                assigned = 4
        elif gesture == 12:
                assigned = 5
        elif gesture == 13:
                assigned = 6
        elif gesture == 14:
                assigned = 7
        elif gesture == 15:
                assigned = 8
        elif gesture == 16:
                assigned = 9
        return assigned

#Function to renumber gestures from DB1 Ex3 to be in 10-14 range
def asign_ex3_gesture(gesture):
        if gesture == 1:
                assigned = 10
        elif gesture == 2:
                assigned = 11
        elif gesture == 4:
                assigned = 12
        elif gesture == 14:
                assigned = 13
        elif gesture == 17:
                assigned = 14
        return assigned
