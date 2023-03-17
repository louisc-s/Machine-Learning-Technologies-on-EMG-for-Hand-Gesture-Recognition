
import scipy.io as spio
import numpy as np
from Functions import windowmaker, makecsv, makefeaturecsv, asign_ex2_gesture, asign_ex3_gesture


##### Sort ###########

# This script loads and segments the sEMG data from the Matlab files provided by the Ninapro DB1 database 
# for each of the 27 subjects and produces three CSV files for each subject: training, validation and test.
# An additional three CSV files are also created containing the extracted features of the specified feature set


#create dictionaries,lists and arrays for storing and sorting data values
emg_data = {}
spec_emg_data = {}
endlist = []
#list of desired gestures for classification within each exercise 
EX2_gest = [5,6,7,11,12,13,14,15,16]
EX3_gest = [1,2,4,14,17]

#define number of subjects
subject_list = list(range(1,28))

#iterate through all subject in database and load subject EMG matlab files into dictionaries
for i in subject_list: 
    emg_data["subject{}".format(i)]={}
    emg_data["subject{}".format(i)]["ex2"]= spio.loadmat("//Volumes/KINGSTON/Data/Ninapro/db1/s{}/S{}_A1_E2.mat".format(i,i),variable_names = "emg") #load files from stored location 
    emg_data["subject{}".format(i)]["ex3"]= spio.loadmat("//Volumes/KINGSTON/Data/Ninapro/db1/s{}/S{}_A1_E3.mat".format(i,i),variable_names = "emg")#load files from stored location 
    

    #create nested dictionaries to store gesture classes for corresponding EMG data and 
    #specific EMG data for each gesture for each subject 

    #access gesure classes for each subject
    emg_data["subject{}".format(i)]["ex2"]["r"] = {}
    emg_data["subject{}".format(i)]["ex2"]["r"]= spio.loadmat("//Volumes/KINGSTON/Data/Ninapro/db1/s{}/S{}_A1_E2.mat".format(i,i),variable_names= "restimulus")
    
    spec_emg_data["subject{}".format(i)]={} #instantiate nested subject specific EMG dicitionary 

    #for each desired gesture in EX2 for specific subject store  
    # corresponding segmented EMG data in nested dictionary 
    for l in EX2_gest:
        a = asign_ex2_gesture(l) #use function to renumber ex2 gestures between 0-9
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)] = {}
        loclist = [] #create list to store positions of specific gesture data within exercise 1 data 
        loc = np.where(emg_data["subject{}".format(i)]["ex2"]["r"]["restimulus"]==l) #search through class data to find locations of specific gesture data within exercise 2 data 
        #determine locations of sEMG data for each repetition of the gesture
        loclist.append(loc[0][0]) 
        for x,y in zip(loc[0][::],loc[0][1::]): #find gesture locations at the border between the gesture and rest 
            if y-x != 1:    
                loclist.append(x)
                loclist.append(y)
        loclist.append(loc[0][-1])#append last value
        
        #use gesture 5 positions in order to extract rest position data as there is no specific 
        # exercise for rest data hence it must be extracted from breaks between gesture repetitions
        if l ==5:
            restlist = loclist
            #add rest data to nested dictionary taken from rests periods between repetitions of the first gesture
            spec_emg_data["subject{}".format(i)]["gesture0"] = {}
            
            #segement each rest gesture repetition using "windowmaker" function and store in specific nested dictionary entry
            spec_emg_data["subject{}".format(i)]["gesture0"]["repitition1"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][restlist[1]+1:restlist[2]])
            spec_emg_data["subject{}".format(i)]["gesture0"]["repitition2"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][restlist[3]+1:restlist[4]])        
            spec_emg_data["subject{}".format(i)]["gesture0"]["repitition3"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][restlist[5]+1:restlist[6]])
            spec_emg_data["subject{}".format(i)]["gesture0"]["repitition4"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][restlist[7]+1:restlist[8]])
            spec_emg_data["subject{}".format(i)]["gesture0"]["repitition5"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][restlist[9]+1:restlist[10]])
            spec_emg_data["subject{}".format(i)]["gesture0"]["repitition6"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][restlist[11]+1:restlist[12]])
            spec_emg_data["subject{}".format(i)]["gesture0"]["repitition7"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][restlist[13]+1:restlist[14]])
            spec_emg_data["subject{}".format(i)]["gesture0"]["repitition8"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][restlist[15]+1:restlist[16]])
            spec_emg_data["subject{}".format(i)]["gesture0"]["repitition9"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][restlist[17]+1:restlist[18]])
            
            
        #retreive 10th rest repetition from gesture 6 data
        if l ==6:
            restlist = loclist
            spec_emg_data["subject{}".format(i)]["gesture0"]["repitition10"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][restlist[1]+1:restlist[2]])
        

        #segement each gesture repetition using "windowmaker" function and store it in specific nested dictionary entry
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)]["repitition1"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][loclist[0]:loclist[1]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)]["repitition2"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][loclist[2]:loclist[3]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)]["repitition3"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][loclist[4]:loclist[5]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)]["repitition4"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][loclist[6]:loclist[7]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)]["repitition5"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][loclist[8]:loclist[9]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)]["repitition6"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][loclist[10]:loclist[11]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)]["repitition7"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][loclist[12]:loclist[13]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)]["repitition8"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][loclist[14]:loclist[15]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)]["repitition9"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][loclist[16]:loclist[17]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)]["repitition10"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][loclist[18]:loclist[19]+1])

        
    #delete unused variables to increase computational efficiency
    del emg_data["subject{}".format(i)]["ex2"]

    ## repeat same process for EX3 gestures ##
    
    #access gesture classes for each subject
    emg_data["subject{}".format(i)]["ex3"]["r"]= {}
    emg_data["subject{}".format(i)]["ex3"]["r"] = spio.loadmat("//Volumes/KINGSTON/Data/Ninapro/db1/s{}/S{}_A1_E3.mat".format(i,i),variable_names = "restimulus")
    
    
    for j in EX3_gest:
        b = asign_ex3_gesture(j) #use function to renumber ex3 gestures between 10-14
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)] = {}
        loclist = []#create list to store positions of specific gesture data within exercise 3 data
        loc = np.where(emg_data["subject{}".format(i)]["ex3"]["r"]["restimulus"]==j) #search through class data to find locations of specific gesture data within exercise 3 data 
        #determine locations of sEMG data for each reptiion of the gesture
        loclist.append(loc[0][0])
        for x,y in zip(loc[0][::],loc[0][1::]):#find gesture locations at the border between the gesture and rest
            if y-x != 1:
                loclist.append(x)
                loclist.append(y)
        
        loclist.append(loc[0][-1]) #append last value
    
        #segement each gesture repetition using "windowmaker" function and store in specific nested dictionary entry
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)]["repitition1"] = windowmaker(emg_data["subject{}".format(i)]["ex3"]["emg"][loclist[0]:loclist[1]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)]["repitition2"] = windowmaker(emg_data["subject{}".format(i)]["ex3"]["emg"][loclist[2]:loclist[3]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)]["repitition3"] = windowmaker(emg_data["subject{}".format(i)]["ex3"]["emg"][loclist[4]:loclist[5]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)]["repitition4"] = windowmaker(emg_data["subject{}".format(i)]["ex3"]["emg"][loclist[6]:loclist[7]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)]["repitition5"] = windowmaker(emg_data["subject{}".format(i)]["ex3"]["emg"][loclist[8]:loclist[9]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)]["repitition6"] = windowmaker(emg_data["subject{}".format(i)]["ex3"]["emg"][loclist[10]:loclist[11]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)]["repitition7"] = windowmaker(emg_data["subject{}".format(i)]["ex3"]["emg"][loclist[12]:loclist[13]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)]["repitition8"] = windowmaker(emg_data["subject{}".format(i)]["ex3"]["emg"][loclist[14]:loclist[15]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)]["repitition9"] = windowmaker(emg_data["subject{}".format(i)]["ex3"]["emg"][loclist[16]:loclist[17]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)]["repitition10"] = windowmaker(emg_data["subject{}".format(i)]["ex3"]["emg"][loclist[18]:loclist[19]+1])

    #delete unused variables to increase computational efficiency
    del emg_data["subject{}".format(i)]
    
    #create training, validation and test CSV files contianing segemented sEMG gesture data
    makecsv(spec_emg_data["subject{}".format(i)],i)

    #create training, validation and test CSV files contianing segemented, 
    #feature extracted sEMG gesture data for specified feature set - in this case set 4 (SampEn Pipeline)
    makefeaturecsv(spec_emg_data["subject{}".format(i)],i,4)
    
   #delete surplus variables to increase computational efficiency
    del spec_emg_data["subject{}".format(i)]

    



