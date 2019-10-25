#Imports required
#==========================================
#This code can only deal with numbers in columns
#==========================================
import pandas as pd #Spreedsheet data base
import seaborn as sns # plotting graphs
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier #Type of AI we use (Just a orgaising tool)
from sklearn.svm import SVC
from sklearn import svm # type of ai we use (Just a orgaising tool)
from sklearn.neural_network import MLPClassifier #type of ai we use (Just a orgaising tool)
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report # prints out reports
from sklearn.preprocessing import StandardScaler, LabelEncoder #scaler function we use
from sklearn.model_selection import train_test_split #split data up, we need to hide information of the program to learn

#Load Data sheet, (Needs to be placed in programming folder)
#Most CSV files use ; but since I played with this data its ,
data = pd.read_csv('opel_corsa.csv',sep=',')

#Prints out data from the CSV file and verables we have to work with
data.head()# prints out the 1st 5 rows of data
data.info()# prints out veriables we are working with in data set
#This tell us how many null values in each coloum
data.isnull().sum()

#Preprocessing data between (Good and bad data for this example)
bins = (2, 14.5, 30)# how many bins a n figures to divide from ////   (aount of bins, spread, index)
group_names = ['Bad', 'Good'] #anything less than 14.5 goes to bad, great than 14.5 goes to good, only name of bin for humans to read
data['FuelConsumptionAverage'] = pd.cut(data['FuelConsumptionAverage'], bins = bins, labels = group_names) # cutting out the "Fuel Consumption" and replacing it with the
data['FuelConsumptionAverage'].unique() # categories objects [labels X, label Y] Catergories (2, Object): [Label X < Label Y ]

#labels data bad = 0 good = 1
label_quality = LabelEncoder() # brought in with SKlearn lib, this encodes the data from above to label (Label X = 0, Label Y = 1)
data['FuelConsumptionAverage'] = label_quality.fit_transform(data['FuelConsumptionAverage'])# label "Veriable" to dataset with 0 or 1 depending on the above code lobelling
print(data.head(10))#prints out the data of the 10 rows, add a number between brackets to extend how many rows you want to see.
#Tell us the good fuel consumption vs bad
print(data['FuelConsumptionAverage'].value_counts())# total number of Label X and total number of label Y

#Plot out graph
sns.countplot(data['FuelConsumptionAverage'])#Graphs out data on a bar chart (Not really working here) plot label X and Label Y with repect to total number of rows

#Seperate the dataset as reponse variable and feature variables
X = data.drop('FuelConsumptionAverage', axis = 1)# here we are removing the labeled information we want the program to figure out by itself
y = data['FuelConsumptionAverage']#telling the program this is that information we want you to learn using all the columns in the above data set
#=========================================================================
#This is the magic part where we let the program hide data from itself
#=========================================================================
#Train and Test splitting data
#X_train = data set we are going to use to train Classifier with
#X_test = information we don't let he program see, so it can test itself
#y_train = the training data giving the answers we want
#y_test = data we test to see if program gives us the right answer
#Test split, part of SKlearn lib
#test size is the % of data to test
#Random state is just a random seed number, just grabs random numbers in the data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)

#Applying a stander scaler to optimise the results, imported with sklearn lib above
#fit transform is to fit the data better
sc = StandardScaler()
X_train = sc.fit_transform(X_train)#this reducs bias in columns that have high numbers or low numbers (changes most data in the X_training data to a value between 0 - 1)
X_test  = sc.transform(X_test)# we want to keep the train and test data the same
X_train[:10]#print out the 1st 10 data on the training data, this is just default


#==============================
#Random Forest Classifier
#Look mammy I'm programming AI, (Classifier with is a fancy way to say organise the data)
#used for medium size data set
#==============================
#Object label = Classifier (how many trees do you want)
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)# just does a simple fit of the data we seperated out for training
pred_rfc = rfc.predict(X_test)# predect the test values
#How the Forest Classifier preforms
print(classification_report(y_test, pred_rfc)) # how the test data compares to the predected values
print(confusion_matrix(y_test, pred_rfc))# this give us a matrix on the mislabels between good  and bad

#=================================
##SVM Classifier
#Support Vector Model
#Libary is pretty much the same as other libs
#=================================
clf = svm.SVC()#calling the function
clf.fit(X_train, y_train)# just does a simple fit of the data we seperated out for training
pred_clf = clf.predict(X_test)# predect the test values
#How the CLF model preformes
print(classification_report(y_test, pred_clf))# how the test data compares to the predected values
print(confusion_matrix(y_test, pred_clf))# this give us a matrix on the mislabels between good  and bad


#=================================
##Neural Network
#hidden layers is the nodes in the NN
#Good for text based code or big data sets, picture processing
#==================================
#object = Classifier(how many nodes in each layer, max many iterations
mlpc = MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=500)
mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict(X_test)
#How the NN model preformes
print(classification_report(y_test, pred_mlpc))# how the test data compares to the predected values
print(confusion_matrix(y_test, pred_mlpc))# this give us a matrix on the mislabels between good  and bad

#Score the AI
from sklearn.metrics import accuracy_score #Test scrore
bn = accuracy_score(y_test, pred_rfc) #Labelling code for printing
dm = accuracy_score(y_test, pred_clf) #Labelling code for printing
cm = accuracy_score(y_test, pred_mlpc) #Labelling code for printing
print(bn, ' is the Forest score')
print(dm, ' is the Classifier score')
print(cm, ' is the Neural Network score')