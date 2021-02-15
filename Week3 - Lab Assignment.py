'''
Lab 3
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

######### Part 1 ###########


'''
    1) Download the iris-data-1 from Canvas, use pandas.read_csv to load it.

'''
# YOUR CODE GOES HERE

df = pd.read_csv("iris-data-1.csv")

'''
    2) Split your data into test set(%30) and train set(%70) randomly. (Hint: you can use scikit-learn package tools for doing this)
    

'''
# YOUR CODE GOES HERE
y = df.species
x = df.drop('species', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7)
        
'''    
    3) Use KNeighborsClassifier from scikit-learn package. Train a KKN classifier using your training dataset  (K = 3, Euclidean distance).   
    
'''
# YOUR CODE GOES HERE  
# I am confused on Euclidean distance? :P

knc = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knc.fit(x_train, y_train)

'''   
    4) Test your classifier (Hint: use predict method) and report the performance (report accuracy, recall, precision, and F1score). (Hint: use classification_report from scikit learn)
'''

# YOUR CODE GOES HERE

y_predict = knc.predict(x_test)
print(classification_report(y_test, y_predict))

'''   

    5) report micro-F1score, macro-F1score, and weighted F1-score.
'''

# YOUR CODE GOES HERE
euMicro = f1_score(y_test, y_predict, average='micro')
euMacro = f1_score(y_test, y_predict, average='macro')
euWeighted = f1_score(y_test, y_predict, average='weighted')
print("micro-F1score: ", euMicro)
print("macro-F1score: ", euMacro)
print("weighted F1-score: ", euWeighted)
print("\n")

'''    
    6) Repeat Q3, Q4, and Q5 for "manhattan" distance function

'''
# YOUR CODE GOES HERE

knc = KNeighborsClassifier(n_neighbors=3, metric="manhattan")
knc.fit(x_train, y_train)

y_predict = knc.predict(x_test)
print(classification_report(y_test, y_predict))

maMicro = f1_score(y_test, y_predict, average='micro')
maMacro = f1_score(y_test, y_predict, average='macro')
maWeighted = f1_score(y_test, y_predict, average='weighted')
print("micro-F1score: ", maMicro)
print("macro-F1score: ", maMacro)
print("weighted F1-score: ", maWeighted)
print("\n")

'''   
    7) Compare your results in Q5 and Q6.

'''
# YOUR CODE GOES HERE
print("===========================================================================================")
print("EUCLIDEAN MICRO, MACRO, WEIGHTED: ", euMicro, euMacro, euWeighted)
print("MANHATTAN MICRO, MACRO, WEIGHTED: ", maMicro, maMacro, maWeighted)
print("\nDIFFERENCE IN VALUES MICRO, MACRO, WEIGHTED: ")
print(abs(euMicro - maMicro), abs(euMacro - maMacro), abs(euWeighted - maWeighted))
print("===========================================================================================\n")
'''
    8) Repeat Q3, Q4, Q5, Q6, and Q7 for K = 11.
'''
# YOUR CODE GOES HERE

'''euclidean'''

knc = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
knc.fit(x_train, y_train)

y_predict = knc.predict(x_test)
print(classification_report(y_test, y_predict))

euMicro = f1_score(y_test, y_predict, average='micro')
euMacro = f1_score(y_test, y_predict, average='macro')
euWeighted = f1_score(y_test, y_predict, average='weighted')
print("micro-F1score: ", euMicro)
print("macro-F1score: ", euMacro)
print("weighted F1-score: ", euWeighted)
print("\n")

''' manhattan '''
knc = KNeighborsClassifier(n_neighbors=3, metric="manhattan")
knc.fit(x_train, y_train)

y_predict = knc.predict(x_test)
print(classification_report(y_test, y_predict))

maMicro = f1_score(y_test, y_predict, average='micro')
maMacro = f1_score(y_test, y_predict, average='macro')
maWeighted = f1_score(y_test, y_predict, average='weighted')
print("micro-F1score: ", maMicro)
print("macro-F1score: ", maMacro)
print("weighted F1-score: ", maWeighted)
print("\n")

print("===========================================================================================")
print("EUCLIDEAN MICRO, MACRO, WEIGHTED: ", euMicro, euMacro, euWeighted)
print("MANHATTAN MICRO, MACRO, WEIGHTED: ", maMicro, maMacro, maWeighted)
print("\nDIFFERENCE IN VALUES MICRO, MACRO, WEIGHTED: ")
print(abs(euMicro - maMicro), abs(euMacro - maMacro), abs(euWeighted - maWeighted))
print("===========================================================================================\n")


######### Part 2 ###########
'''
    0)  Repeat Q1 and Q2 in part 1.

'''
# YOUR CODE GOES HERE

df = pd.read_csv("iris-data-1.csv")

y = df.species
x = df.drop('species', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7)

'''
    1) Train a KKN classifier using your training dataset  (K = 7, Euclidean distance). 
    
    1-1) Test your classifier using predict_proba method. What is the difference between predict_proba and predict method?
    
    1-2) report the performance based on your results in 1-1.
    
'''
# YOUR CODE GOES HERE

knc = KNeighborsClassifier(n_neighbors=7, metric="euclidean")
m = LogisticRegression()
m.fit(x_train, y_train)
print(m.predict_proba(x_test))

'''predict_proba gives the user the probabilities of the occurrence of each target 
predict methods outputs the target that is highest in predict_proba'''


######### Part 3 ###########

'''
    0) Repeat Q1 and Q2 in part 1.

'''
# YOUR CODE GOES HERE

df = pd.read_csv("iris-data-1.csv")

y = df.species
x = df.drop('species', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7)

'''
    1) Use DecisionTreeClassifier from scikit-learn package. Train a DT classifier using your training dataset  (criterion='entropy', splitter= 'best'). 

'''
# YOUR CODE GOES HERE

dec = DecisionTreeClassifier(criterion='entropy', splitter='best')
dec.fit(x_train, y_train)

'''   
    2) Test your classifier (Hint: use predict method) and report the performance (report accuracy, recall, precision, and F1score). (Hint: use classification_report from scikit learn)
'''

# YOUR CODE GOES HERE

y_predict = dec.predict(x_test)
print("\n", classification_report(y_test, y_predict))

'''   
    3) report micro-F1score, macro-F1score, and weighted F1-score
'''

# YOUR CODE GOES HERE
decMicro = f1_score(y_test, y_predict, average='micro')
decMacro = f1_score(y_test, y_predict, average='macro')
decWeighted = f1_score(y_test, y_predict, average='weighted')
print("micro-F1score: ", decMicro)
print("macro-F1score: ", decMacro)
print("weighted F1-score: ", decWeighted)
print("\n")

'''    
    4) Repeat Q1, Q2, and Q3 for "random" splitter.
'''
# YOUR CODE GOES HERE
dec = DecisionTreeClassifier(criterion='entropy', splitter='random')
dec.fit(x_train, y_train)

y_predict = dec.predict(x_test)
print("\n", classification_report(y_test, y_predict))

decRanMicro = f1_score(y_test, y_predict, average='micro')
decRanMacro = f1_score(y_test, y_predict, average='macro')
decRanWeighted = f1_score(y_test, y_predict, average='weighted')
print("micro-F1score: ", decRanMicro)
print("macro-F1score: ", decRanMacro)
print("weighted F1-score: ", decRanWeighted)
print("\n")

'''   
    5) Compare your results in Q4 and Q3.

'''
# YOUR CODE GOES HERE

print("===========================================================================================")
print("SPLITTER BEST: MICRO, MACRO, WEIGHTED: ", decMicro, decMacro, decWeighted)
print("SPLITTER RANDOM: MACRO, WEIGHTED: ", decRanMicro, decRanMacro, decRanWeighted)
print("\nDIFFERENCE IN VALUES MICRO, MACRO, WEIGHTED: ")
print(abs(decMicro - decRanMicro), abs(decMacro - decRanMacro), abs(decWeighted - decRanWeighted))
print("===========================================================================================")

'''   
    6) Repeat Q2, Q3, Q4, and Q5 for criterion = "gini".

'''
# YOUR CODE GOES HERE

dec = DecisionTreeClassifier(criterion='gini', splitter='best')
dec.fit(x_train, y_train)

y_predict = dec.predict(x_test)
print("\n", classification_report(y_test, y_predict))

decMicro = f1_score(y_test, y_predict, average='micro')
decMacro = f1_score(y_test, y_predict, average='macro')
decWeighted = f1_score(y_test, y_predict, average='weighted')
print("micro-F1score: ", decMicro)
print("macro-F1score: ", decMacro)
print("weighted F1-score: ", decWeighted)

dec = DecisionTreeClassifier(criterion='gini', splitter='random')
dec.fit(x_train, y_train)

y_predict = dec.predict(x_test)
print("\n", classification_report(y_test, y_predict))

decRanMicro = f1_score(y_test, y_predict, average='micro')
decRanMacro = f1_score(y_test, y_predict, average='macro')
decRanWeighted = f1_score(y_test, y_predict, average='weighted')
print("micro-F1score: ", decRanMicro)
print("macro-F1score: ", decRanMacro)
print("weighted F1-score: ", decRanWeighted)
print("\n")

print("===========================================================================================")
print("GINI BEST: MICRO, MACRO, WEIGHTED: ", decMicro, decMacro, decWeighted)
print("GINI RANDOM: MICRO, MACRO, WEIGHTED: ", decRanMicro, decRanMacro, decRanWeighted)
print("\nDIFFERENCE IN VALUES MICRO, MACRO, WEIGHTED: ")
print(abs(decMicro - decRanMicro), abs(decMacro - decRanMacro), abs(decWeighted - decRanWeighted))
print("===========================================================================================\n")
