'''
Lab 3
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score

### LINKS FOR ADDITIONAL HELP ###
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# https://appdividend.com/2019/02/01/python-scikit-learn-tutorial-for-beginners-with-example/

######### Part 1 ###########

'''
    1) Download the iris-data-1 from Canvas, use pandas.read_csv to load it.

'''
# YOUR CODE GOES HERE
iris1 = pd.read_csv("iris-data-1.csv")

'''
    2) Split your data into test set(%30) and train set(%70) randomly. (Hint: you can use scikit-learn package tools for doing this)
    
'''
# YOUR CODE GOES HERE   
y = iris1.species
x = iris1.drop('species', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        
'''    
    3) Use KNeighborsClassifier from scikit-learn package. Train a KKN classifier using your training dataset  (K = 3, Euclidean distance).   
    
'''
# YOUR CODE GOES HERE  
knc = KNeighborsClassifier(n_neighbors=3,p=2,metric='euclidean')
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
print("micro-F1score: ", f1_score(y_test, y_predict, average='micro'))
print("macro-F1score: ", f1_score(y_test, y_predict, average='macro'))
print("weighted F1-score: ", f1_score(y_test, y_predict, average='weighted'))

'''    
    6) Repeat Q3, Q4, and Q5 for "manhattan" distance function

'''
# YOUR CODE GOES HERE
knc2 = KNeighborsClassifier(n_neighbors=3,p=1,metric='manhattan')
knc2.fit(x_train, y_train)

y_predict = knc2.predict(x_test)
print(classification_report(y_test, y_predict))

print("micro-F1score: ", f1_score(y_test, y_predict, average='micro'))
print("macro-F1score: ", f1_score(y_test, y_predict, average='macro'))
print("weighted F1-score: ", f1_score(y_test, y_predict, average='weighted'))

'''   
    7) Compare your results in Q5 and Q6.

'''
# YOUR CODE GOES HERE


'''
    8) Repeat Q3, Q4, Q5, Q6, and Q7 for K = 11.
'''
# YOUR CODE GOES HERE


######### Part 2 ###########
'''
    0)  Repeat Q1 and Q2 in part 1.

'''
# YOUR CODE GOES HERE


'''
    1) Train a KKN classifier using your training dataset  (K = 7, Euclidean distance). 
    
    1-1) Test your classifier using predict_proba method. What is the difference between predict_proba and predict method?
    
    1-2) report the performance based on your results in 1-1.
    
'''
# YOUR CODE GOES HERE



######### Part 3 ###########

'''
    0) Repeat Q1 and Q2 in part 1.

'''
# YOUR CODE GOES HERE

'''
    1) Use DecisionTreeClassifier from scikit-learn package. Train a DT classifier using your training dataset  (criterion='entropy', splitter= 'best'). 

'''
# YOUR CODE GOES HERE


'''   
    2) Test your classifier (Hint: use predict method) and report the performance (report accuracy, recall, precision, and F1score). (Hint: use classification_report from scikit learn)
'''

# YOUR CODE GOES HERE

'''   
    3) report micro-F1score, macro-F1score, and weighted F1-score
'''

# YOUR CODE GOES HERE

'''    
    4) Repeat Q1, Q2, and Q3 for "random" splitter.
'''
# YOUR CODE GOES HERE


'''   
    5) Compare your results in Q4 and Q3.

'''
# YOUR CODE GOES HERE

'''   
    6) Repeat Q2, Q3, Q4, and Q5 for criterion = "gini".

'''
# YOUR CODE GOES HERE