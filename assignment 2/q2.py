import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn
from sklearn import tree
from sklearn import metrics
import pydotplus
import graphviz
from IPython.display import Image
from sklearn.model_selection import GridSearchCV


def preprocessing(csvfile): #function for preprocessing

    df=pd.read_csv(csvfile)         #reading csv file
    df = (df - df.min()) / (df.max() - df.min())  #normalizing

    return df



def split_rows(dataframe):        #function to split dataframe
    return dataframe.iloc[0:620],dataframe.iloc[620:887]



def train_test_split(df):             #function to splitting dataset
    X_train, X_test=split_rows(df)        # seperating to test and train
    y_train= X_train.pop('Survived')
    y_test= X_test.pop('Survived')
    y_train= y_train.to_frame()
    y_test = y_test.to_frame()

    return X_train, X_test, y_train, y_test


y=preprocessing('titanic.csv')
print(y)

print("columns",y.index)


# Get a bool series representing which row satisfies the condition i.e. True for
# row in which value of 'Age' column is more than 30

survived_data= y.loc[y['Survived'] == 1]
not_survived_data=y.loc[y['Survived']==0]
print('survived',survived_data)
print("len of full data",len(y))
print("len of P(s=true)",len(survived_data))
print("len of P(s=false)",len(not_survived_data))
seriesObj_1 = survived_data.loc[survived_data['Sex'] == 1]
seriesObj_11=seriesObj_1.loc[seriesObj_1['Pclass']==0]
seriesObj_2 = not_survived_data.loc[not_survived_data['Sex'] == 1]
seriesObj_21=seriesObj_2.loc[seriesObj_2['Pclass']==0]


print(' G=female,C=1 | S= true ', len(seriesObj_11))
print(' G=female,C=1 | S= false ', len(seriesObj_21))




y.to_csv('preprocessed.csv')


x_train,x_test,y_train,y_test = train_test_split(y)
x_train.to_csv('x_train.csv')
x_test.to_csv('x_test.csv')
y_train.to_csv('y_train.csv')
y_test.to_csv('y_test.csv')
#print(y)

x=pd.concat([x_train,x_test])
y=pd.concat([y_train,y_test])
print(x,y)


# The decision tree classifier.
clf = tree.DecisionTreeClassifier()
# Training the Decision Tree
clf_train = clf.fit(x_train, y_train)

y_pred_test=clf_train.predict(x_test)
y_pred_train=clf_train.predict(x_train)


print("test Accuracy:",metrics.accuracy_score(y_test, y_pred_test))
print("train accuracry:",metrics.accuracy_score(y_train,y_pred_train))




# Export/Print a decision tree in DOT format.
#print(tree.export_graphviz(clf_train, None))

#Create Dot Data
dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(x_train.columns.values),
                                class_names=['Not survived', 'survived'], rounded=True, filled=True) #Gini decides which attribute/feature should be placed at the root node, which features will act as internal nodes or leaf nodes
#Create Graph from DOT data
graph = pydotplus.graph_from_dot_data(dot_data)




param_grid = { 'min_samples_leaf':list(range(2, 21))}
t=tree.DecisionTreeClassifier()
#print(metrics.SCORERS.keys())

grid_search_tree = GridSearchCV(estimator = t,scoring ='roc_auc' , param_grid = param_grid, cv = 10, n_jobs = -1, verbose = 2)
grid_search_tree.fit(x, y)
print("best parameter",grid_search_tree.best_params_)
best_grid = grid_search_tree.best_estimator_
y_pred=best_grid.predict(x_test)
print("bestscore,",grid_search_tree.best_score_)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("????")
#print(grid_search_tree.cv_results_)

min_leaf=[]
test_accuracy=[]
train_accuracy=[]
best=0

for i in range(2,21):   #?? loop for finding optimal min number of leafs
    # The decision tree classifier.
    dt = tree.DecisionTreeClassifier(min_samples_leaf=i)
    # Training the Decision Tree
    dt_trained = dt.fit(x_train, y_train)

    y_pred_test = dt_trained.predict(x_test)
    y_pred_train= dt_trained.predict(x_train)

    #print(i)

    if (metrics.roc_auc_score(y_test, y_pred_test) > best):
        optimal_leaf = i
        best=metrics.roc_auc_score(y_test, y_pred_test)
    test_accuracy.append(metrics.roc_auc_score(y_test, y_pred_test))
    min_leaf.append(i)
    train_accuracy.append(metrics.roc_auc_score(y_train, y_pred_train))

    print("\n")

print("optimal_leaf:",optimal_leaf)

print("test_accuracy",test_accuracy)
print("train_accuracy",train_accuracy)


plt.plot(min_leaf, train_accuracy,marker='o', color='blue')
plt.xlabel('min sample leaf')
plt.ylabel('AUC accuracy')
plt.title('training set AUC score')
plt.xticks(np.arange(2, 21, 1.0))
plt.show()