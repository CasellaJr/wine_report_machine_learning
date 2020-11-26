#!/usr/bin/env python
# coding: utf-8

# In[1]:


#BRUNO CASELLA
#Classification using scikit-learn

#Dataset: load_wine(*[, return_X_y, as_frame])

#Provide a comparison of a several classifiers in scikit-learn using data set “wine”.

#Plotting the result using two different color for training points and testing points
#Make a graph of the “score” obtained by varying a parameter of your choice.


# In[2]:


#import of libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


from sklearn.datasets import load_wine 


# In[4]:


wine_data = load_wine() #load of the dataset


# In[5]:


wine_data #to see the data


# In[6]:


wine_data.feature_names #a better view of the feature names of the dataset


# In[7]:


wine_data.target_names #and of the target names


# In[8]:


print(wine_data.DESCR) #and now let's see a general description of the characteristics


# In[9]:


#organize the data into a dataframe; in this way I organize by columns, that are the feature names
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
wine_df.head() #and i print the first 5 rows


# In[10]:


wine_df.tail() #and the last 5 rows


# In[11]:


wine_df.shape 


# In[12]:


#now I add the target variable (the quality) to the dataframe, 
#and then I print the first 5 rows and I check the shape to see it is correct
wine_df['Target']= wine_data.target
wine_df.head()


# In[13]:


wine_df.shape #yes, there is one more column, that is the target variable


# In[14]:


# I plot the conditional distributions of each variable (given target value) 
for name in wine_df.columns:   
    if name!="Target":
        plt.figure()
        sns.kdeplot(wine_df.loc[wine_df["Target"]==0, name], shade=True, color="g",label="class=0")
        sns.kdeplot(wine_df.loc[wine_df["Target"]==1, name], shade=True, color="b",label="class=1")
        sns.kdeplot(wine_df.loc[wine_df["Target"]==2, name], shade=True, color="y",label="class=2")
        plt.title(name + " class conditional densities")
        plt.legend()
        plt.show()


# In[15]:


#import of other libraries
from sklearn import tree
from sklearn.model_selection import train_test_split                         
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from graphviz import Source


# In[17]:


#I split my dataset into train and test set
array = wine_df.values
X = array[:, 0:13] #all rows and all columns except the target
Y = array[:, 13] #all rows, but only the target column (the quality)
test_size=.30 #70% to train dataset, and 30% to test set
set_seed= 42 #in order to allor the public to replicate the same results
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=test_size, random_state= set_seed)
#Standardization is the process of subtracting the means from each feature and then dividing by the feature standard deviations.
#Standardization is a common requirement for machine learning tasks. 
#Many algorithms assume that all features are centered around zero and have approximately the same variance.
#for more info: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html?highlight=wine
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[18]:


#just showing values and shape, for a quick check of correctness
print(Y)
print(X_train.shape)
print(Y_train.shape)
print(X_train)
print(Y_train)


# In[19]:


#Now I start to using different classifiers, starting from the KNN

#KNearest Neighbors Classifier
n5 = KNeighborsClassifier() #trial with 5 as number of neighbors; 5 is n_neighbors of default;
#by default the weights parameter is equal to 'uniform'; another option of weights is 'distance';  
n5.fit(X_train, Y_train) #fit the model using X_train as training data and Y_train as target values
#Now that our model is instantiated and fitted to our training data, let's dive right in with making some predictions. 
pred_n5 = n5.predict(X_test) #Predict the class labels for the provided data.
print(pred_n5)
print(n5.score(X_test, Y_test)) #The score method returns the mean accuracy on the given test data and labels.
print(classification_report(Y_test, pred_n5))


# In[20]:


#KNearest Neighbors Classifier
#I do the same steps of before, but now I try changing the number of neighbors
n3 = KNeighborsClassifier(n_neighbors = 3) #trial with 3 as number of neighbors
n3.fit(X_train, Y_train)
pred_n3 = n3.predict(X_test)
print(pred_n3)
print(n3.score(X_test, Y_test))
print(classification_report(Y_test, pred_n3)) #we obtain the same results
#Doing other attempts we can see how precision and recall (the sensitivity) change, and in consequence
#how change the F1-score, because it depends on both of them.


# In[22]:


#One last thing we can play with is seeing which accuracies work best for our model's fitness. 
neighbors = np.arange(1, 25) #attempts from 1 neighbour to a max of 25 neighbours
train_accuracy, test_accuracy = list(), list()

#a for cycle that for all the 1 to 25 values of neighbors, instantiate the model, then fit the model and calculate the scores
for iterator, kterator in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=kterator)
    knn.fit(X_train, Y_train)
    train_accuracy.append(knn.score(X_train, Y_train))
    test_accuracy.append(knn.score(X_test, Y_test))

plt.figure(figsize=[13, 8]) #figsize is a tuple of the width and height of the figure in inches
plt.plot(neighbors, test_accuracy, label="Testing Accuracy")
plt.plot(neighbors, train_accuracy, label="Training Accuracy")
plt.legend()
plt.title("Value VS Accuracy")
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.xticks(neighbors)
plt.savefig("knn_accuracies.png")
plt.show()

print("Best Accuracy is {} with K={}".format(np.max(test_accuracy), 1 + test_accuracy.index(np.max(test_accuracy))))


# In[23]:


#From the previous analysis we can see that the best accuracy is with K = 20
#So now, I check the results using K = 20
n8 = KNeighborsClassifier(n_neighbors = 20) 
n8.fit(X_train, Y_train)
pred_n8 = n8.predict(X_test)
print(pred_n8)
print(n8.score(X_test, Y_test))
print(classification_report(Y_test, pred_n8))


# In[24]:


#Now, I will plot the decision boundaries for each class, in the best case, so with n_neighbors=20
n_neighbors = 20

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset. It is a mandatory step because dimensions must be correct
X = X_train[:, :2]
Y = Y_train

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, Y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h)) #meshgrid returns coordinate matrices from coordinate vectors.
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) 
    #np.c translates slice objects to concatenation along the second axis.
    #ravel returns a contiguous flattened array. A 1-D array, containing the elements of the input, is returned

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure() #create a new figure
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto') #Create a pseudocolor plot with a non-regular rectangular grid.
    #shading = 'auto' is used to avoid a warning. Indeed, If shading='flat' (the default value) the dimensions of X and Y should be one greater 
    #than those of C, and the quadrilateral is colored due to the value at C[i, j]. If X, Y and C have equal dimensions, a warning will be raised and the last
    #row and column of C will be ignored.
    
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold,
                edgecolor='k', s=20) 
    #A scatter plot of y vs. x
    #the c parameter is a scalar or sequence of n numbers to be mapped to colors using cmap and norm
    plt.xlim(xx.min(), xx.max()) #Get or set the x limits of the current axes.
    plt.ylim(yy.min(), yy.max()) #Get or set the y-limits of the current axes.
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()


# In[25]:


#Now I use a different classifier, the RadiusNeighborsClassifier with radius 4.0. Steps are always the same.
rad = RadiusNeighborsClassifier(radius = 4.0) #with lower values of radius the function does not work.
rad.fit(X_train, Y_train)
pred_rad = rad.predict(X_test)
print(pred_rad)
print(rad.score(X_test, Y_test))
print(classification_report(Y_test, pred_rad))


# In[30]:


#Next Classifier: Random Forests with the default number of estimators (the number of trees in the forest) that is 100.
#The criterion (The function to measure the quality of a split.) I used is the Gini (by default). Another option is "entropy"
#I don't specify the maximum depth of the tree, then the nodes are expanded until all leaves are pure or until all leaves
#contain less than min_samples_split samples.
rf = RandomForestClassifier(random_state = set_seed) #random_state controls the randomness of the estimator.
rf.fit(X_train, Y_train)
pred_rf = rf.predict(X_test)
print(pred_rf)
print(rf.score(X_test, Y_test))
print(classification_report(Y_test, pred_rf))
#Doing other home-attempts the number of estimators to maintain the same accuracy is 41, that is obviously less expensive.


# In[31]:


#Decision Trees. Here the same parameters of the Random Forests, so Gini criterion and max_depth not specified.
dt = DecisionTreeClassifier(random_state=set_seed) #random_state controls the randomness of the estimator.
dt.fit(X_train, Y_train)
pred_dt = dt.predict(X_test)
print(pred_dt)
print(dt.score(X_test, Y_test))
print(classification_report(Y_test, pred_dt))


# In[32]:


# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    #So the variables considered are Alcohol, Malic Acid, Ash and Alcalinity of Ash
    # We only take the two corresponding features
    X = X_train[:, pair]
    y = Y_train

    # Train
    clf = DecisionTreeClassifier().fit(X, y)

    # Plot the decision boundary
    plt.subplot(2, 3, pairidx + 1) 
    #Create a plot of 2 rows and 3 columns (nrows, ncols, index). 
    #Index starts at 1 in the upper left corner and increases to the right. 

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5) #Adjust the padding between and around subplots

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu) #Plot contours.

    plt.xlabel(wine_data.feature_names[pair[0]])
    plt.ylabel(wine_data.feature_names[pair[1]])

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=wine_data.target_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")

plt.figure()
clf = DecisionTreeClassifier().fit(wine_data.data, wine_data.target)
plot_tree(clf, filled=True)
plt.show()

#to export in Wine.pdf
import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=wine_data.feature_names,  
                     class_names=wine_data.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data) 
graph.render("Wine") 
graph


# In[33]:


#Time for Extra Trees Classifier. 
#Same parameters of Random Forests: n_estimators = 100, Gini criterion and max_depth not specified.
et = ExtraTreesClassifier(random_state = set_seed)
et.fit(X_train, Y_train)
pred_et = et.predict(X_test)
print(pred_et)
print(et.score(X_test, Y_test))
print(classification_report(Y_test, pred_et))


# In[34]:


#Now, I show the decision boundaries of different classifiers: DecisionTreeClassifier, RandomForestClassifier and ExtraTreesClassifier
#This plot compares the decision surfaces learned by a decision tree classifier (first column), by a random forest classifier (second column)
#and by an extra-trees classifier (third column).
#In the first row, the classifiers are built using the Alcohol and the Malic Acid features only, 
#on the second row using the Alcohol and Ash only, and on the third row using the Ash and the Alcalinity of Ash only.
#I use parameters to obtain the best results (max_depth=None for DecisionTrees and n_estimators=100 for Random Forests and ExtraTrees)
#In the next example I will try changing these parameters.

# Parameters
n_classes = 3
cmap = plt.cm.RdYlBu
plot_step = 0.02  # fine step width for decision surface contours
plot_step_coarser = 0.5  # step widths for coarse classifier guesses
RANDOM_SEED = 13  # fix the seed on each iteration

plot_idx = 1

models = [DecisionTreeClassifier(max_depth=None),
          RandomForestClassifier(),
          ExtraTreesClassifier()]

for pair in ([0, 1], [0, 2], [2, 3]):
    for model in models:
        # We only take the two corresponding features
        X = X_train[:, pair]
        y = Y_train

        # Shuffle
        idx = np.arange(X.shape[0])
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Standardize
        #mean = X.mean(axis=0)
        #std = X.std(axis=0)
        #X = (X - mean) / std

        # Train
        model.fit(X, y)

        scores = model.score(X, y)
        # Create a title for each column and the console by using str() and
        # slicing away useless parts of the string
        model_title = str(type(model)).split(
            ".")[-1][:-2][:-len("Classifier")]

        model_details = model_title
        if hasattr(model, "estimators_"):
            model_details += " with {} estimators".format(
                len(model.estimators_))
        print(model_details + " with features", pair,
              "has a score of", scores)

        plt.subplot(3, 3, plot_idx)
        if plot_idx <= len(models):
            # Add a title at the top of each column
            plt.title(model_title, fontsize=9)

        # Now plot the decision boundary using a fine mesh as input to a
        # filled contour plot
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        # Plot either a single DecisionTreeClassifier or alpha blend the
        # decision surfaces of the ensemble of classifiers
        if isinstance(model, DecisionTreeClassifier):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=cmap)
        else:
            # Choose alpha blend level with respect to the number
            # of estimators
            # that are in use (noting that AdaBoost can use fewer estimators
            # than its maximum if it achieves a good enough fit early on)
            estimator_alpha = 1.0 / len(model.estimators_)
            for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

        # Build a coarser grid to plot a set of ensemble classifications
        # to show how these are different to what we see in the decision
        # surfaces. These points are regularly space and do not have a
        # black outline
        xx_coarser, yy_coarser = np.meshgrid(
            np.arange(x_min, x_max, plot_step_coarser),
            np.arange(y_min, y_max, plot_step_coarser))
        Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(),
                                         yy_coarser.ravel()]
                                         ).reshape(xx_coarser.shape)
        cs_points = plt.scatter(xx_coarser, yy_coarser, s=15,
                                c=Z_points_coarser, cmap=cmap,
                                edgecolors="none")

        # Plot the training points, these are clustered together and have a
        # black outline
        plt.scatter(X[:, 0], X[:, 1], c=y,
                    cmap=ListedColormap(['r', 'y', 'b']),
                    edgecolor='k', s=20)
        plot_idx += 1  # move on to the next plot in sequence

plt.suptitle("Classifiers on feature subsets of the Wine dataset", fontsize=12)
plt.axis("tight")
plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)
plt.show()


# In[35]:


#I obtained perfect results (score=1.0) in all the previous cases. So I varied the max_depth for the DecisionTreeClassifier,
#and the n_estimators for Random Forest and Extra Trees.
#Again, this plot compares the decision surfaces learned by a decision tree classifier (first column), by a random forest classifier (second column)
#and by an extra-trees classifier (third column).
#In the first row, the classifiers are built using the Alcohol and the Malic Acid features only, 
#on the second row using the Alcohol and Ash only, and on the third row using the Ash and the Alcalinity of Ash only.

# Parameters
n_classes = 3
cmap = plt.cm.RdYlBu
plot_step = 0.02  # fine step width for decision surface contours
plot_step_coarser = 0.5  # step widths for coarse classifier guesses
RANDOM_SEED = 13  # fix the seed on each iteration

plot_idx = 1

models = [DecisionTreeClassifier(max_depth=5),
          RandomForestClassifier(n_estimators=30),
          ExtraTreesClassifier(n_estimators=30)]

for pair in ([0, 1], [0, 2], [2, 3]):
    for model in models:
        # We only take the two corresponding features
        X = X_train[:, pair]
        y = Y_train

        # Shuffle
        idx = np.arange(X.shape[0])
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Standardize
        #mean = X.mean(axis=0)
        #std = X.std(axis=0)
        #X = (X - mean) / std

        # Train
        model.fit(X, y)

        scores = model.score(X, y)
        # Create a title for each column and the console by using str() and
        # slicing away useless parts of the string
        model_title = str(type(model)).split(
            ".")[-1][:-2][:-len("Classifier")]

        model_details = model_title
        if hasattr(model, "estimators_"):
            model_details += " with {} estimators".format(
                len(model.estimators_))
        print(model_details + " with features", pair,
              "has a score of", scores)

        plt.subplot(3, 3, plot_idx)
        if plot_idx <= len(models):
            # Add a title at the top of each column
            plt.title(model_title, fontsize=9)

        # Now plot the decision boundary using a fine mesh as input to a
        # filled contour plot
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        # Plot either a single DecisionTreeClassifier or alpha blend the
        # decision surfaces of the ensemble of classifiers
        if isinstance(model, DecisionTreeClassifier):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=cmap)
        else:
            # Choose alpha blend level with respect to the number
            # of estimators
            # that are in use (noting that AdaBoost can use fewer estimators
            # than its maximum if it achieves a good enough fit early on)
            estimator_alpha = 1.0 / len(model.estimators_)
            for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

        # Build a coarser grid to plot a set of ensemble classifications
        # to show how these are different to what we see in the decision
        # surfaces. These points are regularly space and do not have a
        # black outline
        xx_coarser, yy_coarser = np.meshgrid(
            np.arange(x_min, x_max, plot_step_coarser),
            np.arange(y_min, y_max, plot_step_coarser))
        Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(),
                                         yy_coarser.ravel()]
                                         ).reshape(xx_coarser.shape)
        cs_points = plt.scatter(xx_coarser, yy_coarser, s=15,
                                c=Z_points_coarser, cmap=cmap,
                                edgecolors="none")

        # Plot the training points, these are clustered together and have a
        # black outline
        plt.scatter(X[:, 0], X[:, 1], c=y,
                    cmap=ListedColormap(['r', 'y', 'b']),
                    edgecolor='k', s=20)
        plot_idx += 1  # move on to the next plot in sequence

plt.suptitle("Classifiers on feature subsets of the Wine dataset", fontsize=12)
plt.axis("tight")
plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)
plt.show()


# In[36]:


#Finally I tried the family of naive Bayes Classifiers, starting with the Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
pred_gnb = gnb.predict(X_test)
print(pred_gnb)
print(gnb.score(X_test, Y_test))
print(classification_report(Y_test, pred_gnb))


# In[37]:


#Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train, Y_train)
pred_bnb = bnb.predict(X_test)
print(pred_bnb)
print(bnb.score(X_test, Y_test))
print(classification_report(Y_test, pred_bnb))


# In[39]:


#In summary: I create a table with the scores of all the classifiers I've used in the analysis.
n3_score = n3.score(X_test, Y_test)
n5_score = n5.score(X_test, Y_test)
n8_score = n8.score(X_test, Y_test)
rad_score = rad.score(X_test, Y_test)
rf_score = rf.score(X_test, Y_test)
dt_score = dt.score(X_test, Y_test)
et_score = et.score(X_test, Y_test)
gnb_score = gnb.score(X_test, Y_test)
bnb_score = bnb.score(X_test, Y_test)

columns = ('KNN-3', 'KNN-5', 'KNN-8', 'RadiusNN', 'Random Forests', 'Decision Trees', 'Extra Trees', 'Gaussian Naive Bayes', 'Bernoulli Naive Bayes')
rows = ["Score"]
cell_text = [[n3_score, n5_score, n8_score, rad_score, rf_score, dt_score, et_score, gnb_score, bnb_score]]

fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')  #This will leave the table alone in the window
the_table = ax.table(cellText=cell_text,rowLabels=rows,
                     colLabels=columns,loc='center')
the_table.set_fontsize(30)
the_table.scale(8, 20)

#in blue the best results
the_table[(1, 4)].set_facecolor("#56b5fd")
the_table[(1, 6)].set_facecolor("#56b5fd")
the_table[(1, 7)].set_facecolor("#56b5fd")

plt.show()


# In[ ]:




