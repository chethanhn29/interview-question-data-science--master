## Machine learning
Machine Learning is said as a subset of artificial intelligence that is mainly concerned with the development of algorithms which allow a computer to learn from the data and past experiences on their own.

A Machine Learning system learns from historical data, builds the prediction models, and whenever it receives new data, predicts the output for it. The accuracy of predicted output depends upon the amount of data, as the huge amount of data helps to build a better model which predicts the output more accurately.

### Classification of Machine Learning
At a broad level, machine learning can be classified into three types:

1. Supervised learning
2. Unsupervised learning
3. Reinforcement learning

### 1. Supervised learning
- In supervised machine learning, the algorithm is trained on a labeled dataset, where each data point is associated with a corresponding target or label. ie labelled data
- The goal of supervised learning is to learn a mapping from input features to the target variable so that the algorithm can make accurate predictions or classifications on new, unseen data.
- In supervised learning, input data is provided to the model along with the output.
- Supervised learning is a type of machine learning method in which we provide sample labeled data to the machine learning system in order to train it, and on that basis, it predicts the output.
- The goal of supervised learning is to map input data with the output data.
- The supervised learning is based on supervision, and it is the same as when a student learns things in the supervision of the teacher. 
- The example of supervised learning is spam filtering,Churn Prediction

#### Supervised learning can be further divided into two main subcategories:

##### Regression: 
In regression tasks, the target variable is continuous or numerical. The algorithm's goal is to learn a function that can predict a numerical value. Examples include predicting house prices based on features like square footage and number of bedrooms, or predicting a person's age based on certain health indicators.

**Common regression algorithms include Linear Regression, Polynomial Regression, Support Vector Regression (SVR), and Random Forest Regression. Regression Trees(DT),Non-Linear Regression,Bayesian Linear Regression**

##### Classification: 
In classification tasks, the target variable is categorical or discrete, and the algorithm's objective is to assign data points to predefined classes or categories. Examples include classifying emails as spam or not spam, identifying whether an image contains a cat or a dog, or predicting the disease type based on medical test results.

**Common classification algorithms include Logistic Regression, Decision Trees, Random Forests, Support Vector Machines (SVM), k-Nearest Neighbors (KNN), and Neural Networks for classification.**

### 2. Unsupervised learning
- Unsupervised learning model finds the hidden patterns in data.
- In unsupervised learning, only input data is provided to the model. ie no labelled data


Unsupervised Machine Learning:

In unsupervised machine learning, the algorithm is given a dataset without explicit labels or targets. The goal of unsupervised learning is to discover patterns, structure, or relationships within the data without any prior knowledge of what the algorithm should be looking for. Unsupervised learning is often used for tasks like clustering and dimensionality reduction.

Unsupervised learning can be divided into two main subcategories:

Clustering: Clustering algorithms aim to group similar data points together into clusters or categories based on their inherent similarities. Examples include grouping customers based on purchasing behavior, segmenting news articles into topics, or clustering image data to discover similar patterns.

Common clustering algorithms include K-Means, Hierarchical Clustering, and DBSCAN.

Dimensionality Reduction: Dimensionality reduction techniques are used to reduce the number of features in a dataset while preserving the most important information. This can help in visualizing high-dimensional data or simplifying the data for further analysis.

Common dimensionality reduction techniques include Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE).

In summary, supervised machine learning deals with labeled data and includes regression (predicting numerical values) and classification (assigning categories or classes). Unsupervised machine learning deals with unlabeled data and includes clustering (grouping similar data points) and dimensionality reduction (reducing the number of features while retaining information). The choice of which type of algorithm to use depends on the nature of the data and the specific task at hand.




