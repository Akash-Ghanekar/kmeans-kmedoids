# Mall Customer Segmentation Using K-Means & K-Medoids

## 1. Project Overview

This project focuses on customer segmentation using Unsupervised Machine Learning.

The Mall Customers dataset contains information about customers visiting a shopping mall, including their Customer ID, Gender, Age, Annual Income, and Spending Score.

The main objective is to identify different groups of customers based on their annual income and spending behaviour. This helps the mall owner understand customer purchasing patterns and identify valuable customer segments.

Two clustering algorithms are implemented:

1. K-Means Clustering
2. K-Medoids Clustering

## 2. Dataset Information

**Dataset:** Mall Customers Dataset

**Source:** [Kaggle - Mall Customers](https://www.kaggle.com/datasets/shwetabh123/mall-customers)

### Dataset Attributes

| Attribute | Description |
|---|---|
| CustomerID | Unique customer ID |
| Gender | Male or Female |
| Age | Customer's age |
| Annual Income (k$) | Annual income in thousands |
| Spending Score (1-100) | Customer spending behaviour score |

### Features Selected

This project uses only two features:

- Annual Income (k$)
- Spending Score (1-100)

These features help identify customer groups according to their income and spending patterns.

## 3. Technologies and Libraries Used

- Python
- Pandas
- Scikit-learn
- Scikit-learn-extra
- Matplotlib

### Installation

```python
!pip install scikit-learn-extra
```

## 4. Data Preprocessing

The following preprocessing steps are performed:

1. Load the dataset using Pandas.
2. Select Annual Income and Spending Score.
3. Apply StandardScaler for feature scaling.

### Feature Scaling

StandardScaler converts the selected features into a common scale.

It transforms the data to have approximately:

- Mean = 0
- Standard Deviation = 1

Scaling is important because clustering algorithms use distance calculations.

**Note:** Gender and Age are not used in this project, so Label Encoding is not required.

### Code

```python
x = data[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
x = scaler.fit_transform(x)
```

## 5. Data Preparation (Train-Test Split)

The dataset is divided into training and testing data using `train_test_split()`.

- Training Data (80%): Used to train the clustering models.
- Testing Data (20%): Used to evaluate the cluster assignments.

### Code

```python
x_train, x_test = train_test_split(
    x,
    test_size=0.2,
    random_state=42
)
```

**Important:** Clustering is an unsupervised learning technique, so there is no target column. Train-test splitting is used here to evaluate the clustering models on held-out data.

## 6. Machine Learning Algorithms

### Algorithm 1: K-Means Clustering

**Definition:**

K-Means is an unsupervised machine learning algorithm that divides data points into K clusters based on their distance from the cluster centres (centroids).

**Easy-to-Remember Workflow:**

1. Select the number of clusters (K).
2. Initialize K centroids.
3. Calculate the distance between data points and centroids.
4. Assign each point to its nearest centroid.
5. Recalculate the centroids using the mean.
6. Repeat until the centroids become stable.

### Model Creation

```python
KMeans(
    n_init=10,
    n_clusters=5,
    random_state=42
)
```

**Parameters:**

- `n_clusters=5`: Creates five clusters.
- `n_init=10`: Runs the algorithm with ten different centroid initializations.
- `random_state=42`: Ensures reproducible results.

### Algorithm 2: K-Medoids Clustering

**Definition:**

K-Medoids is an unsupervised clustering algorithm similar to K-Means. However, it uses an actual data point as the centre of each cluster instead of calculating the mean.

**Easy-to-Remember Workflow:**

1. Select the number of clusters (K).
2. Choose K initial medoids.
3. Calculate distances between data points and medoids.
4. Assign each point to its nearest medoid.
5. Update medoids to find better representative points.
6. Repeat until the medoids become stable.

### Model Creation

```python
KMedoids(
    n_clusters=5,
    random_state=42
)
```

**Parameters:**

- `n_clusters=5`: Creates five clusters.
- `random_state=42`: Ensures reproducible results.

## 7. Model Training and Evaluation

Both algorithms are stored inside a dictionary and trained using a loop.

The models are trained using `x_train` and tested using `x_test`.

### Code

```python
models = {
    "KMeans": KMeans(
        n_init=10,
        n_clusters=5,
        random_state=42
    ),

    "KMedoids": KMedoids(
        n_clusters=5,
        random_state=42
    )
}

for name, model in models.items():

    model.fit(x_train)
    pred = model.predict(x_test)

    print("Model:", name)
    print("Silhouette Score:", silhouette_score(x_test, pred))
```

### Evaluation Metric: Silhouette Score

The Silhouette Score measures how well data points fit into their assigned clusters.

**Score Range:** -1 to +1

- Close to +1: Well-separated clusters.
- Around 0: Overlapping clusters.
- Negative value: Possible incorrect cluster assignment.

A higher Silhouette Score generally indicates better-defined clusters.

## 8. Cross-Validation (K-Fold)

K-Fold Cross-Validation is used to evaluate the consistency of both clustering algorithms.

In this project, 5-Fold Cross-Validation is performed.

### Easy-to-Remember Workflow

1. Divide the dataset into five folds.
2. Select one fold for testing.
3. Use the remaining four folds for training.
4. Train both clustering models.
5. Predict cluster labels for the testing fold.
6. Calculate the Silhouette Score.
7. Repeat five times.
8. Calculate the average Silhouette Score.

### Code

```python
from sklearn.model_selection import KFold

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

km_score = []
kmed_score = []

for train_index, test_index in kf.split(x):

    x_train = x[train_index]
    x_test = x[test_index]

    km = KMeans(
        n_init=10,
        n_clusters=5,
        random_state=42
    )

    kmed = KMedoids(
        n_clusters=5,
        random_state=42
    )

    km.fit(x_train)
    kmed.fit(x_train)

    km_pred = km.predict(x_test)
    kmed_pred = kmed.predict(x_test)

    km_score.append(silhouette_score(x_test, km_pred))
    kmed_score.append(silhouette_score(x_test, kmed_pred))

print("Cross Validation")
print("KMeans:", sum(km_score) / 5)
print("KMedoids:", sum(kmed_score) / 5)
```

### Cross-Validation Evaluation

The average Silhouette Score of five folds is calculated for both algorithms.

This helps compare their clustering performance across different data splits.

## 9. Final Model Creation

After comparing both algorithms, a final K-Means model is created using the complete dataset.

The `fit_predict()` method trains the model and assigns a cluster label to every customer.

### Code

```python
final_model = KMeans(
    n_init=10,
    n_clusters=5,
    random_state=42
)

data["Cluster"] = final_model.fit_predict(x)
```

### Why fit_predict()?

- `fit()` trains the model.
- `predict()` assigns cluster labels to new data.
- `fit_predict()` trains the model and returns cluster labels for the training data.

The cluster labels are stored in a new column called `Cluster`.

## 10. Customer Group Analysis

The `groupby()` function is used to calculate the average Annual Income and Spending Score for each cluster.

### Code

```python
print(
    data.groupby("Cluster")[
        ["Annual Income (k$)", "Spending Score (1-100)"]
    ].mean()
)
```

### Customer Segmentation

Based on the average income and spending score, customer groups can be interpreted as:

| Customer Segment | Annual Income | Spending Score |
|---|---|---|
| Premium Customers | High | High |
| Careful Customers | High | Low |
| Potential Customers | Low | High |
| Low-Spending Customers | Low | Low |
| Average Customers | Medium | Medium |

**Business Understanding:**

- Premium Customers: Important target customers for loyalty programs and premium offers.
- Careful Customers: Can be attracted through personalized offers.
- Potential Customers: Can be encouraged to purchase more through promotional campaigns.
- Low-Spending Customers: Can be targeted with affordable products and discounts.
- Average Customers: Can be encouraged to increase their spending.

These interpretations should be confirmed using the actual cluster averages.

## 11. Data Visualization

Matplotlib is used to visualize the five customer groups.

### Code

```python
import matplotlib.pyplot as plt

plt.scatter(
    x[:, 0],
    x[:, 1],
    c=data["Cluster"]
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Mall Customers")
plt.show()
```

The scatter plot displays customers according to their scaled Annual Income and Spending Score.

Different colours represent different customer clusters.

## 12. Algorithm Comparison

| K-Means | K-Medoids |
|---|---|
| Uses centroid as cluster centre | Uses actual data point as cluster centre |
| Calculates mean | Selects medoid |
| More sensitive to outliers | Generally more robust to outliers |
| Usually faster | Can be computationally expensive |

Both algorithms are used to group customers based on similar income and spending patterns.

## 13. Project Workflow

```text
Load Dataset
     |
     v
Select Features
     |
     v
Apply StandardScaler
     |
     v
Train-Test Split (80:20)
     |
     v
Apply Clustering Algorithms
     |
     +-------------------+
     |                   |
     v                   v
   K-Means            K-Medoids
     |                   |
     v                   v
  Evaluation          Evaluation
     |                   |
     +---------+---------+
               |
               v
        5-Fold Cross-Validation
               |
               v
        Compare Algorithms
               |
               v
        Final K-Means Model
               |
               v
        Assign Customer Groups
               |
               v
        Visualize Clusters
```

## 14. Project Structure

```text
Mall-Customer-Segmentation/
│
├── Mall_Customers.csv
├── Kmeans&Kmedoids.ipynb
├── README.md
│
└── images/
    ├── dataset.png
    ├── model_evaluation.png
    ├── cross_validation.png
    ├── customer_groups.png
    └── kmeans_visualization.png
```

## 15. Screenshots

Add your actual Google Colab output screenshots inside the `images` folder.

### Dataset Preview

![Dataset Preview](images/dataset.png)

### Model Evaluation

![Model Evaluation](images/model_evaluation.png)

### Cross-Validation Results

![Cross Validation](images/cross_validation.png)

### Customer Group Analysis

![Customer Groups](images/customer_groups.png)

### K-Means Visualization

![KMeans Visualization](images/kmeans_visualization.png)

## 16. Conclusion

In this project, customer segmentation is performed using two unsupervised machine learning algorithms: K-Means and K-Medoids.

The dataset is preprocessed using StandardScaler and divided into training and testing sets. Both algorithms are trained and evaluated using the Silhouette Score.

5-Fold Cross-Validation is performed to compare the consistency of both algorithms.

Finally, a K-Means model is trained on the complete dataset to assign customers into five different groups. The results help the mall owner understand customer spending behaviour and identify valuable customer segments for better marketing strategies.

---

**Project Title:** Mall Customer Segmentation

**Algorithms:** K-Means and K-Medoids

**Learning Type:** Unsupervised Machine Learning

**Evaluation:** Silhouette Score and 5-Fold Cross-Validation