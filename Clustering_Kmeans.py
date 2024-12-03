import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

''' Step One '''
# Read the csv file to import the data in dataframe
df = pd.read_csv('insurance.csv', sep=',')
''' Step Two '''

# there are three columns been present in the dataset, which are having String values and we need to convert
# those string values to a one_hot encoded scaler values. In addtion to these, we have to pivot the dataset
# based on the categories been present

categorical_columns = ['sex', 'smoker', 'region']
df_encoded = pd.get_dummies(df, prefix='One', prefix_sep='_',
                            columns=categorical_columns, dtype='int8', drop_first=True)
#print('Original Dataset Shape: ', df.shape)
#print('Original Dataset Columns: ', df.columns)
#print('Pivot Dataset Shape: ', df_encoded.shape)
#print('Pivot Dataset Columns: ', df_encoded.columns)
print(df_encoded.head())

''' Step Three '''
print('number of features: ', df_encoded.columns)
# load the KMeans group algorithm with k = 3
kmeansML = KMeans(n_clusters=3, random_state=23)

''' Step four '''
# training the model to generate clusters
df_encoded['clusters'] = kmeansML.fit_predict(df_encoded)

''' Step five : Visualization of the clusters'''
plt.figure(figsize=(10, 6))
for cluster in range(3):
    cluster_data = df_encoded[df_encoded['clusters'] == cluster]
    plt.scatter(cluster_data['age'], cluster_data['charges'], label='Cluster {}'.format(cluster))
plt.title('K-means clustering for insurance data')
plt.xlabel('Sex')
plt.ylabel('Charges')
plt.legend()
plt.grid()
plt.show()

''' Step Six: Cluster Value Distribution '''
cluster_data_distribution = df_encoded['clusters'].value_counts().sort_index()
print(cluster_data_distribution)

''' Visualize these clusters in a bar chart '''
plt.figure(figsize=(10, 6))
cluster_data_distribution.plot(kind='bar', color='skyblue')
plt.title('Data distribution for K-means clustering')
plt.xlabel('Cluster')
plt.ylabel('Number of customers')
plt.grid()
plt.legend()
plt.show()

''' Step Seven: Range of the cluster values '''
cluster_ranges = df_encoded.groupby('clusters')['charges'].agg(['min', 'max']).reset_index()
print('cluster_ranges')
print(cluster_ranges)

plt.figure(figsize=(10, 6))
for cluster in range(3):
    plt.plot([cluster, cluster],
        [cluster_ranges.loc[cluster, 'min'], cluster_ranges.loc[cluster, 'max']], marker='o',
             label='Cluster {}'.format(cluster)
    )

plt.title('Range of cluster values')
plt.xlabel('cluster')
plt.ylabel('Range of ranges')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

