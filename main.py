import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix


csv_file_path = 'E:\Amir kabir university\Term5\hooshh\creditcard.csv'

df = pd.read_csv(csv_file_path)

column_means = df.mean()
column_std = df.std()


null_counts = df.isnull().sum()

#print(null_counts)

#df = df.fillna(column_means)
scaler = MinMaxScaler()

df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

#lower_bound = column_means - 2 * column_std
#upper_bound = column_means + 2 * column_std

#df_no_outliers = df[(df >= lower_bound) & (df <= upper_bound).all(axis=1)]

#remove duplicate records
df_no_duplicates = df.drop_duplicates()

sample_size = 100000
num_clusters = 2

df_sample = df.sample(n=sample_size, random_state=42)

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(df_sample)

labels = kmeans.predict(df)

df['Cluster'] = labels
print(df)
print("-------------------------------------")
cluster_stats = df.groupby('Cluster').describe()
print(cluster_stats)
print("-------------------------------------")

cluster_counts = df['Cluster'].value_counts()
print(cluster_counts)
print("-------------------------------------")
conf_matrix = confusion_matrix(df['Class'], df['Cluster'])

# نمایش ماتریس همبستگی
print("Confusion Matrix:")
print(conf_matrix)
print("-------------------------------------")

TN, FP, FN, TP = conf_matrix.ravel()

# محاسبه دقت
accuracy = (TP + TN) / (TP + TN + FP + FN)

# نمایش دقت
print("Accuracy:", accuracy)