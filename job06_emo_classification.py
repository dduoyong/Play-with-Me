import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as shc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

data = pd.read_csv('./Melon/04_audio_features_data/Dance_lyric_and_audio.csv')
print(data.head())

scaler = MinMaxScaler()
data.loudness = scaler.fit_transform(data.loudness.values.reshape(-1,1))
data.tempo = scaler.fit_transform(data.tempo.values.reshape(-1,1))
print(data.head())


scaled_data = pd.DataFrame(data, columns= data.columns[4:10])
scaled_data.drop(['mode'], axis=1, inplace=True)
scaled_data.info()
print(scaled_data.head())


#----heatmap----
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(scaled_data.corr(),cmap=cmap)
plt.show()

#----Dendrogram----
plt.figure(figsize=(5, 4))
plt.title("Dendrograms")
shc.dendrogram(shc.linkage(scaled_data, method = 'ward'))
plt.axhline(y=4, color='r', linestyle='--')
plt.show()

#----K-Means clustering----
cluster = KMeans(n_clusters = 6)
cluster_label = cluster.fit_predict(scaled_data)
print(cluster_label)

#----clustering 값 dataframe에 추가----
scaled_data['label'] = cluster_label
print(scaled_data.head())

fig = sns.barplot(x=scaled_data['label'].value_counts().index,
                  y=scaled_data['label'].value_counts()
                 )

plt.title('# of Songs in each Group')
plt.ylabel('')
fig = fig.get_figure()
fig.set_size_inches(15, 9)
fig.show()
plt.show()


label_origin = ['Euphoric', 'Cheerful','Happy',  'Angry', 'Depressed', 'Sad']
print(scaled_data['label'].value_counts())
label_name_df = scaled_data.groupby('label')[['valence']].mean()
label_name_df.reset_index(inplace=True, drop=False)
print(label_name_df)
label_name_df.sort_values('valence', inplace=True, ascending=False)
print(label_name_df)
label_trans = [None] * 6
for i in range(len(label_name_df)):
    label_trans[label_name_df.iloc[i, 0]] = label_origin[i]
print('lt', label_trans)
scaled_data['emo'] = 0
for i in range(len(scaled_data)):
    scaled_data.loc[i, 'emo'] = label_trans[scaled_data.loc[i, 'label']]
print(scaled_data.head())