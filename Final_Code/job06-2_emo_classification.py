import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


# music_genre_lst = ['Adultpop', 'Ballad', 'Dance', 'FandB', 'Idol', 'Indie', 'Pop', 'RandB_S', 'RandH', 'RandM']
df = pd.read_csv('../Melon_Data/04_audio_features_data/RandM_lyric_and_audio.csv')

# ---- 감정 분류에 사용할 데이터 컬림 ----
data = df[['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo']]

# ---- data MinMax scaler ----
scaler = MinMaxScaler()
minmax_data = scaler.fit_transform(data)
scaled_data = pd.DataFrame(minmax_data, columns= ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo'])

# ---- Heatmap ----
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.figure(figsize=(10,10))
plt.title('RandM Heatmamp', fontsize=10)
sns.heatmap(scaled_data.corr(), cmap=cmap)
plt.show()
# exit()

# ---- K-Means clustering ----
cluster = KMeans(n_clusters = 6)
cluster_label = cluster.fit_predict(scaled_data)
print(cluster_label)

# ---- clustering label dataframe에 추가 ----
scaled_data['cluster_label'] = cluster_label
print(scaled_data.head())

# ---- Bar plot ----
fig = sns.barplot(x=scaled_data['cluster_label'].value_counts().index, y=scaled_data['cluster_label'].value_counts())
plt.title('RandM cluster label ratio')
plt.ylabel('')
fig = fig.get_figure()
fig.set_size_inches(15, 9)
fig.show()
plt.show()


# ---- 감정 match ----
# -- cluster label 별 counts --
label_origin = ['Energetic', 'Happy', 'Comfortable', 'Chilling', 'Depressed', 'Sad']
print(scaled_data['cluster_label'].value_counts())

# -- label 별 'valance'값 평균 & rename --
label_name_df = scaled_data.groupby('cluster_label')[['valence']].mean()
label_name_df.reset_index(inplace=True, drop=False)
print(label_name_df)
label_name_df.sort_values('valence', inplace=True, ascending=False)
print(label_name_df)

# -- sort된 valance값(clusterlabel) 과 감정 매칭 --
label_trans = [None] * 6
for i in range(len(label_name_df)):
    label_trans[label_name_df.iloc[i, 0]] = label_origin[i]
print('lt', label_trans)

scaled_data['emo'] = 0
for i in range(len(scaled_data)):
    scaled_data.loc[i, 'emo'] = label_trans[scaled_data.loc[i, 'cluster_label']]
print(scaled_data.head())


df['emo'] = scaled_data['emo']
df = df[['artist', 'title', 'lyric', 'emo', 'track_id', 'track_url']]
df.to_csv('../Melon_Data/05_6emo_data/RandM_lyric_and_emo.csv', index=False)
df.info()