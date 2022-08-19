import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer


df = pd.read_csv('../Melon/04_audio_features_data/Adultpop_lyric_and_audio.csv')
# df.info()

# ---- audio feature 변수 ----
data = df[['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo']]
# print(data.head(10))

# ---- K-means Clustering ----
# -- 표준화 --
scaler = StandardScaler()
data_scale = scaler.fit_transform(data)

# -- k개 군집 클러스터링 --
k = 6
kmeans = KMeans(n_clusters = k, random_state = 0)
clusters = kmeans.fit(data_scale)
# k=4 >> 기쁨 슬픔 우울 편안함
# k=6 >> ['Energetic', 'Happy', 'Comfortable', 'Chilling', 'Depressed', 'Sad']

# -- 클러스터 값 원본 df에 추가 --
df['cluster'] = clusters.labels_
print(df.head(10))
print(df.groupby('cluster').count())
print(df.groupby('cluster').mean())

# -- 차원 축소(PCA) --
X = data_scale.copy()
pca = PCA(n_components=2)
pca.fit(X)
x_pca = pca.transform(X)
# print(x_pca)

pca_df = pd.DataFrame(x_pca)
pca_df['cluster'] = df['cluster']
# print(pca_df.head())

# ---- 군집화 시각화 ----
axs = plt.subplots()
axs = sns.scatterplot(0, 1, hue='cluster', data=pca_df)
plt.show()

# ---- Elbow Method ----
# 객관적인 k값 설정 및 확인을 위해 필요
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,30))
visualizer.fit(data_scale)
plt.show()