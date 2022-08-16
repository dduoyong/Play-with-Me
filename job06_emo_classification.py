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
# scaler = normalize(scaled_data)
scaled_data.info()
print(scaled_data.head())

# print(scaled_data.isna().sum())

#----heatmap----
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(scaled_data.corr(),cmap=cmap)
plt.show()

#----Dendrogram----
# plt.figure(figsize=(5, 4))
# plt.title("Dendrograms")
# shc.dendrogram(shc.linkage(scaled_data, method = 'ward'))
# plt.axhline(y=4, color='r', linestyle='--')
# plt.show()

#----K-Means clustering----
cluster = KMeans(n_clusters = 6)
cluster_label = cluster.fit_predict(scaled_data)
print(cluster_label)

#----clustering 값 dataframe에 추가----
scaled_data['label'] = cluster_label
print(scaled_data.head())

# shuffle dataset
fig = sns.barplot(x=scaled_data['label'].value_counts().index,
                  y=scaled_data['label'].value_counts()
                 )

plt.title('# of Songs in each Group')
plt.ylabel('')
fig = fig.get_figure()
fig.set_size_inches(15, 9)
fig.show()
plt.show()

print(scaled_data['label'].value_counts())


scaler = StandardScaler()
sns.set(font_scale=1.6,font='Verdana')
fig = sns.heatmap(scaler.fit_transform(scaled_data.groupby('label').mean()).T,
                  cmap='coolwarm',
                  yticklabels=[x.capitalize() for x in list(scaled_data.columns)],
                  annot=True)
fig = fig.get_figure()
fig.set_size_inches(16, 8)
fig.show()
plt.show()

#----classification emotions----
emotions = ['Euphoric','Angry','Happy',"Depressed", 'Sad', 'Cheerful']

for i in range(6):
    label_cnt = scaled_data['label'] == i
    if label_cnt.count() == 306:
        i = emotions[0] #'Euphoric'

    elif label_cnt.count() == 284:
        i = emotions[1] #'Angry'

    elif label_cnt.count() == 145:
        i = emotions[2] #'Happy'

    elif label_cnt.count() == 131:
        i = emotions[3] #'Depressed'

    elif label_cnt.count() == 86:
        i = emotions[4] #'Cheerful'

    else:
        i = emotions[5]  #'Sad'


print(emotions)

scaler = StandardScaler()
Xtrain, Xtest, ytrain, ytest = train_test_split(scaler.fit_transform(scaled_data.values),cluster_label,test_size =.25,random_state=1)

clf = RandomForestClassifier(n_estimators=30, random_state=10,criterion='entropy')
clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)


classification_matrix = np.zeros((6,6))
for x,y in zip(ytest,ypred):
    classification_matrix[x,y]+=1

ax = sns.heatmap(classification_matrix,
                 cmap='Blues',
                 cbar=False,
                 annot=True,
                 xticklabels = emotions,
                 yticklabels= emotions)

ax.set(xlabel='Preds', ylabel='True')
plt.show()

print(scaled_data.head())
print(scaled_data.info())


# emo_6 = [None] * 6
# for i in range(6):
#     if 0.06 <= scaled_data[scaled_data['label'] == i]['valence'].mean() < 0.45:
#         emo_6[i] = 'sad'
#     elif :
#         0.45 <= scaled_data[scaled_data['label'] == i]['valence'].mean() < 0.5:
#         emo_6[i] = 'sad'
#
#


