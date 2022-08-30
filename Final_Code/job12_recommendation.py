import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle
from gensim.models import Word2Vec
import pprint


# ---- 추천 시스템 함수 ----
def getRecommendation(cosin_sim):
    simScore = list(enumerate(cosin_sim[-1]))
    simScore = sorted(simScore, key = lambda x:x[1], reverse=True)
    simScore = simScore[:30]
    artistIdx = [i[0] for i in simScore]
    recSongList = df_lyrics.iloc[artistIdx, [0, 1]]
    return recSongList


# music_genre_lst = ['Adultpop', 'Ballad', 'Dance', 'FandB', 'Idol', 'Indie', 'Pop', 'RandB_S', 'RandH', 'RandM']
# -- df_lyrics, matrix, pickle, model 명 같은 장르로 바꾸기 --
df_lyrics = pd.read_csv('../Melon_Data/07_clean_gn_concat/RandM_fin.csv')
Tfidf_matrix = mmread('../Melon_Data/Models/Tfidf_RandM_lyric.mtx').tocsr()

with open('../Melon_Data/Models/RandM_tfidf.pickle', 'rb') as f:
    Tfidf = pickle.load(f)

embedding_model = Word2Vec.load('../Melon_Data/Models/RandM_w2v.model')

# ---- keyeord 이용 음악 추천 ----
keyword = '여행'
sim_word = embedding_model.wv.most_similar(keyword, topn=10)

words = [keyword]
for word, _ in sim_word:
    words.append(word)
sentence = []
count = 30
for word in words:
    sentence = sentence + [word] * count
    count -= 1

sentence_vec = Tfidf.transform(sentence)
cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)
recommendation = getRecommendation(cosine_sim)
recommendation['emo'] = df_lyrics['emo']
print(recommendation)
pprint.pprint(recommendation['title'])
# print(recommendation.info())