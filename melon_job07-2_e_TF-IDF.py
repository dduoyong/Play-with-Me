import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.io import mmwrite, mmread
import pickle

lyric_word = pd.read_csv('./melon/04_melon_clear_lyric/Pop_clean_eng_lyric.csv')
lyric_word = lyric_word[lyric_word['English_clean_lyric'].notna()]
lyric_word.info()


Tfidf = TfidfVectorizer(sublinear_tf=True)
Tfidf_matrix = Tfidf.fit_transform(lyric_word['English_clean_lyric'])
print(Tfidf_matrix.shape)
#(3182 리뷰 있는 영화 수, 84461 단어의 갯수)

print(Tfidf_matrix[0].shape)

with open('./melon/melon_models/tfidf.pickle', 'wb') as f :
    pickle.dump(Tfidf, f)

mmwrite('./melon/melon_models/Tfidf_pop_lyrics.mtx', Tfidf_matrix)