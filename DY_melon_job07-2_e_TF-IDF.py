import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.io import mmwrite, mmread
import pickle
music_genre_lst = ['Adultpop', 'Ballad', 'Dance', 'FandB', 'Idol', 'Indie', 'Pop', 'RandB_S', 'RandH', 'RandM']

for i in range(6, 7):

    gn = music_genre_lst[i]

    lyric_word = pd.read_csv('./Melon/06_clear_lyric_data/{}_clean_eng_lyric.csv'.format(gn))
    lyric_word = lyric_word[lyric_word['English_clean_lyric'].notna()]
    lyric_word.info()

    Tfidf = TfidfVectorizer(sublinear_tf=True)
    Tfidf_matrix = Tfidf.fit_transform(lyric_word['English_clean_lyric'])
    print(Tfidf_matrix.shape)

    #(3182 리뷰 있는 영화 수, 84461 단어의 갯수)
    print(Tfidf_matrix[0].shape)

    with open('./Melon/05_models/tfidf.pickle', 'wb') as f :
        pickle.dump(Tfidf, f)

    mmwrite('./Melon/05_models/{}_Tfidf_lyrics.mtx'.format(gn), Tfidf_matrix)