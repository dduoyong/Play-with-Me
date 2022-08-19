import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.io import mmwrite, mmread
import pickle


music_genre_list = ['Adultpop', 'Ballad', 'Dance', 'FandB', 'Idol', 'Indie', 'Pop', 'RandB_S', 'RandH', 'RandM']

for i in range(0,10):

    gn = music_genre_list[i]
    lyric_word = pd.read_csv('./Melon/07_clean_gn_concat/{}_fin.csv'.format(gn))
    # lyric_word.info()

    # ---- TF-IDF 확인 ----
    Tfidf = TfidfVectorizer(sublinear_tf=True)
    Tfidf_matrix = Tfidf.fit_transform(lyric_word['Clean_lyric'])
    print(Tfidf_matrix.shape)
    print(Tfidf_matrix[0].shape)

    # ---- pickle 담기 ----
    with open('Melon/Models/{}_tfidf.pickle'.format(gn), 'wb') as f:
        pickle.dump(Tfidf, f)

    mmwrite('./Melon/Models/Tfidf_{}_lyric.mtx'.format(gn), Tfidf_matrix)