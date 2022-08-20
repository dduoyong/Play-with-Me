import pandas as pd

music_genre_list = ['Adultpop', 'Ballad', 'Dance', 'FandB', 'Idol', 'Indie', 'Pop', 'RandB_S', 'RandH', 'RandM']

for i in range(0, 10):

    gn = music_genre_list[i]

    df1 = pd.read_csv('./Melon/06_lyric_preprocessing_data/ENG/{}_clean_eng_lyric.csv'.format(gn))
    # df1.info()

    df2 = pd.read_csv('./Melon/06_lyric_preprocessing_data/KOR/{}_clean_kr_lyric.csv'.format(gn))
    # df2.info()

    # ---- 한국어 영어 전처리 된 가사 장르별로 합치기 ----
    Clean_all = pd.concat([df1, df2])
    Clean_all = Clean_all[Clean_all['Clean_lyric'].notna()]
    Clean_all.to_csv('./Melon/07_clean_gn_concat/{}_fin.csv'.format(gn), index=False)

    Clean_all.info()
