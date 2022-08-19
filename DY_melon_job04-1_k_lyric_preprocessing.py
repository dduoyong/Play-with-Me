from konlpy.tag import Okt
import pandas as pd
import re

music_genre_lst = ['Adultpop', 'Ballad', 'Dance', 'FandB', 'Idol', 'Indie', 'Pop', 'RandB_S', 'RandH', 'RandM']

for i in range(0, 1):

    gn = music_genre_lst[i]
    df = pd.read_csv('./Melon/05_6emo_data/{}_6emo.csv'.format(gn))
    # df.info()


    # ---- 한국어 가사 stopwords ----
    df_stopwords = pd.read_csv('./melon_lyrics_kor_stopwords.csv')
    k_stopwords = list(df_stopwords['stopwords'])
    k_stopwords = k_stopwords + ['있다', '따위', '니야', '그나마', '문턱', '수행', '임무',
                                 '에에에에', '워워워', '라라라라', '라라라', '마마마라', '너너너너너', '라라라라라라', '아아아아아냐',
                                 '워어어오', '하나나나나','빙글빙글빙글빙글빙글빙글빙']
    # ---- 한국어 가사 토큰화 및 불용어 제거 ----
    okt = Okt()
    Korean_lyric = []

    for lyrics in df.lyric[:3]:
        # -- lyric에서 [가-힣 ] 제외하고 모두 공백으로 대체 --
        lyrics = re.sub('[^가-힣 ]', ' ', lyrics)

        # -- 토큰화 및 품사 tagging --
        token = okt.pos(lyrics, stem=True)
        df_token = pd.DataFrame(token, columns=['word', 'class'])
        # -- 명사/동사/형용사/부사 의 품사만 살림 --
        df_token = df_token[(df_token['class']=='Noun') | (df_token['class']=='Verb') | (df_token['class']=='Adjective') | (df_token['class']=='Adverb')]
        # df_token.info()
        # print(df_token)

        words = []
        for word in df_token.word:
            # -- 한자리 글자 삭제 --
            if len(word) > 1:
                if word not in k_stopwords:
                    words.append(word)
        cleaned_sentence = ' '.join(words)
        Korean_lyric.append(cleaned_sentence)


    df['Korean_clean_lyric'] = Korean_lyric
    df = df[['artist', 'title', 'emo', 'track_id', 'Korean_clean_lyric']]
    df.dropna(inplace=True)
    df.to_csv('./06_clear_lyric_data/{}_clean_kor_lyric.csv'.format(gn), index=False)
    df.info()