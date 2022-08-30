import pandas as pd
from wordcloud import WordCloud
import collections
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ---- 폰트 설정 ----
font_path = './malgun.ttf'
font_name = font_manager.FontProperties(fname= font_path).get_name()
plt.rc('font', family = 'NanumBarunGothic')


music_genre_lst = ['Adultpop', 'Ballad', 'Dance', 'FandB', 'Idol', 'Indie', 'Pop', 'RandB_S', 'RandH', 'RandM']

for i in range(0, 10):

    gn = music_genre_lst[i]

    df_lyrics = pd.read_csv('./Melon/07_clean_gn_concat/{}_fin.csv'.format(gn))
    df_lyrics = df_lyrics[df_lyrics['Clean_lyric'].notna()]

    words = df_lyrics[df_lyrics['title'] == 'Blame Game']['Clean_lyric']
    print(words.iloc[0])

    # -- 해당 음악의 가사를 문자열만 나오게 하여 형태소 분리 --
    words = words.iloc[0].split()
    print(words)

    # -- 단어들의 출현 빈도를 count해서 dictionary형태로 출력해줌 / worddict = {형태소:빈도 수} --
    word_dict = collections.Counter(words)
    word_dict = dict(word_dict)
    print(word_dict)

    # -- wordcloud 그리기 --
    wordcloud_img = WordCloud(background_color='white', max_words=2000).generate_from_frequencies(word_dict)

    plt.figure(figsize=(12, 12))
    plt.imshow(wordcloud_img, interpolation='bilinear')
    plt.axis('off')
    plt.show()