import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import collections
# from matplotlib import font_manager, rc #한글 출력을 위해
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


#
#
# font_path = './malgun.ttf'
# font_name = font_manager.FontProperties(
#     fname= font_path).get_name()
# plt.rc('font', family = 'NanumBarunGothic')

def kor_words(sent):
    return


music_genre_lst = ['Adultpop', 'Ballad', 'Dance', 'FandB', 'Idol', 'Indie', 'Pop', 'RandB_S', 'RandH', 'RandM']

for i in range(6, 7):

    gn = music_genre_lst[i]

    df_eng_lyrics = pd.read_csv('./Melon/04_clear_lyric_data/{}_eng_lyric.csv'.format(gn))
    df_eng_lyrics = df_eng_lyrics[df_eng_lyrics['English_clean_lyric'].notna()]

    words = df_eng_lyrics[df_eng_lyrics['title'] == 'Blame Game']['English_clean_lyric']
    print(words.iloc[0])

    #해당 영화의 리뷰를 문자열만 나오게 하여 형태소 분리
    words = words.iloc[0].split()
    print(words)

    #단어들의 출현 빈도를 count해서 dictionary형태로 출력해줌
    #worddict = {형태소:빈도 수}
    worddict = collections.Counter(words)
    worddict = dict(worddict)
    print(worddict)

    wordcloud_img = WordCloud(
        background_color='white', max_words=2000).generate_from_frequencies(worddict)

    # wordcloud_img = WordCloud(
    #     background_color='white', max_words=2000,
    #     font_path=font_path).generate_from_frequencies(worddict)

    plt.figure(figsize=(12, 12))
    plt.imshow(wordcloud_img, interpolation='bilinear')
    plt.axis('off')
    plt.show()