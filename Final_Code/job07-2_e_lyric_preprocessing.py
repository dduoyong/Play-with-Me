from nltk.corpus import stopwords
from nltk import word_tokenize  #단어 단위를 기준으로 토큰화
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import pandas as pd
import re

# nltk.download()

music_genre_list = ['Adultpop', 'Ballad', 'Dance', 'FandB', 'Idol', 'Indie', 'Pop', 'RandB_S', 'RandH', 'RandM']

for i in range(0, 10):

    gn = music_genre_list[i]
    df = pd.read_csv('../Melon/05_6emo_data/{}_lyric_and_emo.csv'.format(gn))

    # ---- 영어 가사 토큰화 및 불용어 제거 ----
    English_lyric = []

    for lyrics in df.lyric:
        # -- 영어 문자 제외하고 모두 공백으로 대체 --
        lyrics = re.sub('[^a-zA-Z]', ' ', lyrics)
        # -- 모든 영어 텍스트 소문자로 대체 --
        small_lyrics = lyrics.lower()
        # print('small_lyrics :', small_lyrics)

        # -- 토큰화 --
        # nltk.download('tagsets')
        # nltk.download('averaged_perceptron_tagger')
        # nltk.download('wordnet')
        token_words = word_tokenize(small_lyrics)
        # print('token_words :', token_words)
        # exit()

        # -- 품사 tagging --
        tag_words = pos_tag(token_words)
        # print(tag_words)
        # exit()
        # -- 튜플 형태를 데이터 프레임 형태로 변환 --
        df_token = pd.DataFrame(tag_words, columns=['word', 'class'])
        # -- 명사(단복수)/동사(원형,과거,현재분사,과거분사)/형용사(비교급,최고급) 의 품사만 살림 --
        df_noun = df_token[(df_token['class'] == 'NN') | (df_token['class'] == 'NNS')]
        df_verb = df_token[(df_token['class'] == 'VB') | (df_token['class'] == 'VBD') | (df_token['class'] == 'VBG') | (df_token['class'] == 'VBN')]
        df_adjective = df_token[(df_token['class'] == 'JJ') | (df_token['class'] == 'JJR') | (df_token['class'] == 'JJS')]
        # df_adverb = df_token[(df_token['class'] == 'RB') | (df_token['class'] == 'RBR') | (df_token['class'] == 'RBS')]

        # -- 표제어(원형) 추출 --
        lemmatizer = WordNetLemmatizer()
        noun = [lemmatizer.lemmatize(w, pos='n') for w in df_noun['word']]
        verb = [lemmatizer.lemmatize(w, pos='v') for w in df_verb['word']]
        adjective = [lemmatizer.lemmatize(w, pos='a') for w in df_adjective['word']]
        # adverb = [lemmatizer.lemmatize(w, pos='r') for w in df_adverb['word']]

        # print(noun)   #명사
        # print(verb)    #동사
        # print(adjective)   #형용사
        # print(adverb)   #부사
        # exit()

        clean_raw_lyric = noun + verb + adjective   #list 형태

        # -- nltk stopwords & 불용어 제거 --
        stop_words = stopwords.words('english')
        # stop_words_list = stopwords.words('english')
        # print('불용어 개수 :', len(stop_words_list))
        # print('불용어 출력 :', stop_words_list)
        # exit()
        stop_words.extend(['Chorus', 'Shoshanim', 'aaa', 'aaaaaaa', 'aaaaahhhhh', 'afterthe', 'aha', 'ahhh', 'ala',
                           'ann', 'arr','athena', 'athena', 'awooooo', 'ayayaya', 'ayy', 'ayyy', 'ayyyyy', 'baaam',
                           'baam', 'badkiz', 'bae', 'bam','baram', 'beep', 'bello', 'bitch', 'bizzionary', 'bizzyb', 'bom',
                           'boo', 'booh', 'boom', 'bos', 'bubibu', 'buh', 'bum', 'capls', 'coracao', 'corny',
                           'cos', 'coz', 'cra', 'cul', 'cuz', 'dab', 'dadadada', 'dadadadadadada', 'dangy', 'dangy', 'daralala',
                           'dda', 'diggi', 'diggy', 'ding', 'doa', 'dong', 'doo', 'dooroo', 'dotdotdotdot', 'eheheheh',
                           'ehoheh', 'emm', 'fff', 'fraulein', 'fuck', 'gal', 'girardeau', 'gon',
                           'gotto', 'grrrrrah', 'gut', 'hah', 'hahahahahahahahaha', 'hanalee', 'hee', 'hey', 'heyo', 'hmmmm',
                           'hoo', 'hook', 'hooo', 'huh', 'huhu', 'hum', 'iamma', 'imma', 'invi', 'iraq', 'jajaprrrr',
                           'jigga', 'kiki', 'ktp', 'lalala', 'lalalala', 'lalalalala', 'lalalalalalalalalalala', 'lee',
                           'loo', 'louisianna', 'lyricfind', 'mimi', 'mmhmm', 'mmm', 'mmmmm', 'mouan',
                           'nah', 'nananananana', 'oah', 'oather', 'ohehoh', 'ohh', 'ohhh', 'ooh',
                           'ooo', 'oooh', 'ooonnnn', 'oooo', 'oooohhh', 'oooohoo', 'ooooooo', 'ooowoooo', 'oowah', 'org', 'ouu',
                           'pappapara', 'pas', 'phly', 'rememberin', 'rin', 'roo', 'rut', 'ryone', 'sac', 'sayla',
                           'sex', 'shalalalala', 'shed', 'shhhhh', 'shubidubi', 'skrrt', 'sooh', 'suturutururu', 'tatta',
                           'tennessee', 'thangs', 'theeastlight', 'tic', 'toc', 'tryna', 'turururu', 'tvxq', 'uaagh', 'uck', 'ugh',
                           'uhm', 'uhuhuh', 'umm', 'ummm', 'verse', 'wan', 'war', 'whoa', 'whoo', 'woah',
                           'wohuh', 'woo', 'wooooo', 'wow', 'wrah', 'wuh', 'yay', 'yaya', 'yayayaya', 'yea', 'yeah', 'yeh',
                           'yoooooou', 'youuuuuuuuu', 'yuh', 'yum', 'zizi' 'fxxx'])

        result = []
        for token in clean_raw_lyric:
            if len(token) > 2:
                if token not in stop_words:
                    result.append(token)
        cleaned_sentence = ' '.join(result)
        English_lyric.append(cleaned_sentence)
        # print(English_lyric)


    df['Clean_lyric'] = English_lyric
    df = df[['artist', 'title', 'Clean_lyric', 'emo', 'track_id', 'track_url']]
    df.dropna(inplace=True)
    df.to_csv('../Melon/06_lyric_preprocessing_data/ENG/{}_clean_eng_lyric.csv'.format(gn), index=False)
    df.info()

