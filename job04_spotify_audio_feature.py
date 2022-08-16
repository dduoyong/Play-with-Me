import urllib
from urllib.parse import urlparse
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
import requests
import base64
import os


# ---- spotify access & getting access token ----
client_id = "input your client id"
client_pw = "input your client pw"
endpoint = "https://accounts.spotify.com/api/token"

encoded = base64.b64encode("{}:{}".format(client_id, client_pw).encode('utf-8')).decode('ascii')

headers = {"Authorization": "Basic {}".format(encoded)}
payload = {"grant_type": "client_credentials"}

response = requests.post(endpoint, data=payload, headers=headers)
access_token = json.loads(response.text)['access_token']

os.environ["SPOTIPY_CLIENT_ID"] = "input your client id"
os.environ["SPOTIPY_CLIENT_SECRET"] = "input your client pw"


df = pd.read_csv('./Melon/03_lyric_concat_data/RandM_lyric.csv')
df.info()
# ---- 오류난 경우 해당 행 확인용 ----
# print(df['track_id_url'][37])
# exit()

track_id_list = []
track_id_cnt = 0
audio_feat_cnt = 0
audio_features = pd.DataFrame(columns=["danceability", "energy", "loudness", "mode", "acousticness", "valence", "tempo"])


# ---- track_url에서 track_id 추출 ----
for i in range(0, 1011):
    track_id_url = df['track_id_url'][i]

    # -- track_url parsing하기 --
    urllib.parse.urlparse(track_id_url, scheme='', allow_fragments=True)
    track_url_par = urlparse(track_id_url)
    print(track_url_par)

    # -- track_id를 빈 리스트에 추가 --
    track_id = str(track_url_par.path).split('/')[4]
    # adultpop_path='/track/6oHhbZ7H8Akb6YiejwDAVF' [2]
    # metal_path=' https://open.spotify.com/track/4E43XdK12qORYvIS4xfgAb' [4]
    print(track_id)
    track_id_list.append(track_id)
    track_id_cnt += 1
    print(track_id_cnt)


    # ---- Audio Features 가져오기 ----
    spotify = spotipy.Spotify(auth_manager=SpotifyClientCredentials())
    features = spotify.audio_features(tracks=track_id)[0]

    audio_features = audio_features.append({"track_id": track_id, "danceability":features['danceability'], "energy":features['energy'],
                                            "loudness":features['loudness'], "mode":features['mode'], "acousticness":features['acousticness'],
                                            "valence":features['valence'], "tempo":features['tempo']}, ignore_index=True)
    print(features)


df['track_id'] = track_id_list
df = df[['artist', 'title', 'lyric', 'track_id']]
df = df.merge(audio_features, how='left', left_on='track_id', right_on='track_id').drop_duplicates()
df.dropna(inplace=True)
df.to_csv('./Melon/04_audio_features_data/RandM_lyric_and_audio.csv', index=False)
df.info()
