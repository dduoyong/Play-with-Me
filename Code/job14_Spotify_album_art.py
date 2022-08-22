import pandas as pd
import os
import json
import spotipy
import base64
import requests
import re


# ---- spotify access & getting access token ----
client_id = "input your client id"
client_pw = "input your client pw"
endpoint = "https://accounts.spotify.com/api/token"

rURI = "http://localhost:3000"
scope = "ugc-image-upload"

encoded = base64.b64encode("{}:{}".format(client_id, client_pw).encode('utf-8')).decode('ascii')

headers = {"Authorization": "Basic {}".format(encoded)}
payload = {"grant_type": "client_credentials"}

response = requests.post(endpoint, data=payload, headers=headers)
access_token = json.loads(response.text)['access_token']
# print(access_token)
# exit()

os.environ["SPOTIPY_CLIENT_ID"] = "input your client id"
os.environ["SPOTIPY_CLIENT_SECRET"] = "input your client pw"


# ---- 장르별 파일 불러오기 ----
# music_genre_lst = ['Adultpop', 'Ballad', 'Dance', 'FandB', 'Idol', 'Indie', 'Pop', 'RandB_S', 'RandH', 'RandM']
df = pd.read_csv('../Melon/07_clean_gn_concat/RandM_fin.csv')
track_id = df['track_id']

album_art = []
release_date = []
cnt = 0


# ---- Spotify Search API ----
for i in range(len(df)):

    # -- 아티스트 이름과 노래 제목으로 검색 --
    headers = {"Authorization": "Bearer {}".format(access_token)}
    params = {
        "q": df['artist'][i]+df['title'][i],
        "type": "track",
        "limit": "1"
     }

    r = requests.get("https://api.spotify.com/v1/search", params=params, headers=headers)
    # print(r.text)

    # -- images url 찾으면 append 아니면 비우기 --
    try:
        album_art_640 = re.findall('"url" : "(.*)\"', r.text)
        print(album_art_640[0])
        album_art.append(album_art_640[0])
        cnt+=1
        print(cnt)

    except:
        album_art.append(" ")
        print("none images url")

    # -- release date 찾으면 append 아니면 비우기 --
    try:
        release = re.findall('"release_date" : "(.*)\"', r.text)
        print(release[0])
        release_date.append(release[0])
        cnt += 1
        print(cnt)
    except:
        release_date.append(" ")
        print("none date")


df['album_art'] = album_art
df['release_date'] = release_date
df = df[['artist', 'title', 'Clean_lyric', 'emo', 'track_id', 'track_url', 'album_art', 'release_date']]
df.dropna(inplace=True)
df.to_csv('../Melon/08_album_info/RandM_track_data.csv', index = False)
df.info()
print(df.head())
