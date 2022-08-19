# Import libraries
import pandas as pd
import os
import json
import spotipy
import base64
import requests
import re

# spotify access & getting access token
client_id = "97c14de138864e1380720a665f2f6ce9"
client_secret = "78930e971f6a4f8585af34ec554b530a"
endpoint = "https://accounts.spotify.com/api/token"
rURI = "http://localhost:3000"
scope = "ugc-image-upload"

encoded = base64.b64encode("{}:{}".format(client_id, client_secret).encode('utf-8')).decode('ascii')
headers = {"Authorization": "Basic {}".format(encoded)}
payload = {"grant_type": "client_credentials"}
response = requests.post(endpoint, data=payload, headers=headers)
access_token = json.loads(response.text)['access_token']

os.environ["SPOTIPY_CLIENT_ID"] = "97c14de138864e1380720a665f2f6ce9"
os.environ["SPOTIPY_CLIENT_SECRET"] = "78930e971f6a4f8585af34ec554b530a"


# 장르별 파일 불러오기
df = pd.read_csv('./Adultpop_fin.csv')
track_id = df['track_id']

# 앨범아트 없는 노래 확인
headers = {"Authorization": "Bearer {}".format(access_token)}
params = {
    "q": df['artist'][374]+df['title'][374],
    "type": "track",
    "limit": "1"
 }

r = requests.get("https://api.spotify.com/v1/search", params=params, headers=headers)
print(r.text)

album_art_640 = re.findall('"url" : "(.*)\"', r.text)
print(album_art_640[0])
exit()

album_art = []
cnt = 0

## Spotify Search API
for i in range(len(df)):

    headers = {"Authorization": "Bearer {}".format(access_token)}
    params = {
        "q": df['artist'][i]+df['title'][i],
        "type": "track",
        "limit": "1"
     }

    r = requests.get("https://api.spotify.com/v1/search", params=params, headers=headers)
    # print(r.text)

    album_art_640 = re.findall('"url" : "(.*)\"', r.text)
    print(album_art_640[0])
    album_art.append(album_art_640[0])
    cnt+=1
    print(cnt)

df['album_art'] = album_art
df = df[['artist','title','Clean_lyric','emo','track_id','album_art','danceability','energy','loudness','acousticness','valence','tempo']]
df.dropna(inplace=True)
df.to_csv('./Adultpop_album_art.csv', index = False)
df.info()
print(df.head())

