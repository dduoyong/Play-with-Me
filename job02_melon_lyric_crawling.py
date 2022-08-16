from selenium import webdriver as wb
from selenium.webdriver.common.keys import Keys
from time import sleep
import time
import os
from urllib.request import urlretrieve
from tqdm.notebook import tqdm
import pandas as pd
from random import randint


data = pd.read_csv('./melon/01_melon_gn_classification_data/Pop0900.csv', index_col=False)
# data.info()

# ---- 'artist_name_basket'의 []대괄호를 ' ' 띄어쓰기로 trans ----
songName = data['song_name']
singerName = data['artist_name_basket']    
trs = str.maketrans("['']", '    ')
for i in tqdm(range(singerName.size)):
    a = data.iloc[i,7].translate(trs)
    a.replace('  ','')
    data.iloc[i,7] = a
print(songName[:11]+singerName[:11])
# exit()


# MAX_SLEEP_TIME = 5
# rand_value = randint(1, MAX_SLEEP_TIME)

# ---- 크롤링 사이트 차단 해제 ----
options = wb.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver = wb.Chrome(options=options)
driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.53 Safari/537.36'})
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

# ---- selenium chrome web driver options ----
options = wb.ChromeOptions()
# options.add_argument('--headless')  #크롤링 창 띄우기 옵션 / 코랩에서는 주석풀기
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = wb.Chrome('./chromedriver.exe', options=options)  #코랩에서는 'chromedriver'


# ---- 멜론 창 접속 ----
driver.get("https://www.melon.com/")
time.sleep(1)

# ---- 멜론 로그인 ----
# sleep(rand_value)
# div = driver.find_element("css selector", "#gnbLoginDiv > div > button > span").click()
# sleep(rand_value)
# div = driver.find_element("css selector", "#conts_section > div > div > div:nth-child(3) > button").click()
# sleep(rand_value)
#
# driver.find_element("css selector", "#id").send_keys('input your melon id')
# driver.find_element("css selector", "#pwd").send_keys('input your melon pw')
# sleep(rand_value)
# div = driver.find_element("css selector", "#btnLogin").click()
# sleep(rand_value)


# ---- ---- ---- ---- ---- ---- ---- ----
artist = []
song_title = []
lyric2 = []
cnt = 0
# ---- data 파일의 행 개수 ----
start = 25500
end = 26000


for i in range(start,end):
    # rand_value = randint(1, MAX_SLEEP_TIME)
    time.sleep(5)

    if i == start:
        # ---- 검색창에 '노래제목, 가수이름' 검색 ----
        driver.find_element("css selector", "#top_search").send_keys(songName[i]+','+singerName[i])
        time.sleep(2)
        # ---- 검색 버튼 클릭 ----
        div = driver.find_element("css selector", "#gnb > fieldset > button.btn_icon.search_m > span").click()
        time.sleep(2)
        print(songName[i]+','+singerName[i], "01")

        try:
            # ---- 노래 가사 보기 버튼 클릭 ----
            driver.find_element("css selector", '#frm_songList > div > table > tbody > tr:nth-child(1) > td:nth-child(3) > div > div > a.btn.btn_icon_detail').click()
            time.sleep(2)

            # ---- 스크롤 컨트롤 ----
            # -- 스크롤 높이 가져오기 --
            last_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                # -- 끝까지 스크롤 다운 --
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                # -- 스크롤 다운 후 스크롤 높이 다시 가져오기 --
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            print('스크롤 완료 01')
            time.sleep(2)

            try:
                # ---- '펼치기' 버튼 클릭 ----
                driver.find_element("css selector", '#lyricArea > button').click()
                time.sleep(2)
                # ---- 가사 내용 크롤링 / 줄바꿈 형태는 띄어쓰기로 바꾸기 ----
                lyric = driver.find_element("css selector", '#d_video_summary')
                lyric = lyric.text.replace("\n", " ")
                print(lyric)
                lyric2.append(lyric)
                song_title.append(songName[i])
                artist.append(singerName[i])

                cnt += 1
                print(cnt)

            except:
                # ---- '펼치기' 버튼 없으면 패스 ----
                pass

            # ---- 뒤로 가기 ----
            driver.back()
            print('debug01')

        except:
            # ---- 노래 가사 보기 버튼 없으면 패스 ----
            pass

    else:
        try:
            # ---- 검색창에 '노래제목, 가수이름' 검색 ----
            driver.find_element("css selector", "#top_search").send_keys(songName[i]+','+singerName[i])
            time.sleep(2)
            # ---- 검색 버튼 클릭 ----
            div = driver.find_element("css selector", "#header_wrap > div.wrap_search_field > fieldset > button.btn_icon.search_m > span").click()
            time.sleep(2)
            print(songName[i]+','+singerName[i], "02")

            try:
                # ---- 노래 가사 보기 버튼 클릭 ----
                driver.find_element("css selector", '#frm_songList > div > table > tbody > tr:nth-child(1) > td:nth-child(3) > div > div > a.btn.btn_icon_detail').click()
                time.sleep(2)

                # ---- 스크롤 컨트롤 ----
                last_height = driver.execute_script("return document.body.scrollHeight")
                while True:
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)
                    new_height = driver.execute_script("return document.body.scrollHeight")
                    if new_height == last_height:
                        break
                    last_height = new_height
                print('스크롤 완료 02')
                time.sleep(2)

                try:
                    # ---- '펼치기' 버튼 클릭 ----
                    driver.find_element("css selector", '#lyricArea > button').click()
                    time.sleep(2)
                    # ---- 가사 내용 크롤링 / 줄바꿈 형태는 띄어쓰기로 바꾸기 ----
                    lyric = driver.find_element("css selector", '#d_video_summary')
                    lyric = lyric.text.replace("\n", " ")
                    print(lyric)
                    lyric2.append(lyric)
                    song_title.append(songName[i])
                    artist.append(singerName[i])

                    cnt += 1
                    print(cnt)

                except:
                    # ---- '펼치기' 버튼 없으면 패스 ----
                    pass

                # ---- 뒤로 가기 ----
                driver.back()
                print('debug02')

            except:
                # ---- 노래 가사 보기 버튼 없으면 패스 ----
                pass
            print('debug03')

        except:
            # ---- i에 해당하는 노래/가수 없으면 패스 ----
            pass
            time.sleep(1)
    print('debug04')
    time.sleep(1)

    # ---- 검색창 지우기(for 새로운 검색) ----
    driver.find_element("css selector", "#top_search").clear()

# ---- DataFrame 형태로 저장 ----
df = pd.DataFrame({'artist':artist, 'title':song_title, 'lyric':lyric2})
df.to_csv('./melon/02_melon_lyric_crawling_data/Pop/Pop_lyrics_{}_{}_cnt_{}.csv'.format(start, end, cnt), index=False)