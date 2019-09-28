import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
from random import randint

#Initalizing Lists

game_results = []
modes = []
lengths = []
mmrs = []
ssrs = []
relics_1 = []
relics_2 = []
kills = []
deaths = []
assists = []
kdas = []
gods = []
levelsgold = []

#Monitor Scraping Efficiency
start_time = time.time()
req = 0

#Setting Number of Pages for loop
pages = [str(i) for i in range(1, 45)]

for page in pages:
    r = requests.get('https://smite.guru/profile/599339-Incon/matches?page=' + page)
    time.sleep(randint(1, 3))

    req += 1
    now = time.time()
    time_lapse = now - start_time
    print('Request #: {}; Frequency: {} requests per second'.format(req, req / time_lapse))

    soup = BeautifulSoup(r.text, 'html.parser')
    games = soup.find('div', attrs={'class': 'column col-8 col-md-12'})
    winloss = games.findAll('div', attrs={'class': ['widget match-widget match-widget--defeat',
                                                    'widget match-widget match-widget--victory']})
    for i in winloss:
        #Game Result
        game_result = i['class'][2][14:]
        game_results.append(game_result)

        #Getting game mode
        mode = i.find('div', attrs={'class': 'title'}).text
        modes.append(mode)

        #Getting length of the game
        length = i.find('div', attrs={'class': 'sub'}).text
        lengths.append(length)

        #Getting MMR/elo
        mmr = i.findAll('div', attrs={'class': 'sub'})[1].text[0:3]
        mmrs.append(mmr)

        #Getting skill rank
        ssr = i.findAll('div', attrs={'class': 'sub'})[2].text
        ssrs.append(ssr)

        #Getting Relics
        relics = i.findAll('div', attrs={'class':'item'})
        relic_1 = relics[0].img['alt']
        relic_2 = relics[1].img['alt']
        relics_1.append(relic_1)
        relics_2.append(relic_2)

        #Kills
        kill = i.find('span', attrs={'class': 'match-widget__kills'}).text
        kills.append(kill)

        #Deaths
        death = i.find('span', attrs={'class': 'match-widget__deaths'}).text
        deaths.append(death)

        #Assists
        assist = i.find('span', attrs={'class': 'match-widget__assists'}).text
        assists.append(assist)

        #KDA
        kda = i.find('span', attrs={'class' : 'match-widget--sub'}).text
        kdas.append(kda)

        #God
        god = i.find('div', attrs={'class': 'text-center match-widget--title'}).text
        gods.append(god)

        #Level/Gold
        levelgold = i.find('div', attrs={'class': 'text-center match-widget--sub match-widget__stats--small'}).text
        levelsgold.append(levelgold)

data_intro = pd.DataFrame({'Game Result': game_results, 'Game Mode': modes, 'Time Length': lengths, 'MMR or Elo': mmrs,
                           'Skill Rank': ssrs, 'Relic 1': relics_1, 'Relic 2': relics_2, 'Kill #': kills,
                           'Death #': deaths, 'Assist #': assists, 'KDA': kdas, 'God Name': gods,
                           'Level and Total Gold': levelsgold})
data_intro.info()
data_intro.head(5)

data_intro.to_csv('C:/Users/donav/Documents/Projects/Smite/Summary.csv')