import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
from random import randint

# Initializing Lists
gods = []
levels = []
kdas = []
gold = []
gpms = []
damage_dealt = []
damage_taken = []
damage_mitigated = []
damage_inhand = []
team_healing = []
self_healing = []
structure_damage = []
wards = []
distance_traveled = []
god_name = ' '
gpm_check = ' '

# Monitor Scraping Efficiency
start_time = time.time()
req = 0

# read in all match #'s for Incon matches
match_number = pd.read_csv('smite_incon_matches.csv', dtype=str)
match_number.columns = ['Match']
matchcol = match_number['Match']
list1 = pd.Series.tolist(matchcol)

for match in list1:
    r = requests.get('https://smite.guru/match/' + match)
    time.sleep(randint(1, 3))

    req += 1
    now = time.time()
    time_lapse = now - start_time
    print('Request #: {}; Frequency: {} requests per second'.format(req, req / time_lapse))

    soup = BeautifulSoup(r.text, 'html.parser')
    # Grabbing Information from the Match Stats Table
    if soup.find('section', attrs={'id': 'match-stats'}) is not None:
        matches = soup.find('section', attrs={'id': 'match-stats'})
        if matches.findAll('div', attrs={'class': 'row match-table__row'}) is not None:
            players = matches.findAll('div', attrs={'class': 'row match-table__row'})
            for i in players:
                if i.find('a').text == 'Incon':
                    # Grabbing the name of the god
                    god = i.div.div.div.text
                    god_name = god
                    gods.append(god)
                    # Grab the level, K/D/A, Gold Per Minute, Damage Dealt, Damage Taken, Damage Mitigated
                    first_table_info = i.findAll('div', attrs={'class': 'row__item'})
                    level = first_table_info[0].text
                    levels.append(level)

                    kda = first_table_info[1].text
                    kdas.append(kda)

                    gold1 = first_table_info[2].text
                    gold.append(gold1)

                    gpm = first_table_info[3].text
                    gpm_check = gpm
                    gpms.append(gpm)

                    dd = first_table_info[4].text
                    damage_dealt.append(dd)

                    dt = first_table_info[5].text
                    damage_taken.append(dt)

                    dm = first_table_info[6].text
                    damage_mitigated.append(dm)
        else:
            level = 'NA'
            levels.append(level)

            kda = 'NA'
            kdas.append(kda)

            gold1 = 'NA'
            gold.append(gold1)

            gpm = 'NA'
            gpms.append(gpm)

            dd = 'NA'
            damage_dealt.append(dd)

            dt = 'NA'
            damage_taken.append(dt)

            dm = 'NA'
            damage_mitigated.append(dm)
    else:
        level = 'NA'
        levels.append(level)

        kda = 'NA'
        kdas.append(kda)

        gold1 = 'NA'
        gold.append(gold1)

        gpm = 'NA'
        gpms.append(gpm)

        dd = 'NA'
        damage_dealt.append(dd)

        dt = 'NA'
        damage_taken.append(dt)

        dm = 'NA'
        damage_mitigated.append(dm)

    if soup.findAll('div', attrs={'class': 'match-table'}) is not None:
        d_insights = soup.findAll('div', attrs={'class': 'match-table'})[3]
        if d_insights.findAll('div', attrs={'class': 'row match-table__row'}) is not None:
            players_damage = d_insights.findAll('div', attrs={'class': 'row match-table__row'})
            for i in players_damage:
                if i.find('a').text == 'Incon':
                    second_table_info = i.findAll('div', attrs={'class': 'row__item'})

                    # In Hand Damage
                    ihd = second_table_info[2].text
                    damage_inhand.append(ihd)

                    # Team Healing
                    th = second_table_info[3].text
                    team_healing.append(th)

                    # Self Healing
                    sh = second_table_info[4].text
                    self_healing.append(sh)

                    # Structure Damage
                    st = second_table_info[7].text
                    structure_damage.append(st)
        else:
            # In Hand Damage
            ihd = 'NA'
            damage_inhand.append(ihd)

            # Team Healing
            th = 'NA'
            team_healing.append(th)

            # Self Healing
            sh = 'NA'
            self_healing.append(sh)

            # Structure Damage
            st = 'NA'
            structure_damage.append(st)
    else:
        # In Hand Damage
        ihd = 'NA'
        damage_inhand.append(ihd)

        # Team Healing
        th = 'NA'
        team_healing.append(th)

        # Self Healing
        sh = 'NA'
        self_healing.append(sh)

        # Structure Damage
        st = 'NA'
        structure_damage.append(st)

    if soup.findAll('div', attrs={'class': 'match-table'}) is not None:
        farm_insights = soup.findAll('div', attrs={'class': 'match-table'})[4]
        if farm_insights.findAll('div', attrs={'class': 'row match-table__row'}) is not None:
            player_farm = farm_insights.findAll('div', attrs={'class': 'row match-table__row'})
            for i in player_farm:
                if god_name == i.find('img')['alt'] \
                        and gpm_check == i.findAll('div', attrs={'class': 'row__item'})[3].text:
                    if len(player_farm) >= 6:
                        index = player_farm.index(i)
                        third_table_info = player_farm[index].findAll('div', attrs={'class': 'row__item'})
                        # Wards Placed
                        ward = third_table_info[8].text
                        wards.append(ward)

                        # Distance Traveled
                        dist = third_table_info[7].text
                        distance_traveled.append(dist)
                    else:
                        ward = 'NA'
                        wards.append(ward)

                        dist = 'NA'
                        distance_traveled.append(dist)
        else:
            ward = 'NA'
            wards.append(ward)

            dist = 'NA'
            distance_traveled.append(dist)
    else:
        ward = 'NA'
        wards.append(ward)

        dist = 'NA'
        distance_traveled.append(dist)
damage_data = pd.DataFrame({'God': gods, 'Level': levels, 'KDA': kdas, 'Gold Per Minute': gpms,
                            'Damage Dealt': damage_dealt, 'In Hand Damage Dealt': damage_inhand,
                            'Damage Taken': damage_taken, 'Damage Mitigated': damage_mitigated,
                            'Team Healing': team_healing, 'Self Healing': self_healing,
                            'Structure Damage': structure_damage, 'Wards': wards, 'Distance Traveled': distance_traveled
                            })
damage_data.info()
damage_data.head(5)

#Send to CSV
damage_data.to_csv('C:/Users/donav/Documents/Projects/Smite/Damage.csv')
