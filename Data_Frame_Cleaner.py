import pandas as pd
import string as str
import numpy as np
############################################# Work on Summary DataFrame ################################################
summary_df = pd.read_csv('Data/Summary.csv')


#Remove uneccessary column
summary_df.drop('Unnamed: 0', axis = 1, inplace = True)

#Clean Game Mode Column so we recieve the game type and ranked or unranked
columns= summary_df['Game Mode'].str.split(' ', expand = True)
summary_df['Ranked'] = columns[0]
summary_df['Game Mode'] = columns[1]

#Remove m from the time
summary_df['Time Length'] = summary_df['Time Length'].str.replace('m', '')

#Remove SR from skill rank
summary_df['Skill Rank'] = summary_df['Skill Rank'].str.replace('SR', '')
#Rename Relics
relic_columns = ['Relic 1', 'Relic 2']
relic_names = {'Aegis Amulet': 'Aegis', 'Aegis Amulet Upgrade': 'Aegis', 'Belt of Frenzy': 'Frenzy',
               'Belt of Frenzy Upgrade': 'Frenzy', 'Bracer of Undoing': 'Bracer', 'Bracer of Undoing Upgrade': 'Bracer'
               , 'Cursed Ankh': 'Ankh', 'Cursed Ankh Upgrade': 'Ankh', 'Heavenly Wings': 'Heavenly',
               'Heavenly Wings Upgrade': 'Heavenly', 'Horrific Emblem': 'Horrific',
               'Horrific Emblem Upgrade': 'Horrific', 'Magic Shell': 'Magic', 'Magic Shell Upgrade': 'Shell',
                'Meditation Cloak': 'Meditation', 'Meditation Cloak Upgrade' : 'Meditation',
               'Phantom Veil': 'Phantom', 'Phantom Veil Upgrade': 'Phantom', 'Purification Beads': 'Beads',
               'Purification Beads Upgrade': 'Beads', 'Relic': 'Unchosen', 'Shield of Thorns': 'Thorns',
               'Shield of Thorns Upgrade': 'Thorns', 'Sundering Spear': 'Sunder', 'Sundering Spear Upgrade': 'Sunder',
               'Teleport Glyph': 'Teleport', 'Teleport Glyph Upgrade': 'Teleport', 'Blink Rune': 'Blink',
               'Blink Rune Upgrade': 'Blink'}
for i in relic_columns:
    summary_df[i] = summary_df[i].map(relic_names)

    #Remove KDa from KDA
summary_df['KDA'] = summary_df['KDA'].str.replace('KDA', '')
#Remove colon from Ranked Column
summary_df.Ranked = summary_df.Ranked.str.replace(':', '')
#Remove Plus Sign from Skill Rank
summary_df['Skill Rank'] = summary_df['Skill Rank'].str.replace('+', "")

#Edit Level and Total Gold Column
level_and_gold = summary_df['Level and Total Gold'].str.split('/', expand = True)
summary_df['Level'] = level_and_gold[0].str.strip()
summary_df['Level'] = summary_df['Level'].str.replace('Level ', '')
summary_df['Total_Gold'] = level_and_gold[1].str.replace('G', '')
summary_df['Total_Gold'] = summary_df['Total_Gold'].str.replace(',', '')
summary_df['Total_Gold'] = summary_df['Total_Gold'].str.replace(' ', '')

summary_df.drop('Level and Total Gold', axis='columns', inplace=True)

#Convert Victory to numeric
summary_df['Game Result'] = summary_df['Game Result'].map({'victory': 1, 'defeat': 0})
#Convert Ranked to Numeric
summary_df['Ranked'] = summary_df['Ranked'].map({'Ranked': 1, 'Normal': 0})

############################################## Damage Data Frame########################################################
damage_df = pd.read_csv('Damage.csv')

#Split KDA into 3 different columns
kda = damage_df['KDA'].str.split('/', expand = True)
new_names = ['Kills', 'Deaths', 'Assists']
n = 0
for i in list(kda):
    damage_df[new_names[n]] = kda[n]
    n += 1

#Drop Unnecessary columns
damage_df.drop('Unnamed: 0', axis=1, inplace=True)
damage_df.drop('Level', axis=1, inplace=True)
damage_df.drop('KDA', axis=1, inplace=True)

#Remove commas from Column data
com_names = ['Gold Per Minute', 'Damage Dealt', 'In Hand Damage Dealt', 'Damage Taken', 'Damage Mitigated',
             'Team Healing', 'Self Healing', 'Structure Damage', 'Distance Traveled']
for i in com_names:
   damage_df[i] = damage_df[i].str.replace(',', '')

#Check and Replace missing values
damage_df.isnull().sum().sort_values(ascending=False)
missing_col = ['Distance Traveled', 'Wards']
for name in missing_col:
    damage_df[name] = pd.to_numeric(damage_df[name], errors='coerce')
    damage_df[name] = damage_df[name].fillna(damage_df[name].mean())
    damage_df[name] = round(damage_df[name], 0)
    damage_df[name] = damage_df[name].astype(int)

#Convert the Rest of the Dataframe
int_columns = ['Gold Per Minute', 'Damage Dealt', 'In Hand Damage Dealt', 'Damage Taken', 'Team Healing', 'Self Healing'
               , 'Damage Mitigated', 'Structure Damage', 'Kills', 'Deaths', 'Assists']
for i in int_columns:
    damage_df[i] = damage_df[i].astype(int)

#Create Column identifying the class of the god
def create_class(df, god_col):
    Assassin = ('Arachne', 'Awilix', 'Bakasura', 'Bastet', 'Camazotz', 'Da Ji', 'Fenrir', 'Hun Batz', 'Kali', 'Loki',
                'Mercury', 'Ne Zha', 'Nemesis', 'Pele', 'Ratatoskr', 'Ravana', 'Serqet', 'Set', 'Susano', 'Thanatos',
                'Thor')
    Mage = ('Agni', 'Ah Puch', 'Anubis', 'Ao Kuang', 'Aphrodite', 'Baron Samedi', 'Chang\'e', 'Chronos', 'Discordia', 'Freya',
           'Hades', 'He Bo', 'Hel', 'Hera', 'Isis', 'Janus', 'Kukulkan', 'Merlin', 'Nox', 'Nu Wa', 'Olorun', 'Poseidon',
           'Ra', 'Raijin', 'Scylla', 'Sol', 'The Morrigan', 'Thoth', 'Vulcan', 'Zeus', 'Zhong Kui')
    Hunter = ('Ah Muzen Cab', 'Anhur', 'Apollo', 'Artemis', 'Cernunnos', 'Chernobog', 'Chiron', 'Cupid', 'Hachiman',
              'Hou Yi', 'Izanami', 'Jing Wei', 'Medusa', 'Neith', 'Rama', 'Skadi', 'Ullr', 'Xbalanque')
    Warrior = ('Achilles', 'Amaterasu', 'Bellona', 'Chaac', 'Cu Chulainn', 'Erlang Shen', 'Guan Yu', 'Hercules', 'Horus',
               'King Arthur', 'Nike', 'Odin', 'Osiris', 'Sun Wukong', 'Tyr', 'Vamana')
    Guardian = ('Ares', 'Artio', 'Athena', 'Bacchus', 'Cabrakan', 'Cerberus', 'Fafnir', 'Ganesha', 'Geb', 'Jormungandr',
                'Khepri', 'Kumbhakarna', 'Kuzenbo', 'Sobek', 'Sylvanus', 'Terra', 'Xing Tian', 'Ymir')

    #Get class types for gods
    classes = [Assassin, Mage, Hunter, Warrior, Guardian]
    class_str = ['Assassin', 'Mage', 'Hunter', 'Warrior', 'Guardian']
    class1 = {}
    for i in classes:
        ind2 = classes.index(i)
        class_str_pos = class_str[ind2]
        class1.update(dict.fromkeys(i, class_str_pos))
    new_series = df[god_col].map(class1)
    return new_series


#damage_df['Class'] = damage_df['God'].map(class1)
damage_df['Class'] = create_class(damage_df, 'God')
summary_df['Class'] = create_class(summary_df, 'God Name')

###################################Concatenate the DataFrames into one Master###########################################
incon_df = summary_df.merge(damage_df, how='inner', left_on= ['God Name', 'Kill #', 'Death #', 'Assist #'], right_on=
                            ['God', 'Kills', 'Deaths', 'Assists'])
incon_df.drop(['God Name', 'Kill #', 'Death #', 'Assist #', 'Class_y'], axis = 1, inplace=True)
incon_df.rename(columns={'Class_x': 'Class'}, inplace=True)

#Convert numeric columns to per minute
numeric_columns = ['Kills', 'Deaths', 'Assists', 'Damage Dealt', 'In Hand Damage Dealt', 'Damage Taken',
                   'Damage Mitigated', 'Structure Damage', 'Self Healing', 'Distance Traveled', 'Wards']
incon_df['Time Length'] = incon_df['Time Length'].astype(int)
for col in numeric_columns:
    incon_df[col] = incon_df[col] / incon_df['Time Length']

#Export Out
incon_df.to_csv('InconStats.csv')
