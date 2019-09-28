import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Data_Frame_Cleaner import create_class

# Read Dataframe in and Drop Unnecessary columns
incon_df = pd.read_csv('InconStats.csv')


#Check Distribution of Game Modes
incon_df['Game Mode'].value_counts()
incon_df = incon_df[incon_df['Game Mode'] == 'Conquest']


incon_df.loc[incon_df['Team Healing'] != 0, 'Team Healing'] = 1

#Statistical Summaries and Dataframe Shape
incon_df.info()
incon_df[['Kills', 'Deaths', 'Assists']].describe()
incon_df[['Time Length', 'MMR or Elo', 'Skill Rank']].describe()
incon_df[['Total_Gold', 'Gold Per Minute']].describe()
incon_df[['Damage Dealt', 'In Hand Damage Dealt', 'Damage Taken', 'Damage Mitigated', 'Structure Damage']].describe()
incon_df[['Team Healing', 'Self Healing', 'Team Healing', 'Wards', 'Distance Traveled']].describe()
incon_df[['Game Result', 'Ranked']].describe()
incon_df.shape

# pd.set_option('display.max_rows', 8)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 100)

#Showing Percentage of Times certain Relics are Used
relics = ['Relic 1', 'Relic 2']
val_columns = []
for relic in relics:
    col = incon_df[relic].value_counts()
    val_columns.append(col)
df_relic = pd.concat(val_columns, ignore_index= True, axis = 1, sort = False)

for i in range(len(df_relic.columns)):
    df_relic[i] = pd.to_numeric(df_relic[i], errors = 'coerce')
    df_relic[i] = df_relic[i].fillna(0)
    df_relic[i] = df_relic[i].astype(int)
df_relic['Total'] = df_relic[0] + df_relic[1]

total_games = 1000
total_perc = (df_relic['Total']/total_games) * 100
relic_use = pd.concat([df_relic['Total'], total_perc], axis = 1, keys = ['Count', '%']).sort_values(by =['Count'], ascending = False)



############################################# Plots ####################################################################
#Graph Showing Game Result, Ranked, Time Length
grid = sns.FacetGrid(incon_df, col = 'Ranked', row = 'Game Result')
grid.map(plt.hist, 'Time Length')

#Plot Showing the Gods used
incon_df['God'].value_counts().head(8).plot(kind='bar')
plt.title('Games God Played')

#Victory, Class, GPM
fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (15, 6))
assassin = incon_df[incon_df['Class'] == 'Assassin']
hunter = incon_df[incon_df['Class'] == 'Hunter']
mage = incon_df[incon_df['Class'] == 'Mage']
warrior = incon_df[incon_df['Class'] == 'Warrior']
guardian = incon_df[incon_df['Class'] == 'Guardian']
ax = sns.distplot(assassin[assassin['Game Result']== 1]['Gold Per Minute'], label = 'Victory', ax = axes[0,0], kde = False)
ax = sns.distplot(assassin[assassin['Game Result'] == 0]['Gold Per Minute'], label = 'Loss', ax = axes[0,0], kde = False)
ax.legend()
ax.set_title('Assassin')
plt.xlim(375,850)
ax = sns.distplot(hunter[hunter['Game Result'] == 1]['Gold Per Minute'], label = 'Victory', ax = axes[0,1], kde = False)
ax = sns.distplot(hunter[hunter['Game Result'] == 0]['Gold Per Minute'], label = 'Loss', ax = axes[0,1], kde = False)
ax.legend()
ax.set_title('Hunter')
plt.xlim(375,850)
ax = sns.distplot(mage[mage['Game Result'] == 1]['Gold Per Minute'], label = 'Victory', ax = axes[0,2], kde = False)
ax = sns.distplot(mage[mage['Game Result'] == 0]['Gold Per Minute'], label = 'Loss', ax = axes[0,2], kde = False)
ax.legend()
ax.set_title('Mage')
plt.xlim(375,850)
ax = sns.distplot(warrior[warrior['Game Result'] == 1]['Gold Per Minute'], label = 'Victory', ax = axes[1,0], kde = False)
ax = sns.distplot(warrior[warrior['Game Result'] == 0]['Gold Per Minute'], label = 'Loss', ax = axes[1,0], kde = False)
ax.legend()
ax.set_title('Warrior')
plt.xlim(375,850)
ax = sns.distplot(guardian[guardian['Game Result'] == 1]['Gold Per Minute'], label = 'Victory', ax = axes[1,1], kde = False)
ax = sns.distplot(guardian[guardian['Game Result'] == 0]['Gold Per Minute'], label = 'Loss', ax = axes[1,1], kde = False)
ax.legend()
ax.set_title('Guardian')
plt.xlim(375,850)

#Victory, God
count = incon_df.groupby(['God'])['Game Result'].count()
count = count[count >= 5]
wins = incon_df.groupby(['God'])['Game Result'].sum()
percent = round((wins/count), 2)
god_wins = pd.concat([count, wins, percent], axis =1, join = 'inner', keys =['Game Count', 'Total Wins', 'Win %'], sort=True)
god_wins.reset_index(level=0, inplace = True)

god_wins['Class'] = create_class(god_wins, 'God')
sns.scatterplot(x = 'Game Count', y = 'Win %', hue = 'Class', data = god_wins)
plt.text(48, 0.59, 'Scylla', size = 'medium')
plt.text(45, 0.73, 'Apollo', size = 'medium')
plt.text(41, 0.44, 'Anhur', size = 'medium')
plt.text(12, 0.1, 'Janus', size = 'medium')
plt.text(10, 0.84, 'Discordia', size = 'small')
plt.text(16, 0.83, 'Arachne', size = 'small')
plt.title('God Win %')

#Victory, and Ranked
sns.barplot( x = 'Ranked', y = 'Game Result', hue = 'Class', data = incon_df, ci=None)
plt.title('Ranked Win %')
plt.legend(prop={'size': 6})
labels = ['Unranked', 'Ranked']
plt.xticks(ticks = [0,1], labels = labels, size = 12)
plt.ylabel('Win%', fontsize=12)
plt.xlabel(None)

#Victory, Damage Dealt
sns.boxplot(x='Game Result', y='Damage Dealt', hue = 'Class', data= incon_df)
plt.title('Game Result and Damage Dealt Per Minute')
plt.xticks(ticks = [0,1], labels = ['Defeat', 'Victory'], size = 12)
plt.legend(prop={'size': 6})

#Victory, Relic, Class
sns.barplot(x='Relic 2', y='Game Result', hue = 'Class', data = incon_df, ci=None)
plt.xlabel(None)
plt.title('1st Relic Chosen', fontdict={'fontsize': 18, 'fontweight': 'bold'})
sns.barplot(x='Relic 1', y='Game Result', hue = 'Class', data = incon_df, ci=None)
plt.xlabel(None)
plt.title('2nd Relic Chosen', fontdict={'fontsize': 18, 'fontweight': 'bold'})

relic_df_2nd = incon_df['Relic 1'].value_counts(sort=True)
relic_df_1st = incon_df['Relic 2'].value_counts(sort=True)
relic_df = pd.concat([relic_df_1st.rename('1st Chosen'), relic_df_2nd.rename('2nd Chosen')], axis=1, join='outer', sort=True)
relic_df['1st Chosen'] = pd.to_numeric(relic_df['1st Chosen'], errors = 'coerce')
relic_df['1st Chosen'] = relic_df['1st Chosen'].fillna(0)
relic_df['1st Chosen'] = relic_df['1st Chosen'].astype(int)
relic_df['Game Total'] = relic_df['1st Chosen'] + relic_df['2nd Chosen']
relic_df['Game Total %'] = relic_df['Game Total'] / 1000
relic_df = relic_df[['Game Total', 'Game Total %']].sort_values(by=['Game Total'], ascending = False)
relic_df = relic_df.reset_index()
sns.barplot(x='index', y='Game Total', data=relic_df)
plt.title('Games Used Relic')
plt.xlabel(None)

#Victory, Class, Kills, Deaths, Assists
labels = ['Defeat', 'Victory']
fig, ax = plt.subplots(1,3)
sns.stripplot(x='Game Result', y='Kills', data = incon_df, hue='Class', ax = ax[0])
ax[0].set_xticklabels(labels = labels)
ax[0].set_xlabel(None)
ax[0].set_ylabel(None)
ax[0].set_title('Kills')
ax[0].legend(loc=2, prop={'size':6})
sns.stripplot(x='Game Result', y='Deaths', data = incon_df, hue='Class', ax = ax[1])
ax[1].set_xticklabels(labels = labels)
ax[1].set_xlabel(None)
ax[1].set_ylabel(None)
ax[1].set_title('Deaths')
ax[1].legend(loc=1, prop={'size':6})
sns.stripplot(x='Game Result', y='Assists', data = incon_df, hue='Class', ax = ax[2])
ax[2].set_xticklabels(labels = labels)
ax[2].set_xlabel(None)
ax[2].set_ylabel(None)
ax[2].set_title('Assists')
ax[2].legend(loc=2, prop={'size':6})


#Victory, Team Healing
sns.barplot(x= 'Team Healing', y='Game Result', hue = 'Class', data=incon_df, ci=None)
plt.xticks(ticks = [0,1], labels = ['No Healing', 'Healing'], size = 12)
plt.xlabel(None)
plt.title('Wins While Playing a Healer')

#Victory, class, Wards
sns.boxplot(x='Game Result', y = 'Wards', hue = 'Class', data=incon_df)
plt.xticks(ticks = [0,1], labels = ['Loss', 'Wins'], size = 12)
plt.xlabel(None)
plt.title('Game Result with Wards Placed')
plt.legend(loc=2, prop={'size':6})

#Victory, Structure Damage
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,4))
sns.distplot(assassin[assassin['Game Result'] == 0]['Structure Damage'], ax = ax[0,0], label = 'Loss', bins = 15, kde=False)
sns.distplot(assassin[assassin['Game Result'] == 1]['Structure Damage'], ax = ax[0,0], label='Win', bins = 15, kde=False)
ax[0,0].set_title('Assassin')
ax[0,0].legend()

sns.distplot(mage[mage['Game Result'] == 0]['Structure Damage'], label = 'Loss', ax = ax[0,1], bins = 15, kde=False)
sns.distplot(mage[mage['Game Result'] == 1]['Structure Damage'], label='Win', ax=ax[0,1], bins = 15, kde=False)
ax[0,1].set_title('Mage')
ax[0,1].legend()

sns.distplot(hunter[hunter['Game Result'] == 0]['Structure Damage'], label = 'Loss', ax = ax[1,0], bins = 15, kde=False)
sns.distplot(hunter[hunter['Game Result'] == 1]['Structure Damage'], label='Win', ax = ax[1,0], bins = 15, kde=False)
ax[1,0].set_title('Hunter')
ax[1,0].legend()

sns.distplot(warrior[warrior['Game Result'] == 0]['Structure Damage'], label = 'Loss', ax = ax[1,1], bins = 15, kde=False)
sns.distplot(warrior[warrior['Game Result'] == 1]['Structure Damage'], label='Win', ax= ax[1,1], bins = 15, kde=False)
ax[1,1].set_title('Warrior')
ax[1,1].legend()
fig.suptitle('Structure Damage vs. Game Result by Class')

#Victory, Class, Distance Traveled
#Side Note: There's a mage outlier on the graph that show 43645.7 (10 min per game), 0.0m (21 min oer game )
# and 6763.333 (24 min game) meter per minute. Left out; threw graph off
sns.boxplot(x='Game Result', y='Distance Traveled', hue = 'Class', data= incon_df)
plt.title('Game Result and Distance Traveled per Minute')
plt.xticks(ticks = [0,1], labels = ['Defeat', 'Victory'], size = 12)
plt.legend(prop={'size': 6})
plt.ylim(9000, 30000)
#pd.set_option('display.max_columns', 40)
#pd.reset_option('display')


#Victory, Inhand as a percentage of damage dealt
incon_df['InhandDD%'] = incon_df['In Hand Damage Dealt'] / incon_df['Damage Dealt']
FG = sns.FacetGrid(incon_df, row='Class', col='Game Result')
FG.map(sns.distplot, 'InhandDD%', kde=False)
incon_df.drop(labels=['InhandDD%'], axis=1, inplace=True)

#Victory, Damage Mitigated
sns.violinplot(x='Game Result', y='Damage Mitigated', hue='Class', data=incon_df)
plt.xticks(ticks = [0,1], labels = ['Defeat', 'Victory'], size = 12)
plt.title('Victory vs Damage Mitigated')

#victory, Damage Taken
sns.violinplot(x='Game Result', y='Damage Taken', hue='Class', data=incon_df)
plt.xticks(ticks = [0,1], labels = ['Defeat', 'Victory'], size = 12)
plt.title('Victory vs Damage Taken')
plt.legend(loc=8, prop={'size': 6})


