import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
from collections import Counter
from scipy import stats
#from plotnine import *
import sklearn

pd.set_option('display.expand_frame_repr', False) #expands columns in pandas removing truncation

df = pd.read_csv(r'C:\Users\jason\Desktop\vgsales.csv')
col_names = df.columns

# Remove unwanted columns
df = df[['Name', 'Platform', 'Genre', 'Publisher', 'Developer', 'Global_Sales', 'NA_Sales', 'PAL_Sales', 'JP_Sales', 'Other_Sales', 'Release_Date']]

# import ign df with scores
ign = pd.read_csv(r'C:\Users\jason\Desktop\ign.csv')
ign = ign[['title', 'platform', 'score', 'editors_choice', 'release_year', 'release_month', 'release_day']]

# Rename date columns to capitol for consistency
ign.rename(columns={'release_year': 'Release_Year', 'release_month': 'Release_Month', 'release_day': 'Release_Day'},
           inplace=True)


# Fix platform data in ign so it matches that in games df
consoles_ign = ['NES', 'Super NES', 'Nintendo 64', 'GameCube', 'Wii', 'Wii U', 'NS', 'Game Boy', 'Game Boy Color',
                'Game Boy Advance', 'Game Gear', 'Nintendo DS', 'Nintendo 3DS', 'PlayStation', 'PlayStation 2', 'PlayStation 3',
                'PlayStation 4', 'PlayStation Portable', 'PlayStation Vita', 'Xbox', 'Xbox 360', 'Xbox One', 'Genesis', 'Dreamcast']

consoles_games = ['NES', 'SNES', 'N64', 'GC', 'Wii', 'WiiU', 'NS', 'GB', 'GBC', 'GBA', 'GG', 'DS', '3DS', 'PS', 'PS2',
            'PS3', 'PS4', 'PSP', 'PSV', 'XB', 'X360', 'XOne', 'GEN', 'DC']



ign = ign.replace(to_replace=consoles_ign, value=consoles_games)


# ~~~~~~~~~PLATFORM~~~~~~~~~~

# Looking at the different types of Platforms to remove unnecessary ones (nobody cares about the commodore 64 from '85)
df.Platform.value_counts(dropna=False)


consoles = ['NES', 'SNES', 'N64', 'GC', 'Wii', 'WiiU', 'NS', 'GB', 'GBA', 'GBC', 'GG', 'DS', '3DS', 'PS', 'PS2',
            'PS3', 'PS4', 'PSP', 'PSV', 'XB', 'X360', 'XOne', 'GEN', 'DC']


# Removes all rows NOT associated with the above consoles and makes new db, games, by merging
vgsales = df[df['Platform'].isin(consoles)]
ign = ign[ign['platform'].isin(consoles)]


# join the two dfs based on game name (and add platform)
games = pd.merge(vgsales, ign, left_on=['Name', 'Platform'], right_on=['title', 'platform'], how='left')




# Fixing Sales columns (removing m)
games.Global_Sales = games.Global_Sales.str.replace('m', '')
games.NA_Sales = games.NA_Sales.str.replace('m', '')
games.PAL_Sales = games.PAL_Sales.str.replace('m', '')
games.JP_Sales = games.JP_Sales.str.replace('m', '')
games.Other_Sales = games.Other_Sales.str.replace('m', '')

# Splitting Release_Date into Day/Month/Year
games['Day'] = games.Release_Date.str[0:2]
games['Month'] = games.Release_Date.str[5:8]
games['Year'] = games.Release_Date.str[9:]

# Converting to appropriate data types
games.Genre = games.Genre.astype('category')
games.Name = games.Name.astype('str')
games.Platform = games.Platform.astype('category')
games.Publisher = games.Publisher.astype('category')
games.Developer = games.Developer.astype('category')
games.Global_Sales = games.Global_Sales.astype('float')
games.NA_Sales = games.NA_Sales.astype('float')
games.PAL_Sales = games.PAL_Sales.astype('float')
games.JP_Sales = games.JP_Sales.astype('float')
games.Other_Sales = games.Other_Sales.astype('float')
games.title = games.title.astype('str')
games.platform = games.platform.astype('category')
games.editors_choice = games.editors_choice.astype('category')



# Converting to Datetime
games.Release_Date = games.Month + '/' + games.Day + '/' + games.Year.astype('str') #combine to single column
games.Release_Date = [None if pd.isnull(dates) else datetime.strptime(str(dates), '%b/%d/%y') for dates in games.Release_Date]

# Remove titles with 1970 release dates (wrong)
games = games[games['Release_Date'].dt.year >= 1980]

# Fixing titles discovered incorrect dates
# check on "A value is trying to be set on a copy of a slice"
games.Release_Date[21278] = games.Release_Date[21278].replace(2003)
games.Release_Date[2232] = games.Release_Date[2232].replace(2010)
games.Release_Date[11572] = games.Release_Date[11572].replace(2007)
games.Release_Date[17935] = games.Release_Date[17935].replace(2008)
games.Release_Date[25883] = games.Release_Date[25883].replace(2009)
games.Release_Date[15941] = games.Release_Date[15941].replace(2006)
games.Release_Date[22295] = games.Release_Date[22295].replace(2007)
games.Release_Date[1005] = games.Release_Date[1005].replace(2012)
games.Release_Date[1751] = games.Release_Date[1751].replace(2009)
games.Release_Date[10413] = games.Release_Date[10413].replace(2013)
games.Release_Date[19837] = games.Release_Date[19837].replace(2006)
games.Release_Date[21229] = games.Release_Date[21229].replace(2011)
games.Release_Date[21763] = games.Release_Date[21763].replace(2011)
games.Release_Date[2012] = games.Release_Date[2012].replace(2011)
games.Release_Date[11368] = games.Release_Date[11368].replace(2014)
games.Release_Date[5363] = games.Release_Date[5363].replace(2000)
games.Release_Date[6033] = games.Release_Date[6033].replace(2001)





# Find other titles with improper release and no info, as discovered (never released, cancelled, etc)
consoles = ['NES', 'SNES', 'N64', 'GC', 'Wii', 'WiiU', 'NS', 'GB', 'GBA', 'GBC', 'GG', 'DS', '3DS', 'PS', 'PS2',
            'PS3', 'PS4', 'PSP', 'PSV', 'XB', 'X360', 'XOne', 'GEN', 'DC']
begin = [1983, 1990, 1996, 2001, 2006, 2012, 2017, 1989, 2001, 1998, 1990, 2004, 2011, 1994, 2000,
            2006, 2013, 2005, 2011, 2001, 2005, 2013, 1988, 1998]
end = [1995, 1999, 2002, 2007, 2017, 2018, 2020, 2003, 2008, 2003, 1997, 2016, 2020, 2006, 2013,
            2017, 2020, 2016, 2020, 2009, 2017, 2020, 1999, 2001]
life = [j - i for i, j in zip(begin, end)]


systems_info = pd.DataFrame({'Consoles': consoles,
                             'Start': begin,
                             'Discontinued': end,
                             'Lifespan': life})

systems_info = systems_info.sort_values('Start')
systems_info = systems_info.reset_index(inplace=False)
systems_info = systems_info.drop('index', axis=1)



# Create df's for each system with erroneous dates to fix as needed
bad_NES = games.loc[((games['Release_Date'].dt.year < begin[0]) | (games['Release_Date'].dt.year > end[0]))
                      & (games['Platform'] == consoles[0])]
bad_SNES = games.loc[((games['Release_Date'].dt.year < begin[1]) | (games['Release_Date'].dt.year > end[1]))
                      & (games['Platform'] == consoles[1])]
bad_N64 = games.loc[((games['Release_Date'].dt.year < begin[2]) | (games['Release_Date'].dt.year > end[2]))
                      & (games['Platform'] == consoles[2])]
bad_GC = games.loc[((games['Release_Date'].dt.year < begin[3]) | (games['Release_Date'].dt.year > end[3]))
                      & (games['Platform'] == consoles[3])]
bad_Wii = games.loc[((games['Release_Date'].dt.year < begin[4]) | (games['Release_Date'].dt.year > end[4]))
                      & (games['Platform'] == consoles[4])]
bad_WiiU = games.loc[((games['Release_Date'].dt.year < begin[5]) | (games['Release_Date'].dt.year > end[5]))
                      & (games['Platform'] == consoles[5])]
bad_NS = games.loc[((games['Release_Date'].dt.year < begin[6]) | (games['Release_Date'].dt.year > end[6]))
                      & (games['Platform'] == consoles[6])]
bad_GB = games.loc[((games['Release_Date'].dt.year < begin[7]) | (games['Release_Date'].dt.year > end[7]))
                      & (games['Platform'] == consoles[7])]
bad_GBA = games.loc[((games['Release_Date'].dt.year < begin[8]) | (games['Release_Date'].dt.year > end[8]))
                      & (games['Platform'] == consoles[8])]
bad_GBC = games.loc[((games['Release_Date'].dt.year < begin[9]) | (games['Release_Date'].dt.year > end[9]))
                      & (games['Platform'] == consoles[9])]
bad_GG = games.loc[((games['Release_Date'].dt.year < begin[10]) | (games['Release_Date'].dt.year > end[10]))
                      & (games['Platform'] == consoles[10])]
bad_DS = games.loc[((games['Release_Date'].dt.year < begin[11]) | (games['Release_Date'].dt.year > end[11]))
                      & (games['Platform'] == consoles[11])]
bad_3DS = games.loc[((games['Release_Date'].dt.year < begin[12]) | (games['Release_Date'].dt.year > end[12]))
                      & (games['Platform'] == consoles[12])]
bad_PS = games.loc[((games['Release_Date'].dt.year < begin[13]) | (games['Release_Date'].dt.year > end[13]))
                      & (games['Platform'] == consoles[13])]
bad_PS2 = games.loc[((games['Release_Date'].dt.year < begin[14]) | (games['Release_Date'].dt.year > end[14]))
                      & (games['Platform'] == consoles[14])]
bad_PS3 = games.loc[((games['Release_Date'].dt.year < begin[15]) | (games['Release_Date'].dt.year > end[15]))
                      & (games['Platform'] == consoles[15])]
bad_PS4 = games.loc[((games['Release_Date'].dt.year < begin[16]) | (games['Release_Date'].dt.year > end[16]))
                      & (games['Platform'] == consoles[16])]
bad_PSP = games.loc[((games['Release_Date'].dt.year < begin[7]) | (games['Release_Date'].dt.year > end[17]))
                      & (games['Platform'] == consoles[17])]
bad_PSV = games.loc[((games['Release_Date'].dt.year < begin[18]) | (games['Release_Date'].dt.year > end[18]))
                      & (games['Platform'] == consoles[18])]
bad_XB = games.loc[((games['Release_Date'].dt.year < begin[19]) | (games['Release_Date'].dt.year > end[19]))
                      & (games['Platform'] == consoles[19])]
bad_X360 = games.loc[((games['Release_Date'].dt.year < begin[20]) | (games['Release_Date'].dt.year > end[20]))
                      & (games['Platform'] == consoles[20])]
bad_XOne = games.loc[((games['Release_Date'].dt.year < begin[21]) | (games['Release_Date'].dt.year > end[21]))
                      & (games['Platform'] == consoles[21])]
bad_GEN = games.loc[((games['Release_Date'].dt.year < begin[22]) | (games['Release_Date'].dt.year > end[22]))
                      & (games['Platform'] == consoles[22])]
bad_DC = games.loc[((games['Release_Date'].dt.year < begin[23]) | (games['Release_Date'].dt.year > end[23]))
                      & (games['Platform'] == consoles[23])]
bad_DC.drop(24721, inplace=True) # drop value that actually was released outside of release date so not removed below


# Drop games with no info and bad dates (assume never released) 246 games
games = games.drop(bad_NES.index.values)
games = games.drop(bad_SNES.index.values)
games = games.drop(bad_GC.index.values)
games = games.drop(bad_Wii.index.values)
games = games.drop(bad_WiiU.index.values)
games = games.drop(bad_GB.index.values)
games = games.drop(bad_GBA.index.values)
games = games.drop(bad_DS.index.values)
games = games.drop(bad_3DS.index.values)
games = games.drop(bad_PS.index.values)
games = games.drop(bad_PS2.index.values)
games = games.drop(bad_PS3.index.values)
games = games.drop(bad_PSP.index.values)
games = games.drop(bad_X360.index.values)
games = games.drop(bad_GEN.index.values)
games = games.drop(bad_DC.index.values)


# Check on number of missing values
games.isnull().sum()













# ~~~~~~~~~~~EDA~~~~~~~~~~~~~~~
# game sales by year
by_year = games.groupby(by=games['Release_Date'].dt.year).sum()
by_year.reset_index(inplace=True)
sns.set(style='whitegrid', font_scale=0.75)
gsby = sns.barplot(x='Global_Sales', y='Release_Date', data=by_year, orient='h', color='green').set_title('Global Video Game Sales')
plt.xlabel('Sales in Thousands ($)')
plt.ylabel('Year')
plt.show()
#1


# number of games released each year
sns.set(style='whitegrid', font_scale=0.75)
gpy = sns.countplot(y=games['Release_Date'].dt.year, data=games, color='blue').set_title('Global Video Games Sold Yearly')
plt.xlabel('Games')
plt.ylabel('Year')
plt.show()
#2

# histogram of game scores
scores = games[np.isfinite(games['score'])] # 8k games
sns.distplot(scores.score, bins=19, kde=False).set_title('Distribution of Game Scores')
sns.set(style='ticks')
plt.ylabel('Count')
plt.xlabel('Score')
plt.show()
#3


# Global Sales by Console
by_console = games.groupby(by='Platform').sum()
by_console.reset_index(inplace=True)
by_console = by_console[by_console['Platform'] != 'GG']
by_console = by_console.sort_values(['Global_Sales']).reset_index(drop=True)
order = by_console.Platform
gsbc = sns.catplot(x='Platform', y='Global_Sales', kind='bar', data=by_console, order=order, palette=('Blues_d'))
plt.xticks(rotation=45)
plt.ylabel('Sales in Thousands ($)')
sns.set(style='whitegrid', font_scale=0.75)
plt.title('Global Sales by Platform')
plt.show()
#4


# Average score by genre
by_genre = games.groupby(by='Genre').mean()
by_genre.reset_index(inplace=True)
by_genre = by_genre.sort_values(['score']).reset_index(drop=True)
order = by_genre.Genre
sns.catplot(y='Genre', x='score', data=by_genre, kind='bar', palette='Greens_d',
            order=order)
plt.xlabel('Score')
plt.title('Average Scores by Genre')
plt.show()
#can see that there is little change in avg score

sns.boxplot(y='Genre', x='score', data=games, order=order)
plt.title('Scores by Genre')
plt.xlabel('Score')
plt.show()



# Sales by genre
by_genre = games.groupby(by='Genre').sum()
by_genre.reset_index(inplace=True)
by_genre = by_genre.sort_values(['Global_Sales']).reset_index(drop=True)
order = by_genre.Genre
sns.catplot(y='Genre', x='Global_Sales', data=by_genre, kind='bar', order=order, palette=('Greens_d'))
plt.xlabel('Global Sales in Thousands ($)')
plt.title('Global Sales by Genre')
plt.show()



# So scores don't seem to have any correlation to sales on a large scale by genre
# By Region
f, axes = plt.subplots(1, 4, sharey=True)
sns.catplot(y='Genre', x='NA_Sales', data=by_genre, kind='bar', order=order, palette=('Greens_d'), ax=axes[0])
sns.catplot(y='Genre', x='JP_Sales', data=by_genre, kind='bar', order=order, palette=('Greens_d'), ax=axes[1])
sns.catplot(y='Genre', x='PAL_Sales', data=by_genre, kind='bar', order=order, palette=('Greens_d'), ax=axes[2])
sns.catplot(y='Genre', x='Other_Sales', data=by_genre, kind='bar', order=order, palette=('Greens_d'), ax=axes[3])



# Year vs Platform by type/group
handheld_systems = ['GB', 'GG', 'GBC', 'GBA', 'DS', '3DS', 'PSP', 'PSV', 'NS']
nintendo_systems = ['NES', 'SNES', 'N64', 'GC', 'Wii', 'WiiU', 'NS']
microsoft_systems = ['XB', 'X360', 'XOne', 'GEN', 'DC']
ps_systems = ['PS', 'PS2', 'PS3', 'PS4', 'PSP', 'PSV']
early_consoles = ['NES', 'SNES', 'GEN', 'PS', 'N64', 'DC']
advanced_consoles = ['PS2', 'GC', 'XB', 'PSP', 'DS', 'X360', 'PS3', 'Wii', 'XOne', 'PS4', 'NS']

handhelds = games[games['Platform'].isin(handheld_systems)]
nintendo = games[games['Platform'].isin(nintendo_systems)]
microsoft = games[games['Platform'].isin(microsoft_systems)]
playstation = games[games['Platform'].isin(ps_systems)]
earlygen = games[games['Platform'].isin(early_consoles)]
advancedgen = games[games['Platform'].isin(advanced_consoles)]


sns.set(style='whitegrid')

fig, axes = plt.subplots(2, 2)
plt1 = sns.violinplot(x=handhelds['Release_Date'].dt.year, y='Platform', data=handhelds, scale='count', order=handheld_systems)
plt2 = sns.violinplot(x=nintendo['Release_Date'].dt.year, y='Platform', data=nintendo, scale='count', order=nintendo_systems)
plt3 = sns.violinplot(x=microsoft['Release_Date'].dt.year, y='Platform', data=microsoft, scale='count', order=microsoft_systems)
plt4 = sns.violinplot(x=playstation['Release_Date'].dt.year, y='Platform', data=playstation, scale='count', order=ps_systems)


earlygenplot = sns.violinplot(x=earlygen['Release_Date'].dt.year, y='Platform', data=earlygen, scale='count', order=early_consoles, inner='stick')
advancedgenplot = sns.violinplot(x=advancedgen['Release_Date'].dt.year, y='Platform', data=advancedgen, palette='Set1', scale='count', order=advanced_consoles)

allplot = sns.violinplot(x=games['Release_Date'].dt.year, y='Platform', data=games, scale='count', order=systems_info.Consoles)






# How to facet this?
# Top selling games by region and platform/genre?
colnames = games.columns

def top_ten(DF, platforms):
    "This gets the top ten games in sales by region for each platform in a list and makes a new df"
    top_sellers = pd.DataFrame(columns=colnames)
    for item in platforms:
        X = DF[DF['Platform'].isin([item])]
        XNA = X.sort_values(by='NA_Sales', ascending=False).head(10)
        XPAL = X.sort_values(by='PAL_Sales', ascending=False).head(10)
        XJP = X.sort_values(by='JP_Sales', ascending=False).head(10)
        XOther = X.sort_values(by='Other_Sales', ascending=False).head(10)

        Y = pd.concat([XNA, XPAL])
        Z = pd.concat([XJP, XOther])
        combined = pd.concat([Y, Z])

        top_sellers = pd.concat([top_sellers, combined])

    return top_sellers


top_games = top_ten(games, consoles)






# Top games globally and platform/type (write genre in bar)
top_sellers = games.groupby(by='Name').sum()
top_sellers = top_sellers.sort_values(by='Global_Sales', ascending=False).head(10)
top_sellers = top_sellers.reset_index(inplace=False)
#top_sellers = top_sellers.reset_index(inplace=True)
sns.catplot(x='Global_Sales', y='Name', kind='bar', data=top_sellers)
plt.show()


# average global sales per year by platform
plt.subplots(figsize=(15, 15))
test = games.groupby('Platform')['Platform'].count()
test.sort_values(ascending=True, inplace=True)
mean_test = games[games['Platform'].isin(test.index)]
abc = mean_test.groupby([mean_test['Release_Date'].dt.year, 'Platform'])['Global_Sales'].sum().reset_index()


abc = abc.pivot('Release_Date', 'Platform', 'Global_Sales')
abc = abc[['NES', 'GEN', 'GB', 'SNES', 'GG', 'PS', 'N64', 'GBC', 'DC', 'PS2', 'GC', 'XB',
                                                   'GBA', 'DS', 'PSP', 'X360', 'Wii', 'PS3', '3DS', 'PSV', 'WiiU',
                                                   'XOne', 'PS4', 'NS']]
abc = np.round(abc, decimals=0) # remove decimals to avoid scientific notation
sns.heatmap(abc, annot=True, cmap='YlGnBu', linewidths=0.4, fmt='g')
plt.title('Global Sales by Platforms')
plt.yticks(rotation=0)
plt.show()



# Top Publishers

top_pub = games.groupby('Publisher').sum()
top_pub = top_pub.sort_values('Global_Sales', ascending=False).head(10)
top_pub = top_pub.reset_index(drop=False)
ord = list(top_pub.Publisher.head(10))
sns.catplot(x='Global_Sales', y='Publisher', data=top_pub, kind='bar', order=ord)
plt.title('Top Publishers')
plt.show()

# publishers by sales/game
game_count = ([1048, 1089, 836, 910, 609, 1089, 863, 726, 1203, 1258])
se = pd.Series(game_count)
top_pub['Game_Count'] = se.values
ppg = top_pub['Global_Sales']/top_pub['Game_Count']
top_pub['SPG'] = ppg.values
sns.catplot(x='SPG', y='Publisher', data=top_pub, kind='bar', order=ord)



# Top Developers
top_dev = games.groupby('Developer').sum()
top_dev = top_dev.sort_values('Global_Sales', ascending=False).head(10)
top_dev = top_dev.reset_index(drop=False)
ord = list(top_dev.Developer.head(10))
sns.catplot(x='Global_Sales', y='Developer', data=top_dev, kind='bar', order=ord)
plt.show()




pandas.get_dummies



# Prepare model

# Cleaning up df

games = games.drop(['title', 'platform', 'Day'], axis=1)

# drop
test = games.drop(['Name'], axis=1)

# create test/train sets for prediction
gamestest = test[(test['Release_Date'].dt.year < 2018)]
gamestrain = test[(test['Release_Date'].dt.year > 2017)]


reg_all = LinearRegression()
reg_all.fit(gamestest, gamestrain)
reg_all.predict(gamestest)












# ~~~~~~~~~~~Questions~~~~~~~~~~~~


# Help with subplotting (333)

# sns in jupyter

# Errors about "value trying to be set on a copy of slice" mute? (setting year values for erroneous entries) (~100)

# Top dev (~430) why does it not cut off after top 10, tries to graph all entries?

# for a regression fit, does everything have to be a numerical value? (genre, platform, etc)?







# what to predict: Sales of a game, time series prediction

# linear regression, tree based, polynomials on time variables (year * action in genre),
# remove game name for modelleing (lasso, gbm, random forrest)