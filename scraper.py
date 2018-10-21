from bs4 import BeautifulSoup
import urllib
import pandas as pd
import time

pages = 1
rec_count = 0

rank = []
boxart = []
gname = []
platform = []
genre = []
publisher = []
developer = []
vgchartz = []
critic = []
users = []
sales_gl = []
sales_na = []
sales_pal = []
sales_jp = []
sales_ot = []
release = []


genres = {'Action':38, 'Action-Adventure':2, 'Adventure':26, 'Board+Game':1, 'Fighting':11, 'Misc':51, 'MMO':1, 'Music':1,
         'Party':1, 'Platform':17, 'Puzzle':16, 'Racing':15, 'Role-Playing':22, 'Shooter':23, 'Simulation':13,
         'Sports':26, 'Strategy':16, 'Visual+Novel':1}

#url = 'http://www.vgchartz.com/games/games.php?page=38&results=200&name=&console=&keyword=&publisher=&genre=Action&order=Sales&ownership=Both&boxart=Both&banner=Both&showdeleted=&region=All&goty_year=&developer=&direction=DESC&showtotalsales=1&shownasales=1&showpalsales=1&showjapansales=1&showothersales=1&showpublisher=1&showdeveloper=1&showreleasedate=1&showlastupdate=1&showvgchartzscore=1&showcriticscore=1&showuserscore=1&alphasort=1'

for item in genres:
    pages = genres[item] +1
    for page in range(1, pages):
        surl = 'http://www.vgchartz.com/games/games.php?page={}&results=200&name=&console=&keyword=&publisher=&genre={}&order=Sales&ownership=Both&boxart=Both&banner=Both&showdeleted=&region=All&goty_year=&developer=&direction=DESC&showtotalsales=1&shownasales=1&showpalsales=1&showjapansales=1&showothersales=1&showpublisher=1&showdeveloper=1&showreleasedate=1&showlastupdate=1&showvgchartzscore=1&showcriticscore=1&showuserscore=1'.format(page, item)
        print(surl)
        r = urllib.request.urlopen(surl).read()
        time.sleep(1)
        soup = BeautifulSoup(r, features="lxml")
        print(page)
        chart = soup.find('div', id='generalBody').find('table')
        for row in chart.find_all('tr')[3:]:
            try:
                col = row.find_all('td')

                # extract data into column data
                column_1 = col[0].string.strip() #rank
                #column_2 = col[1].find('img')['alt'].strip() #boxart
                column_3 = col[2].find('a').string.strip() #name
                column_4 = col[3].find('img')['alt'].strip() #Console
                column_5 = col[4].string.strip() #publisher
                column_6 = col[5].string.strip() #Developer
                column_7 = col[6].string.strip() #CGChartz Score
                column_8 = col[7].string.strip() #Critic Score
                column_9 = col[8].string.strip() #User Score
                column_10 = col[9].string.strip() #Total Sales
                column_11 = col[10].string.strip() #NA Sales
                column_12 = col[11].string.strip() #PAL Sales
                column_13 = col[12].string.strip() #JP Sales
                column_14 = col[13].string.strip() #Other Sales
                column_15 = col[14].string.strip() #Release

                # Add Data to columns
                # Adding data only if able to read all of the columns
                rank.append(column_1)
                #boxart.append(column_2)
                gname.append(column_3)
                platform.append(column_4)
                genre.append(item)
                publisher.append(column_5)
                developer.append(column_6)
                vgchartz.append(column_7)
                critic.append(column_8)
                users.append(column_9)
                sales_gl.append(column_10)
                sales_na.append(column_11)
                sales_pal.append(column_12) #PAL
                sales_jp.append(column_13)
                sales_ot.append(column_14)
                release.append(column_15)


                rec_count += 1

            except:
                print('Got Exception')
                continue

columns = {'Rank': rank, 'Name': gname, 'Platform': platform, 'Genre': genre, 'Publisher': publisher,
           'Developer': developer, 'VGChartz_Score': vgchartz, 'Critic_Score': critic, 'User_Score': users,
           'Global_Sales': sales_gl, 'NA_Sales': sales_na, 'PAL_Sales': sales_pal, 'JP_Sales': sales_jp,
           'Other_Sales': sales_ot, 'Release_Date': release}

print (rec_count)
df = pd.DataFrame(columns)
print(df)
df = df[['Rank', 'Name', 'Platform', 'Genre', 'Publisher', 'Developer', 'VGChartz_Score', 'Critic_Score', 'User_Score',
         'Global_Sales', 'NA_Sales', 'PAL_Sales', 'JP_Sales', 'Other_Sales', 'Release_Date']]
del df.index.name
df.to_csv("vgsales.csv", sep=",", encoding='utf-8')