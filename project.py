import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from mpl_toolkits.mplot3d import Axes3D

movie_data = pd.read_csv("./archive/IMDB Top 250 Movies.csv")
#print(movie_data.tail())
modmovie_data = movie_data[movie_data.budget.apply(lambda x: x.isnumeric())]
#print(movie_data['budget'])
#print(modmovie_data["budget"])
#print(modmovie_data.columns)
#modmovie_data.to_csv('modified.csv')
modmovie_data = modmovie_data[modmovie_data.box_office.apply(lambda x: x.isnumeric())]
#modmovie_data.to_csv('modified.csv')
#print(modmovie_data.index)
#print(modmovie_data.iloc[1]['name'])

modmovie_data = pd.read_csv("modified.csv")
#print(modmovie_data['rank'])
#print(modmovie_data.year)
first_genre = lambda x:  x if ',' not in x else x.partition(',')[0]
colours = {'Drama':'red', 'Crime':'orange', 'Action':'yellow', 'Biography':'green', 
           'Adventure':'blue', 'Comedy':'purple', 'Animation':'pink', 'Horror':'brown', 'Western':'gray', 'Mystery':'teal'}
def unique_categories(arr):
    unique = []
    for i in arr:
        if i not in unique:
            unique.append(i)
    print(unique)


modmovie_data['first_gen'] = modmovie_data.genre.apply(first_genre)
"""
ax =sns.scatterplot(data=modmovie_data, x='year', y='rank', hue='first_gen', ec=None)
ax.legend(title='Primary Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_title("IMDb Movie Ranking Over Release Year and Primary Genre")
plt.show()


#unique_categories(modmovie_data.genre.apply(first_genre).to_numpy())
c = modmovie_data.genre.apply(first_genre).apply(lambda x: colours[x]).values
plt.scatter(modmovie_data.year, modmovie_data['rank'], c=c)
for category, colour in colours.items():
    plt.scatter([],[], c=colour, label = category)
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
plt.xlabel('Year Released')
plt.ylabel('Rank')
plt.title('iMDb Movie Ranking over Release Year and Primary Genre')
plt.show()
"""
"""
#unique_categories(modmovie_data.certificate.to_numpy())
colours = {'G':'green', 'PG':'yellow', 'PG-13':'orange', 'R':'red', 'Approved':'blue', 'Passed':'blue', 'Not Rated':'gray', 'Unrated':'gray', 
           '13+':'orange', '18+':'red', 'X':'red', 'GP':'yellow'}
c = modmovie_data.certificate.apply(lambda x: colours[x]).values
plt.scatter(modmovie_data.year, modmovie_data['rank'], c=c)
for category, colour in colours.items():
    plt.scatter([],[], c=colour, label = category)
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
plt.xlabel('Year Released')
plt.ylabel('Rank')
plt.title('iMDb Movie Ranking over Release Year and Age Rating')
plt.show()

counts = modmovie_data.certificate.value_counts()
plt.bar(counts.index, counts.values)
plt.title("Total Number of Films with Each Certificate")
plt.show()



modmovie_data['yearbin'] = pd.cut(modmovie_data['year'], bins=np.arange(1910, 2025, 20))
g = sns.catplot(data=modmovie_data, x='certificate', col='yearbin', kind='count', palette='Set2')
g.add_legend(title="Certificate")
plt.show()

"""
#pairwise plot
minutes = lambda time: (int(time.split(' ')[0].rstrip('h')) * 60) + int(time.split(' ')[1].rstrip('m'))
modmovie_data['run_min'] = modmovie_data['run_time'].apply(minutes)

included = ["rank", "year", "run_min", "budget", "box_office"]


pp = sns.pairplot(modmovie_data[included])
"""
log_columns = ["budget", "box_office"]
for ax in pp.axes.flat:
    if ax.get_xlabel() in log_columns:
        ax.set(xscale='log')
    if ax.get_ylabel() in log_columns:
        ax.set(yscale='log')
"""
plt.suptitle('Pairplot of Numerical Variables in IMDb Data')
plt.show()


"""
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
ax = sns.heatmap(modmovie_data.groupby(['first_gen', 'run_min']).size().unstack(), cmap="cool")
ax.set_title('Frequency of Primary Genre and Runtime')
ax.set_xlabel('Runtime (minutes)')
ax.set_ylabel('Primary Genre')
plt.show()

"""

categories = ['rating', 'box_office', 'budget']
N = len(categories)

means = modmovie_data.groupby('first_gen')[categories].mean()
means = means.reset_index(drop=False)

# create a list of values for each genre
values = means.loc[:, categories].values.tolist()
values += values[:1]


# calculate angles for each category
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# create the radar chart
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)

rat_max = max(means['rating'])
box_max = max(means['box_office'])
bud_max = max(means['budget'])

# plot the values for each genre
for i in range(len(means)):
    genre_values = means.loc[i, categories].values.tolist()
    #genre_values[0] = genre_values[0]/rat_max
    genre_values[1] = genre_values[1]/box_max
    genre_values[2] = genre_values[2]/bud_max

    genre_values += genre_values[:1]
    
    ax.plot(angles, genre_values, linewidth=1, linestyle='solid', label=means.loc[i, 'first_gen'])
    ax.fill(angles, genre_values, alpha=0.1)

# add the categories as labels on the x-axis
ax.set_thetagrids(np.degrees(angles[:-1]), categories)

# add a legend
ax.legend(loc='lower right', bbox_to_anchor=(-0.05, 0.2))

# show the plot
plt.show()

#print(modmovie_data.budget)
#print(modmovie_data.box_office)


plt.scatter(pd.to_numeric(modmovie_data.budget), pd.to_numeric(modmovie_data.box_office), c=modmovie_data['rank'], cmap='cool', edgecolors='black')
plt.xlabel('Budget')
plt.ylabel('Box Office')
plt.xscale('log')
plt.yscale('log')
plt.colorbar().set_label('Rank')
plt.title("Logarithmic Relationship between Budget and Box Office")
plt.show()


import plotly.graph_objects as go

trace = go.Scatter3d(
    x=modmovie_data['budget'], y=modmovie_data['box_office'], z=modmovie_data['rank'], mode='markers',marker=dict(color=modmovie_data['rank'], colorscale='Viridis',size=5))


layout = go.Layout(scene=dict(xaxis=dict(type='log', title='Budget', titlefont=dict(size=16)),
                               yaxis=dict(type='log', title='Box Office', titlefont=dict(size=16)),
                               zaxis=dict(title='Rank', titlefont=dict(size=16))))


fig = go.Figure(data=[trace], layout=layout)
fig.show()

"""
fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')
ax.scatter(modmovie_data['budget'], modmovie_data['box_office'], modmovie_data['rank'], cmap='cool')
ax.invert_zaxis()
ax._set_yscale('log')
ax.set_xlabel('Budget')
ax.set_ylabel('Box Office')
ax.set_zlabel('Rank')

plt.show()

"""
"""
exploratory analysis:
- rank: descending order integers
- name: strings of varying length
- year: year released, integer
- rating: double, directly linked to rank
- genre: categorical? string
- certificate: these hands are rated E for everyone
- run_time: #h ##m
- tagline: string
- budget: integer
- casts: comma-separated strings
- directors: ""
- writers: ""
"""

"""
plot matrix?
find unique words in film titles vs rank
- maybe find other ways to categorise the words too (noun verb etc)
year vs rank
genre vs rank
run_time vs rank
genre vs rank vs year
tagline same with title
budget vs rank
budget vs directors ?
budget vs cast
"""

