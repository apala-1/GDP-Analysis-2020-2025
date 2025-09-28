import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

url = "https://statisticstimes.com/geography/countries-by-continents.php"
tables = pd.read_html(url)

for i, table in enumerate(tables):
    print(table.head())

df = tables[-1]
df_clean = df[['Country or Area','Continent']].rename(columns={'Country or Area':'Country'})
print(df_clean.head())

country = pd.read_csv('data.csv');
print(country.head())

country_ag = country.melt(id_vars='Country',var_name='year',value_name='gdp')
print(country_ag)

nepal = country_ag[country_ag['Country'] == 'Nepal']
print(nepal)
zimbabwe = country_ag[country_ag['Country'] == 'Zimbabwe']
print(zimbabwe)

sns.lineplot(x=nepal['year'],y=nepal['gdp'])
plt.title('GDP of Nepal throughout the years')
plt.xlabel('Years 2020 - 2025')
plt.ylabel('GDP of Nepal')

sns.lineplot(x=zimbabwe['year'],y=zimbabwe['gdp'])
plt.title('GDP of Zimbabwe throughout the years')
plt.xlabel('Years 2020 - 2025')
plt.ylabel('GDP of Zimbabwe')
plt.show()

mostGDP = country_ag[country_ag['gdp'] == max(country_ag['gdp'])]
print(mostGDP)

mostGDPCountry = mostGDP['Country'].iloc[0]
print(mostGDPCountry)

now = country_ag[country_ag['Country'] == mostGDPCountry]
print(now)

sns.lineplot(x=now['year'], y=now['gdp'])
plt.title('Highest GDP throughout the years')
plt.xlabel('Years 2020 - 2025')
plt.ylabel('GDP of United States')
plt.show()

top5GDP2020 = country_ag.sort_values(by=['gdp','Country'], ascending=[False, True]).head(25)
print(top5GDP2020)

sns.scatterplot(x=top5GDP2020['year'], y=top5GDP2020['gdp'], hue=top5GDP2020['Country'])
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.show()

country['avg_gdp'] = country[['2020','2021','2022','2023','2024','2025']].mean(axis=1)
print(country.head())

avg = country.sort_values(by='avg_gdp', ascending=False)
print(avg.head())

avg_2020 = avg.melt(id_vars='Country', value_name='gdp', var_name='year').head(5)
print(avg_2020)

sns.barplot(x=avg_2020['Country'], y=avg_2020['gdp'])
plt.show()

countries_need = ['United States', 'China', 'Japan', 'Germany', 'India']

top_5 = country_ag[country_ag['Country'].isin(countries_need)]
print(top_5)

sns.lineplot(data=top_5, x='year', y='gdp', hue='Country')
plt.show()

btm = country.sort_values(by='avg_gdp', ascending=True).head(5)
print(btm)

bottom_countries = ['Tuvalu','Nauru','Marshall Islands','Palau','Kiribati']

bottom_5 = country_ag[country_ag['Country'].isin(bottom_countries)]
print(bottom_5)

sns.lineplot(data=bottom_5, x='year', y='gdp', hue='Country')
plt.show()

top_and_btm = pd.concat([top_5, bottom_5], ignore_index=True)
print(top_and_btm)

# Together
sns.lineplot(data=top_and_btm, x='year', y='gdp', hue='Country')
plt.yscale('log')
plt.show()

new_country = country_ag.merge(df_clean, on='Country')
print(new_country.head())

continent_year = new_country.groupby(['Continent', 'year'])['gdp'].sum().reset_index()
print("Continent_year: ", continent_year)
heatmap_data = continent_year.pivot(index='Continent', values='gdp', columns='year')
print("Heatmap_data: ", heatmap_data)
sns.heatmap(data=heatmap_data, annot=True)
plt.show()

country_continent_year = new_country.groupby(['Continent', 'year', 'Country'])['gdp'].sum().reset_index()
print("Continent_year: ", country_continent_year)

fig, ax = plt.subplots(figsize=(10, 6))
years = sorted(country_continent_year['year'].unique())

def animate(i):
    ax.clear()
    current_year = years[i]
    
    data = country_continent_year[country_continent_year['year'] == current_year]
    
    sns.scatterplot(
        data=data,
        x="Continent",
        y="gdp",
        hue="Continent",
        size="gdp",
        sizes=(100, 2000),
        alpha=0.6,
        ax=ax,
        legend=False 
    )
    
    ax.set_title(f"GDP Bubble Plot - Year {current_year}", fontsize=16)
    ax.set_ylabel("GDP (log scale)")
    ax.set_xlabel("Continent")
    ax.set_yscale("log")
    ax.set_ylim(1e6, 1e8)

animation = FuncAnimation(fig, animate, frames=len(years), interval=1000, repeat=False)
plt.show()