# COVID-19 Global Data Tracker
# Complete analysis in a single Python file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

def main():
    # Set visualization style
    plt.style.use('ggplot')
    sns.set_palette("husl")
    
    print("COVID-19 Global Data Analysis")
    print("============================")
    
    # 1. Data Loading
    print("\nLoading data...")
    try:
        df = pd.read_csv('owid-covid-data.csv')
        print("Dataset successfully loaded!")
    except FileNotFoundError:
        print("Error: Dataset file 'owid-covid-data.csv' not found.")
        print("Please download it from Our World in Data and save it in the same directory.")
        return
    
    # 2. Data Cleaning
    print("\nCleaning data...")
    df['date'] = pd.to_datetime(df['date'])
    
    countries_of_interest = ['Kenya', 'United States', 'India', 'Brazil', 'United Kingdom', 
                           'South Africa', 'Germany', 'China', 'Japan', 'Australia']
    
    df_filtered = df[df['location'].isin(countries_of_interest)].copy()
    
    key_columns = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths', 
                  'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']
    
    df_filtered[key_columns] = df_filtered.groupby('location')[key_columns].fillna(method='ffill')
    df_filtered[key_columns] = df_filtered[key_columns].fillna(0)
    
    df_filtered['death_rate'] = df_filtered['total_deaths'] / df_filtered['total_cases']
    df_filtered['cases_per_million'] = df_filtered['total_cases'] / df_filtered['population'] * 1e6
    df_filtered['deaths_per_million'] = df_filtered['total_deaths'] / df_filtered['population'] * 1e6
    df_filtered = df_filtered.dropna(subset=['date', 'location'])
    
    # 3. Analysis and Visualization
    print("\nGenerating visualizations...")
    
    # Create a figure with all plots
    plt.figure(figsize=(18, 24))
    
    # Plot 1: Total Cases Over Time
    plt.subplot(4, 2, 1)
    for country in countries_of_interest:
        country_data = df_filtered[df_filtered['location'] == country]
        plt.plot(country_data['date'], country_data['total_cases'], label=country)
    plt.title('Total COVID-19 Cases Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Cases')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # Plot 2: New Cases (7-day average)
    plt.subplot(4, 2, 2)
    for country in countries_of_interest:
        country_data = df_filtered[df_filtered['location'] == country]
        plt.plot(country_data['date'], country_data['new_cases'].rolling(7).mean(), label=country)
    plt.title('Daily New Cases (7-day Average)')
    plt.xlabel('Date')
    plt.ylabel('New Cases')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # Plot 3: Death Rates
    plt.subplot(4, 2, 3)
    latest_data = df_filtered.sort_values('date').groupby('location').last()
    latest_data = latest_data.sort_values('death_rate', ascending=False)
    sns.barplot(x=latest_data.index, y=latest_data['death_rate']*100, palette='viridis')
    plt.title('Death Rates by Country (%)')
    plt.xlabel('Country')
    plt.ylabel('Death Rate (%)')
    plt.xticks(rotation=45)
    
    # Plot 4: Cases Per Million
    plt.subplot(4, 2, 4)
    latest_data = latest_data.sort_values('cases_per_million', ascending=False)
    sns.barplot(x=latest_data.index, y=latest_data['cases_per_million'], palette='magma')
    plt.title('Cases Per Million Population')
    plt.xlabel('Country')
    plt.ylabel('Cases Per Million')
    plt.xticks(rotation=45)
    
    # Plot 5: Deaths Per Million
    plt.subplot(4, 2, 5)
    latest_data = latest_data.sort_values('deaths_per_million', ascending=False)
    sns.barplot(x=latest_data.index, y=latest_data['deaths_per_million'], palette='plasma')
    plt.title('Deaths Per Million Population')
    plt.xlabel('Country')
    plt.ylabel('Deaths Per Million')
    plt.xticks(rotation=45)
    
    # Plot 6: Vaccination Progress
    plt.subplot(4, 2, 6)
    for country in countries_of_interest:
        country_data = df_filtered[df_filtered['location'] == country]
        plt.plot(country_data['date'], country_data['people_fully_vaccinated_per_hundred'], label=country)
    plt.title('Percentage Fully Vaccinated Over Time')
    plt.xlabel('Date')
    plt.ylabel('Percentage Fully Vaccinated')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # Plot 7: Latest Vaccination Status
    plt.subplot(4, 2, 7)
    latest_vaccination = latest_data.sort_values('people_fully_vaccinated_per_hundred', ascending=False)
    sns.barplot(x=latest_vaccination.index, y=latest_vaccination['people_fully_vaccinated_per_hundred'], 
                palette='coolwarm')
    plt.title('Percentage Fully Vaccinated')
    plt.xlabel('Country')
    plt.ylabel('Percentage Fully Vaccinated')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('covid_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualizations saved as 'covid_analysis.png'")
    
    # 4. Choropleth Maps (will open in browser)
    print("\nGenerating interactive world maps...")
    latest_global = df.sort_values('date').groupby('location').last().reset_index()
    
    # Cases per million map
    fig_cases = px.choropleth(latest_global, 
                             locations="iso_code",
                             color="total_cases_per_million",
                             hover_name="location",
                             color_continuous_scale=px.colors.sequential.Plasma,
                             title="Total COVID-19 Cases Per Million Population")
    fig_cases.write_html('covid_cases_map.html')
    
    # Vaccination map
    fig_vacc = px.choropleth(latest_global, 
                            locations="iso_code",
                            color="people_fully_vaccinated_per_hundred",
                            hover_name="location",
                            color_continuous_scale=px.colors.sequential.Viridis,
                            title="Percentage of Population Fully Vaccinated")
    fig_vacc.write_html('covid_vaccination_map.html')
    
    print("Interactive maps saved as HTML files:")
    print("- covid_cases_map.html")
    print("- covid_vaccination_map.html")
    
    # 5. Key Insights
    print("\nKey Insights:")
    print("1. Vaccination Progress: Countries showed varying speeds in vaccination rollout.")
    print("2. Case Trends: Different countries experienced waves at different times.")
    print("3. Death Rates: Significant variation between countries (see visualization).")
    print("4. Cases Per Million: Higher in densely populated countries.")
    print("5. Global Disparities: Clear differences between developed and developing nations.")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
