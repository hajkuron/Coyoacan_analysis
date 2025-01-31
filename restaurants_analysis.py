#%%
import pandas as pd
import matplotlib.pyplot as plt
import folium
import json
from folium.plugins import HeatMap
import glob
import numpy as np
from scipy.stats import gaussian_kde
import seaborn as sns
from scipy.spatial.distance import cdist
import math
from scipy import stats

def extract_coordinates(coord_str):
    try:
        coord_str = coord_str.replace('""', '"')
        coords = json.loads(coord_str)
        return pd.Series([coords['latitude'], coords['longitude']])
    except:
        return pd.Series([None, None])

def explore_location_data():
    print("Loading data from multiple files...")
    # Create a list of all CSV files in the data directory
    csv_files = glob.glob('data/*.csv')
    
    # Read and combine all CSV files
    df_list = []
    for file in csv_files:
        try:
            temp_df = pd.read_csv(file)
            df_list.append(temp_df)
            print(f"Loaded: {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Combine all dataframes
    df = pd.concat(df_list, ignore_index=True)
    
    # Remove duplicates based on name and address
    initial_count = len(df)
    df = df.drop_duplicates(subset=['name', 'address'], keep='first')
    duplicates_removed = initial_count - len(df)
    print(f"\nRemoved {duplicates_removed} duplicate entries")
    print(f"Total unique records: {len(df)}")
    
    print("Extracting coordinates...")
    df[['latitude', 'longitude']] = df['coordinates'].apply(extract_coordinates)
    
    # Check for Mexican indicators across multiple columns
    mexican_keywords = (
        'mexican|taco|burrito|tex-mex|cantina|'
        'guadalajara|tijuana|cancun|monterrey|'
        'enchilada|quesadilla|guacamole|salsa|'
        'tortilla|fajita|chipotle'
    )
    
    df['is_mexican'] = (
        df['main_category'].str.lower().str.contains('mexican', na=False) |
        df['name'].str.lower().str.contains(mexican_keywords, na=False) |
        df['categories'].str.lower().str.contains(mexican_keywords, na=False) |
        df['description'].str.lower().str.contains(mexican_keywords, na=False)
    )
    
    # Display relevant columns
    print("\nRestaurant Details:")
    print(df[['name', 'main_category', 'categories', 'description', 'is_mexican', 'rating', 'address']].to_string())

    # Create interactive folium map
    print("Creating interactive map...")
    amsterdam_map = folium.Map(
        location=[52.3676, 4.9041],  # Amsterdam center
        zoom_start=13,
        tiles='cartodbpositron'  # Clean, modern map style
    )

    # Add markers for each restaurant
    for idx, row in df.iterrows():
        if pd.notnull(row['latitude']) and pd.notnull(row['longitude']):
            # Color code: orange for Mexican, blue for others
            color = 'orange' if row['is_mexican'] else 'blue'
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=8,
                popup=folium.Popup(
                    f"""
                    <b>{row['name']}</b><br>
                    {'🌮 Mexican Restaurant<br>' if row['is_mexican'] else ''}
                    Rating: {row['rating']}<br>
                    Address: {row['address']}<br>
                    Price Range: {row['price_range']}<br>
                    """,
                    max_width=300
                ),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                weight=2
            ).add_to(amsterdam_map)

    # Add heatmap layer
    heat_data = [[row['latitude'], row['longitude']] for idx, row in df.iterrows() 
                 if pd.notnull(row['latitude']) and pd.notnull(row['longitude'])]
    HeatMap(heat_data).add_to(amsterdam_map)

    # Add a legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 150px; height: 90px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white;
                padding: 10px;
                border-radius: 5px;
                ">
      <p><span style="color:orange;">●</span> Mexican Restaurants</p>
      <p><span style="color:blue;">●</span> Other Restaurants</p>
    </div>
    '''
    amsterdam_map.get_root().html.add_child(folium.Element(legend_html))

    # Save interactive map
    amsterdam_map.save('restaurants_map.html')

    # Print summary
    mexican_count = df['is_mexican'].sum()
    total_count = len(df)
    print(f"\nFound {mexican_count} Mexican restaurants out of {total_count} total restaurants")
    
    # Save the full dataset to CSV
    df.to_csv('all_restaurants_data.csv', index=False)
    print("Saved complete restaurant dataset to 'all_restaurants_data.csv'")
    
    return df

df = explore_location_data()
#%%
# Basic overview of the data
print("Dataset Overview:")
print(f"Total restaurants: {len(df)}")
print(f"Mexican restaurants: {df['is_mexican'].sum()}")
print("\nRating Statistics:")
print(df.groupby('is_mexican')['rating'].describe())

#%%
# Basic statistics by area
print("Rating Analysis by Location:")
# Create location bins based on latitude and longitude
df['lat_bin'] = pd.qcut(df['latitude'], q=4, labels=['South', 'Central-South', 'Central-North', 'North'])
df['long_bin'] = pd.qcut(df['longitude'], q=4, labels=['West', 'Central-West', 'Central-East', 'East'])

# Calculate average ratings by area
area_ratings = df.groupby(['lat_bin', 'long_bin'])['rating'].agg(['mean', 'count']).round(2)
print("\nAverage ratings by area:")
print(area_ratings)

#%%
# Create a heatmap of average ratings by area
plt.figure(figsize=(12, 8))
ratings_pivot = df.pivot_table(
    values='rating',
    index='lat_bin',
    columns='long_bin',
    aggfunc='mean'
)
sns.heatmap(ratings_pivot, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Average Restaurant Ratings by Area')
plt.show()

#%%
# Analyze Mexican restaurant competition
# Calculate distance to nearest Mexican restaurant for each location
mexican_coords = df[df['is_mexican']][['latitude', 'longitude']].values
all_coords = df[['latitude', 'longitude']].values

# Calculate minimum distances to Mexican restaurants
distances = cdist(all_coords, mexican_coords)
df['dist_to_nearest_mexican'] = distances.min(axis=1)

# Plot average ratings vs distance to nearest Mexican restaurant
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='dist_to_nearest_mexican', y='rating', alpha=0.5)
plt.title('Restaurant Ratings vs Distance to Nearest Mexican Restaurant')
plt.xlabel('Distance to Nearest Mexican Restaurant (degrees)')
plt.ylabel('Rating')
plt.show()

#%%
# Analyze high-performing areas
high_rated = df[df['rating'] >= 4.5]
print("\nHigh-rated Restaurant Analysis:")
print(f"Number of high-rated restaurants (4.5+): {len(high_rated)}")
print(f"Number of high-rated Mexican restaurants: {len(high_rated[high_rated['is_mexican']])}")

# Calculate density of high-rated restaurants by area
high_rated_density = df[df['rating'] >= 4.5].groupby(['lat_bin', 'long_bin']).size()
print("\nNumber of high-rated restaurants by area:")
print(high_rated_density)

#%%
# Create a bubble plot showing restaurant density and ratings
plt.figure(figsize=(12, 8))
for is_mex in [False, True]:
    subset = df[df['is_mexican'] == is_mex]
    plt.scatter(
        subset['longitude'],
        subset['latitude'],
        s=subset['rating'] * 50,  # Size based on rating
        alpha=0.5,
        c='orange' if is_mex else 'blue',
        label=f"{'Mexican' if is_mex else 'Other'} Restaurants"
    )

plt.title('Restaurant Locations (Size = Rating)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
#%%

# Create a choropleth-style map showing average ratings by area
rating_map = folium.Map(
    location=[52.3676, 4.9041],  # Amsterdam center
    zoom_start=12,
    tiles='cartodbpositron'
)

# Create different colored circles based on rating ranges
for idx, row in df.iterrows():
    if pd.notnull(row['latitude']) and pd.notnull(row['longitude']):
        # Color code based on rating
        if row['rating'] >= 4.5:
            color = 'darkgreen'
        elif row['rating'] >= 4.0:
            color = 'green'
        elif row['rating'] >= 3.5:
            color = 'orange'
        else:
            color = 'red'
            
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            popup=folium.Popup(
                f"""
                <b>{row['name']}</b><br>
                {'🌮 Mexican Restaurant<br>' if row['is_mexican'] else ''}
                Rating: {row['rating']}<br>
                Address: {row['address']}<br>
                """,
                max_width=300
            ),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7
        ).add_to(rating_map)

# Add a legend for ratings
legend_html = '''
<div style="position: fixed; 
            bottom: 50px; right: 50px; width: 180px;
            border:2px solid grey; z-index:9999; font-size:14px;
            background-color:white;
            padding: 10px;
            border-radius: 5px;">
    <p><span style="color:darkgreen;">●</span> Rating ≥ 4.5</p>
    <p><span style="color:green;">●</span> Rating 4.0-4.4</p>
    <p><span style="color:orange;">●</span> Rating 3.5-3.9</p>
    <p><span style="color:red;">●</span> Rating < 3.5</p>
</div>
'''
rating_map.get_root().html.add_child(folium.Element(legend_html))
rating_map.save('restaurant_ratings_map.html')

#%%
# Create a map showing Mexican restaurant density
density_map = folium.Map(
    location=[52.3676, 4.9041],
    zoom_start=12,
    tiles='cartodbpositron'
)

# Add heatmap layer for Mexican restaurants only
mexican_heat_data = [[row['latitude'], row['longitude']] for idx, row in mexican_restaurants.iterrows() 
                     if pd.notnull(row['latitude']) and pd.notnull(row['longitude'])]
HeatMap(mexican_heat_data, radius=15).add_to(density_map)

# Add markers for high-rated Mexican restaurants (rating >= 4.5)
for idx, row in mexican_restaurants[mexican_restaurants['rating'] >= 4.5].iterrows():
    if pd.notnull(row['latitude']) and pd.notnull(row['longitude']):
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            popup=folium.Popup(
                f"""
                <b>{row['name']}</b><br>
                Rating: {row['rating']}<br>
                Address: {row['address']}<br>
                """,
                max_width=300
            ),
            color='gold',
            fill=True,
            fill_color='gold',
            fill_opacity=0.7
        ).add_to(density_map)

# Add legend
legend_html = '''
<div style="position: fixed; 
            bottom: 50px; right: 50px; width: 200px;
            border:2px solid grey; z-index:9999; font-size:14px;
            background-color:white;
            padding: 10px;
            border-radius: 5px;">
    <p>Heat Map: Mexican Restaurant Density</p>
    <p><span style="color:gold;">●</span> High-rated Mexican (≥4.5)</p>
</div>
'''
density_map.get_root().html.add_child(folium.Element(legend_html))
density_map.save('mexican_density_map.html')

#%%
def create_complete_popularity_data(json_string):
    """
    Parse popularity data with better handling of closed/unknown hours.
    Returns a DataFrame with all hours (0-23) for each day, using:
    - actual popularity values when available
    - NaN for hours with no data (likely closed/unknown)
    """
    try:
        # Create template for all days and hours
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        template = []
        for day in days:
            for hour in range(24):
                template.append({
                    'day': day,
                    'hour': hour,
                    'time': f"{hour:02d}:00",
                    'popularity': np.nan,  # Use NaN instead of 0
                    'description': 'Unknown'
                })
        
        # Create template DataFrame
        complete_df = pd.DataFrame(template)
        
        # If we have actual data, parse and update the template
        if json_string != "Not Present" and pd.notna(json_string):
            data = json.loads(json_string)
            
            # Update template with actual values
            for day, day_data in data.items():
                for hour_data in day_data:
                    hour = hour_data['hour_of_day']
                    mask = (complete_df['day'] == day) & (complete_df['hour'] == hour)
                    complete_df.loc[mask, 'popularity'] = hour_data['popularity_percentage']
                    complete_df.loc[mask, 'description'] = hour_data['popularity_description']
        
        # Sort by day and hour
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        complete_df['day'] = pd.Categorical(complete_df['day'], categories=day_order, ordered=True)
        complete_df = complete_df.sort_values(['day', 'hour']).reset_index(drop=True)
        
        return complete_df
    except Exception as e:
        print(f"Error parsing data: {e}")
        return None

def analyze_restaurant_popularity(restaurants_df):
    """
    Analyzes restaurant popularity with improved handling of operating hours.
    Shows popularity patterns and data confidence across different days/times.
    
    Args:
        restaurants_df: DataFrame containing restaurant data with 'popular_times' column
    """
    # Filter restaurants with hours data
    restaurants_with_hours = restaurants_df[
        (restaurants_df['popular_times'] != "Not Present") & 
        (restaurants_df['popular_times'].notna())
    ].copy()
    
    print(f"Number of restaurants with hours data: {len(restaurants_with_hours)}")

    # Collect popularity data for each restaurant
    all_popularity_data = []
    for _, restaurant in restaurants_with_hours.iterrows():
        pop_df = create_complete_popularity_data(restaurant['popular_times'])
        if pop_df is not None:
            # Add restaurant info to help with analysis
            pop_df['restaurant_name'] = restaurant['name']
            pop_df['rating'] = restaurant['rating']
            all_popularity_data.append(pop_df)

    print(f"Successfully parsed popularity data for {len(all_popularity_data)} restaurants")

    if not all_popularity_data:
        print("No popularity data available for analysis")
        return None

    # Combine all restaurant data
    combined_popularity = pd.concat(all_popularity_data)

    # Calculate statistics by day and hour
    stats = combined_popularity.groupby(['day', 'hour']).agg({
        'popularity': ['mean', 'count', 'std'],
        'restaurant_name': 'nunique'
    }).reset_index()
    
    # Rename columns for clarity
    stats.columns = ['day', 'hour', 'avg_popularity', 'data_points', 'std_popularity', 'unique_restaurants']
    
    # Calculate confidence (what percentage of restaurants have data for this time)
    total_restaurants = len(all_popularity_data)
    stats['confidence'] = stats['unique_restaurants'] / total_restaurants

    # Create visualization with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Color scheme for days
    colors = plt.cm.viridis(np.linspace(0, 1, 7))
    
    # Plot 1: Average Popularity with Standard Deviation
    for day, color in zip(stats['day'].unique(), colors):
        day_data = stats[stats['day'] == day]
        
        # Plot mean line
        line = ax1.plot(
            day_data['hour'],
            day_data['avg_popularity'],
            label=day,
            color=color,
            marker='o',
            markersize=4
        )
        
        # Add standard deviation shading
        ax1.fill_between(
            day_data['hour'],
            day_data['avg_popularity'] - day_data['std_popularity'],
            day_data['avg_popularity'] + day_data['std_popularity'],
            color=color,
            alpha=0.1
        )

    ax1.grid(True, alpha=0.3)
    ax1.set_title('Average Restaurant Popularity Throughout the Week\n(with ±1 standard deviation bands)', pad=10)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Average Popularity (%)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot 2: Data Confidence
    for day, color in zip(stats['day'].unique(), colors):
        day_data = stats[stats['day'] == day]
        ax2.plot(
            day_data['hour'],
            day_data['confidence'] * 100,
            label=day,
            color=color,
            marker='o',
            markersize=4
        )
        
        # Add light horizontal lines at 25%, 50%, 75%
        for conf in [25, 50, 75]:
            ax2.axhline(y=conf, color='gray', linestyle='--', alpha=0.3)

    ax2.grid(True, alpha=0.3)
    ax2.set_title('Data Confidence Level\n(percentage of restaurants reporting data for each hour)', pad=10)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Confidence (%)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Format both plots
    for ax in [ax1, ax2]:
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{i:02d}:00" for i in range(24)], rotation=45)
        ax.set_xlim(-0.5, 23.5)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Print analysis of peak hours
    print("\nPeak Hours Analysis (>30% popularity with >50% confidence):")
    for day in stats['day'].unique():
        peak_data = stats[
            (stats['day'] == day) & 
            (stats['avg_popularity'] >= 30) &
            (stats['confidence'] >= 0.5)
        ]
        if not peak_data.empty:
            print(f"\n{day}:")
            for _, row in peak_data.iterrows():
                print(
                    f"{row['hour']:02d}:00 - "
                    f"Popularity: {row['avg_popularity']:.1f}% "
                    f"(±{row['std_popularity']:.1f}%) "
                    f"Confidence: {row['confidence']*100:.1f}%"
                )

    return stats

# Usage example:
# high_rated_mexican = df[df['is_mexican'] & (df['rating'] >= 4.0)]
# popularity_stats = analyze_restaurant_popularity(high_rated_mexican)

#%%
# Get Mexican restaurants and create complete popularity data for each
mexican_restaurants = df[df['is_mexican']].copy()
print(f"Total Mexican restaurants: {len(mexican_restaurants)}")

# Collect all popularity data with complete hours
all_popularity_data = []
for _, restaurant in mexican_restaurants.iterrows():
    pop_df = create_complete_popularity_data(restaurant['popular_times'])
    if pop_df is not None:
        all_popularity_data.append(pop_df)

print(f"Restaurants with parsed popularity data: {len(all_popularity_data)}")

#%%
# Combine all data
combined_popularity = pd.concat(all_popularity_data)

# Calculate average popularity by day and hour
avg_popularity = combined_popularity.groupby(['day', 'hour'])['popularity'].mean().reset_index()

#%%
# Create line plot
plt.figure(figsize=(15, 8))

# Plot each day's data
for day in avg_popularity['day'].unique():
    day_data = avg_popularity[avg_popularity['day'] == day]
    plt.plot(
        day_data['hour'],
        day_data['popularity'],
        label=day,
        marker='o'
    )

plt.grid(True, alpha=0.3)
plt.title('Average Popularity of Mexican Restaurants Throughout the Week')
plt.xlabel('Hour of Day (24-hour format)')
plt.ylabel('Average Popularity (%)')

# Set x-axis ticks for every hour
plt.xticks(
    range(24),
    [f"{i:02d}:00" for i in range(24)],
    rotation=45
)

plt.legend()
plt.tight_layout()
plt.show()

#%%
# Print peak hours by day
print("\nPeak Hours by Day:")
for day in avg_popularity['day'].unique():
    day_data = avg_popularity[avg_popularity['day'] == day]
    peak_hours = day_data[day_data['popularity'] >= 30]  # Lowered threshold to 30% given we now include all hours
    if not peak_hours.empty:
        print(f"\n{day}:")
        for _, row in peak_hours.iterrows():
            print(f"{row['hour']:02d}:00 - {row['popularity']:.1f}% popularity")

#%%
# After combining all data, save to CSV
combined_popularity.to_csv('mexican_restaurants_hourly_popularity.csv', index=False)
print("Saved detailed popularity data to 'mexican_restaurants_hourly_popularity.csv'")

# Save the averaged data as well
avg_popularity.to_csv('mexican_restaurants_avg_popularity.csv', index=False)
print("Saved average popularity data to 'mexican_restaurants_avg_popularity.csv'")

#%%
# Location Analysis: Distribution and Success Patterns
print("Analyzing geographical distribution and success patterns...")

# First, let's create location categories based on distance from city center
amsterdam_center = (52.3676, 4.9041)  # Amsterdam centrum coordinates

def calculate_distance_from_center(row):
    """Calculate distance from city center in kilometers"""
    if pd.isna(row['latitude']) or pd.isna(row['longitude']):
        return None
    
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth's radius in km
    lat1, lon1 = map(radians, amsterdam_center)
    lat2, lon2 = map(radians, [row['latitude'], row['longitude']])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance

# Add distance from center and categorize locations
df['distance_from_center'] = df.apply(calculate_distance_from_center, axis=1)
df['location_category'] = pd.cut(
    df['distance_from_center'],
    bins=[0, 1, 2, 3, float('inf')],
    labels=['City Center', 'Inner Ring', 'Outer Ring', 'Suburbs']
)

#%%
# Create visualization of restaurant distribution with location categories
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    df['longitude'], 
    df['latitude'],
    c=df['rating'],
    s=100 * df['is_mexican'].astype(int) + 50,  # Mexican restaurants are larger
    alpha=0.6,
    cmap='viridis'
)

# Add city center marker
plt.plot(amsterdam_center[1], amsterdam_center[0], 'r*', markersize=15, label='City Center')

# Add concentric circles for distance categories
for radius in [1, 2, 3]:  # km
    circle = plt.Circle(
        amsterdam_center[::-1], 
        radius/111,  # Convert km to degrees (approximately)
        fill=False,
        linestyle='--',
        color='gray',
        alpha=0.5
    )
    plt.gca().add_artist(circle)

plt.colorbar(scatter, label='Rating')
plt.title('Restaurant Distribution by Location and Rating\n(Larger markers = Mexican restaurants)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

#%%
# Analyze success metrics by location category
location_analysis = df.groupby('location_category').agg({
    'rating': ['mean', 'count', 'std'],
    'is_mexican': 'sum'
}).round(2)

location_analysis.columns = ['avg_rating', 'total_restaurants', 'rating_std', 'mexican_restaurants']
location_analysis['mexican_percentage'] = (
    location_analysis['mexican_restaurants'] / location_analysis['total_restaurants'] * 100
).round(2)

print("\nLocation Category Analysis:")
print(location_analysis)

#%%
# Visualize ratings distribution by location category
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='location_category', y='rating', hue='is_mexican')
plt.title('Restaurant Ratings Distribution by Location')
plt.xticks(rotation=45)
plt.legend(title='Mexican Restaurant', labels=['No', 'Yes'])
plt.show()

#%%
# Statistical analysis of location impact
from scipy import stats

# Perform ANOVA test for ratings across location categories
mexican_restaurants = df[df['is_mexican']]
f_stat, p_value = stats.f_oneway(
    *[group['rating'].dropna() for name, group in mexican_restaurants.groupby('location_category')]
)

print("\nStatistical Analysis for Mexican Restaurants:")
print(f"ANOVA test p-value: {p_value:.4f}")
if p_value < 0.05:
    print("There is a significant difference in ratings across location categories")
else:
    print("No significant difference in ratings across location categories")

#%%
# Correlation analysis
correlation = df[['rating', 'distance_from_center']].corr()
print("\nCorrelation between rating and distance from center:")
print(correlation)

# Visualize the correlation
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='distance_from_center',
    y='rating',
    hue='is_mexican',
    alpha=0.6
)
plt.title('Rating vs Distance from City Center')
plt.xlabel('Distance from Center (km)')
plt.ylabel('Rating')
plt.legend(title='Mexican Restaurant', labels=['No', 'Yes'])
plt.show()
