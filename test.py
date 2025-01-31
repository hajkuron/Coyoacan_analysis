#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import ast
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Define color scheme
main_color = '#303C68'
lighter_color = '#4C5F99'

def parse_time(time_str):
    if pd.isna(time_str):
        return None
    
    # Remove spaces and convert to lowercase
    time_str = time_str.lower().replace(' ', '')
    
    # Handle different formats
    if 'am' in time_str or 'pm' in time_str:
        # Handle cases like "9am", "12pm", etc.
        try:
            return datetime.strptime(time_str, '%I%p').hour
        except:
            try:
                # Handle cases like "9:30am", "12:30pm", etc.
                return datetime.strptime(time_str, '%I:%M%p').hour
            except:
                return None
    else:
        # Handle 24-hour format
        try:
            return int(time_str)
        except:
            return None

def parse_timing(timing_str):
    if pd.isna(timing_str):
        return None, None
    
    try:
        # Split into opening and closing times
        times = timing_str.split('-')
        opening = parse_time(times[0])
        closing = parse_time(times[1])
        
        # Adjust closing time if it's PM and less than opening time
        if closing is not None and closing < opening:
            closing += 12
            
        return opening, closing
    except:
        return None, None

def analyze_restaurant_data(original_df, is_mexican=False):
    # Create a copy of the dataframe to avoid warnings
    df = original_df.copy()
    
    # Process closed days
    df.loc[:, 'closed_days_count'] = df['closed_on'].apply(
        lambda x: len(ast.literal_eval(x)) if pd.notna(x) and x != 'Open All Days' else 0
    )
    
    # Create success metrics
    df.loc[:, 'success_score'] = df['rating'] * np.log1p(df['reviews'])
    
    # Prepare data for analysis
    X = df['closed_days_count'].values.reshape(-1, 1)
    y_rating = df['rating'].values
    y_reviews = df['reviews'].values
    y_success = df['success_score'].values
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('white')
    title_prefix = "Mexican" if is_mexican else "All"
    
    # 1. Closed Days vs Rating
    ax1.scatter(X, y_rating, alpha=0.5, color=main_color)
    ax1.set_xlabel('Number of Closed Days')
    ax1.set_ylabel('Rating')
    ax1.set_title(f'{title_prefix} Restaurants: Closed Days vs Rating', color=main_color, pad=20)
    
    # Add regression line
    z1 = np.polyfit(X.flatten(), y_rating, 1)
    p1 = np.poly1d(z1)
    ax1.plot(X.flatten(), p1(X.flatten()), color=lighter_color)
    
    # Calculate correlation
    corr_rating = stats.pearsonr(X.flatten(), y_rating)
    ax1.text(0.05, 0.95, f'Correlation: {corr_rating[0]:.3f}\np-value: {corr_rating[1]:.3f}', 
             transform=ax1.transAxes, verticalalignment='top')
    
    # 2. Closed Days vs Review Count
    ax2.scatter(X, np.log1p(y_reviews), alpha=0.5, color=main_color)
    ax2.set_xlabel('Number of Closed Days')
    ax2.set_ylabel('Log(Review Count + 1)')
    ax2.set_title(f'{title_prefix} Restaurants: Closed Days vs Review Count', color=main_color, pad=20)
    
    # Add regression line
    z2 = np.polyfit(X.flatten(), np.log1p(y_reviews), 1)
    p2 = np.poly1d(z2)
    ax2.plot(X.flatten(), p2(X.flatten()), color=lighter_color)
    
    # Calculate correlation
    corr_reviews = stats.pearsonr(X.flatten(), np.log1p(y_reviews))
    ax2.text(0.05, 0.95, f'Correlation: {corr_reviews[0]:.3f}\np-value: {corr_reviews[1]:.3f}', 
             transform=ax2.transAxes, verticalalignment='top')
    
    # 3. Closed Days vs Combined Success Score
    ax3.scatter(X, y_success, alpha=0.5, color=main_color)
    ax3.set_xlabel('Number of Closed Days')
    ax3.set_ylabel('Success Score (Rating × log(Reviews))')
    ax3.set_title(f'{title_prefix} Restaurants: Closed Days vs Success Score', color=main_color, pad=20)
    
    # Add regression line
    z3 = np.polyfit(X.flatten(), y_success, 1)
    p3 = np.poly1d(z3)
    ax3.plot(X.flatten(), p3(X.flatten()), color=lighter_color)
    
    # Calculate correlation
    corr_success = stats.pearsonr(X.flatten(), y_success)
    ax3.text(0.05, 0.95, f'Correlation: {corr_success[0]:.3f}\np-value: {corr_success[1]:.3f}', 
             transform=ax3.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print(f"\nDetailed Statistical Analysis for {title_prefix} Restaurants:")
    print("\n1. Rating Analysis:")
    print(f"Correlation coefficient: {corr_rating[0]:.3f}")
    print(f"P-value: {corr_rating[1]:.3f}")
    
    print("\n2. Review Count Analysis:")
    print(f"Correlation coefficient: {corr_reviews[0]:.3f}")
    print(f"P-value: {corr_reviews[1]:.3f}")
    
    print("\n3. Success Score Analysis:")
    print(f"Correlation coefficient: {corr_success[0]:.3f}")
    print(f"P-value: {corr_success[1]:.3f}")
    
    # Calculate average metrics by number of closed days
    avg_metrics = df.groupby('closed_days_count').agg({
        'rating': 'mean',
        'reviews': 'mean',
        'success_score': 'mean',
        'place_id': 'count'  # Count of restaurants
    }).round(2)
    
    print(f"\nAverage Metrics by Number of Closed Days for {title_prefix} Restaurants:")
    print(avg_metrics)

def analyze_closed_days_impact(original_df, is_mexican=False):
    # Create a copy of the dataframe
    df = original_df.copy()
    
    # Create columns for each day
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Parse closed_on and create binary columns for each day
    for day in days:
        df[f'closed_{day}'] = df['closed_on'].apply(
            lambda x: 1 if pd.notna(x) and x != 'Open All Days' and day in ast.literal_eval(x) else 0
        )
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    fig.patch.set_facecolor('white')
    title_prefix = "Mexican" if is_mexican else "All"
    
    # Prepare data for plotting
    day_stats = []
    for day in days:
        # Calculate average rating for restaurants closed vs open on this day
        closed_rating = df[df[f'closed_{day}'] == 1]['rating'].mean()
        open_rating = df[df[f'closed_{day}'] == 0]['rating'].mean()
        
        # Calculate average review count for restaurants closed vs open on this day
        closed_reviews = df[df[f'closed_{day}'] == 1]['reviews'].mean()
        open_reviews = df[df[f'closed_{day}'] == 0]['reviews'].mean()
        
        # Perform t-test for ratings
        t_stat_rating, p_val_rating = stats.ttest_ind(
            df[df[f'closed_{day}'] == 1]['rating'],
            df[df[f'closed_{day}'] == 0]['rating']
        )
        
        day_stats.append({
            'day': day,
            'closed_rating': closed_rating,
            'open_rating': open_rating,
            'closed_reviews': closed_reviews,
            'open_reviews': open_reviews,
            'rating_diff': closed_rating - open_rating,
            'reviews_diff': closed_reviews - open_reviews,
            'p_value_rating': p_val_rating
        })
    
    # Convert to DataFrame for easier plotting
    stats_df = pd.DataFrame(day_stats)
    
    # Plot 1: Rating Difference
    bars1 = ax1.bar(days, stats_df['rating_diff'], color=main_color, alpha=0.7)
    ax1.set_title(f'{title_prefix} Restaurants: Impact of Closing Day on Rating\n(Difference in Average Rating: Closed - Open)', 
                  color=main_color, pad=20)
    ax1.set_xlabel('Closed Day')
    ax1.set_ylabel('Rating Difference')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add significance stars
    for idx, p_val in enumerate(stats_df['p_value_rating']):
        if p_val < 0.05:
            ax1.text(idx, stats_df['rating_diff'][idx], '*', 
                    ha='center', va='bottom' if stats_df['rating_diff'][idx] >= 0 else 'top')
    
    # Plot 2: Review Count Difference
    bars2 = ax2.bar(days, stats_df['reviews_diff'], color=lighter_color, alpha=0.7)
    ax2.set_title(f'{title_prefix} Restaurants: Impact of Closing Day on Review Count\n(Difference in Average Reviews: Closed - Open)', 
                  color=main_color, pad=20)
    ax2.set_xlabel('Closed Day')
    ax2.set_ylabel('Review Count Difference')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print(f"\nDetailed Statistics for {title_prefix} Restaurants:")
    print("\nAverage Metrics by Day:")
    for day in days:
        closed_count = df[f'closed_{day}'].sum()
        open_count = len(df) - closed_count
        print(f"\n{day}:")
        print(f"Restaurants closed: {closed_count}")
        print(f"Average rating when closed: {stats_df[stats_df['day'] == day]['closed_rating'].values[0]:.2f}")
        print(f"Average rating when open: {stats_df[stats_df['day'] == day]['open_rating'].values[0]:.2f}")
        print(f"Average reviews when closed: {stats_df[stats_df['day'] == day]['closed_reviews'].values[0]:.0f}")
        print(f"Average reviews when open: {stats_df[stats_df['day'] == day]['open_reviews'].values[0]:.0f}")
        print(f"P-value for rating difference: {stats_df[stats_df['day'] == day]['p_value_rating'].values[0]:.3f}")

# Read the CSV file
df = pd.read_csv('all_restaurants_data.csv')

# Analyze all restaurants
print("\n=== Analysis for All Restaurants ===")
analyze_restaurant_data(df)

# Analyze only Mexican restaurants
print("\n=== Analysis for Mexican Restaurants ===")
mexican_df = df[df['is_mexican']]
analyze_restaurant_data(mexican_df, is_mexican=True)

# Analyze closed days impact
analyze_closed_days_impact(df)

# Analyze only Mexican restaurants closed days impact
analyze_closed_days_impact(mexican_df, is_mexican=True)