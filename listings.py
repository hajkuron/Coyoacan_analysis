#%%
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time
import re

# Read the CSV file
df = pd.read_csv('/Users/jonkuhar/Desktop/Universtity/Hayuacan - Venture challenge/listings.csv')

# Clean price data
def extract_price(price_str):
    if pd.isna(price_str):
        return None
    # Extract numbers from string using regex
    numbers = re.findall(r'€?([\d,]+)', str(price_str))
    if numbers:
        # Take the first number found and remove commas
        return float(numbers[0].replace(',', ''))
    return None

# Clean size data
def extract_size(size_str):
    if pd.isna(size_str):
        return None
    numbers = re.findall(r'([\d,]+)', str(size_str))
    if numbers:
        return float(numbers[0].replace(',', ''))
    return None

# Clean the data
df['price'] = df['Yearly Rent (excl. VAT)'].apply(extract_price)
df['size'] = df['Size (m²)'].apply(extract_size)

# Filter for places under 50k before geocoding
df = df[df['price'] < 50000].copy()
print(f"Number of locations under €50,000: {len(df)}")

# Initialize the geocoder with increased timeout
geolocator = Nominatim(user_agent="my_app", timeout=10)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=2)

# Function to get coordinates from address with better error handling
def get_coordinates(address):
    if pd.isna(address):
        return None
    try:
        cleaned_address = address.replace('\n', ' ').strip()
        location = geocode(f"{cleaned_address}, Amsterdam, Netherlands")
        if location:
            return (location.latitude, location.longitude)
        return None
    except Exception as e:
        print(f"Error geocoding {address}: {str(e)}")
        time.sleep(2)
        return None

# Get coordinates for each location with progress indicator
print("Geocoding addresses... This may take a few minutes...")
total = len(df)
df['coordinates'] = None
for idx, row in df.iterrows():
    print(f"Processing {idx + 1}/{total}: {row['Location']}")
    df.at[idx, 'coordinates'] = get_coordinates(row['Location'])

# Create HTML for the filter controls
filter_html = """
<div style="position: fixed; top: 10px; right: 10px; width: 200px; 
            background-color: white; padding: 10px; border-radius: 5px; 
            box-shadow: 0 0 10px rgba(0,0,0,0.3); z-index: 1000;">
    <h4>Filters</h4>
    <label for="price-min">Min Price (€):</label><br>
    <input type="range" id="price-min" min="0" max="100000" value="0" 
           oninput="updateMarkers()"><br>
    <span id="price-min-value">0</span>€<br>
    
    <label for="price-max">Max Price (€):</label><br>
    <input type="range" id="price-max" min="0" max="100000" value="100000" 
           oninput="updateMarkers()"><br>
    <span id="price-max-value">100000</span>€<br>
    
    <label for="size-min">Min Size (m²):</label><br>
    <input type="range" id="size-min" min="0" max="1000" value="0" 
           oninput="updateMarkers()"><br>
    <span id="size-min-value">0</span>m²<br>
    
    <label for="size-max">Max Size (m²):</label><br>
    <input type="range" id="size-max" min="0" max="1000" value="1000" 
           oninput="updateMarkers()"><br>
    <span id="size-max-value">1000</span>m²
</div>

<script>
function updateMarkers() {
    var priceMin = document.getElementById('price-min').value;
    var priceMax = document.getElementById('price-max').value;
    var sizeMin = document.getElementById('size-min').value;
    var sizeMax = document.getElementById('size-max').value;
    
    document.getElementById('price-min-value').textContent = priceMin;
    document.getElementById('price-max-value').textContent = priceMax;
    document.getElementById('size-min-value').textContent = sizeMin;
    document.getElementById('size-max-value').textContent = sizeMax;
    
    var markers = document.getElementsByClassName('leaflet-marker-icon');
    for (var i = 0; i < markers.length; i++) {
        var marker = markers[i];
        var price = marker.getAttribute('data-price');
        var size = marker.getAttribute('data-size');
        
        if (price >= priceMin && price <= priceMax && 
            size >= sizeMin && size <= sizeMax) {
            marker.style.display = 'block';
        } else {
            marker.style.display = 'none';
        }
    }
}
</script>
"""

# Create a map centered on Amsterdam
m = folium.Map(location=[52.3676, 4.9041], zoom_start=12)

# Add the filter controls to the map
m.get_root().html.add_child(folium.Element(filter_html))

# Add markers for each listing
for idx, row in df.iterrows():
    if row['coordinates']:
        lat, lon = row['coordinates']
        folium.Marker(
            location=[lat, lon],
            popup=f"""
                <b>{row['Name']}</b><br>
                Location: {row['Location']}<br>
                Yearly Rent: {'€{:,.0f}'.format(row['price']) if pd.notnull(row['price']) else 'Not specified'}<br>
                Size: {'{:,.0f}'.format(row['size']) if pd.notnull(row['size']) else 'Not specified'} m²
            """,
            tooltip=row['Name'],
            icon=folium.Icon(color='red'),
            html=f'<div data-price="{row["price"] or 0}" data-size="{row["size"] or 0}"></div>'
        ).add_to(m)

# Save the map
m.save('amsterdam_listings_map.html')

print("Map has been created with price and size filters!")

# %%
