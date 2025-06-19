"""
Cost of Living Comparison App

This Shiny app provides interactive visualizations for comparing
cost of living across different countries. It includes:
- Historical trend analysis
- Country comparisons
- Global map visualization
- Inflation trends
- Detailed data tables
- Radar chart comparing cost of living indicators

Required data files:
- Housing-related-expenditure-of-households.csv
- Cost_of_living.csv
- Monthly_salary.csv
- CPI_2.csv
- VAT.csv
- Crude_oil.csv (Brent Crude)
- Oil_prices.csv (Multiple Benchmarks)
- Currency.csv (Exchange rates)
"""

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle # Not actively used in the final plots
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots # Not actively used
import requests # Not actively used
from typing import Dict, Optional, List
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('agg') 
import os
import traceback
from io import StringIO
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from pycountry import countries
#from fastapi.staticfiles import StaticFiles

# --- Configuration ---
DATA_FOLDER = "cost of living" 
CURRENT_YEAR = datetime.now().year
YEARS = list(range(2015, CURRENT_YEAR + 1))

# Country coordinates data (expanded for better map visualization)
COUNTRY_DATA = {
    'USA': {'lat': 37.0902, 'lon': -95.7129, 'iso_alpha': 'USA'},
    'United Kingdom': {'lat': 55.3781, 'lon': -3.4360, 'iso_alpha': 'GBR'},
    'France': {'lat': 46.2276, 'lon': 2.2137, 'iso_alpha': 'FRA'},
    'Germany': {'lat': 51.1657, 'lon': 10.4515, 'iso_alpha': 'DEU'},
    'Canada': {'lat': 56.1304, 'lon': -106.3468, 'iso_alpha': 'CAN'},
    'Australia': {'lat': -25.2744, 'lon': 133.7751, 'iso_alpha': 'AUS'},
    'Japan': {'lat': 36.2048, 'lon': 138.2529, 'iso_alpha': 'JPN'},
    'China': {'lat': 35.8617, 'lon': 104.1954, 'iso_alpha': 'CHN'},
    'India': {'lat': 20.5937, 'lon': 78.9629, 'iso_alpha': 'IND'},
    'Brazil': {'lat': -14.2350, 'lon': -51.9253, 'iso_alpha': 'BRA'},
    'South Africa': {'lat': -30.5595, 'lon': 22.9375, 'iso_alpha': 'ZAF'},
    'Mexico': {'lat': 23.6345, 'lon': -102.5528, 'iso_alpha': 'MEX'},
    'Italy': {'lat': 41.8719, 'lon': 12.5674, 'iso_alpha': 'ITA'},
    'Spain': {'lat': 40.4637, 'lon': -3.7492, 'iso_alpha': 'ESP'},
    'Argentina': {'lat': -38.4161, 'lon': -63.6167, 'iso_alpha': 'ARG'},
    'Egypt': {'lat': 26.8206, 'lon': 30.8025, 'iso_alpha': 'EGY'},
    'Indonesia': {'lat': -0.7893, 'lon': 113.9213, 'iso_alpha': 'IDN'},
    'Turkey': {'lat': 38.9637, 'lon': 35.2433, 'iso_alpha': 'TUR'},
    'Saudi Arabia': {'lat': 23.8859, 'lon': 45.0792, 'iso_alpha': 'SAU'},
    'Russia': {'lat': 61.5240, 'lon': 105.3188, 'iso_alpha': 'RUS'},
    'South Korea': {'lat': 35.9078, 'lon': 127.7669, 'iso_alpha': 'KOR'},
    'Sweden': {'lat': 60.1282, 'lon': 18.6435, 'iso_alpha': 'SWE'},
    'Norway': {'lat': 60.4720, 'lon': 8.4689, 'iso_alpha': 'NOR'},
    'Finland': {'lat': 61.9241, 'lon': 25.7482, 'iso_alpha': 'FIN'},
    'Denmark': {'lat': 56.2639, 'lon': 9.5018, 'iso_alpha': 'DNK'},
    'Netherlands': {'lat': 52.1326, 'lon': 5.2913, 'iso_alpha': 'NLD'},
    'Belgium': {'lat': 50.8333, 'lon': 4.0000, 'iso_alpha': 'BEL'},
    'Switzerland': {'lat': 46.8182, 'lon': 8.2275, 'iso_alpha': 'CHE'},
    'Austria': {'lat': 47.5162, 'lon': 14.5501, 'iso_alpha': 'AUT'},
    'Poland': {'lat': 51.9194, 'lon': 19.1451, 'iso_alpha': 'POL'},
    'Ireland': {'lat': 53.4129, 'lon': -8.2439, 'iso_alpha': 'IRL'},
    'Greece': {'lat': 39.0742, 'lon': 21.8243, 'iso_alpha': 'GRC'},
    'Portugal': {'lat': 39.3999, 'lon': -8.2245, 'iso_alpha': 'PRT'},
    'New Zealand': {'lat': -40.9006, 'lon': 174.8860, 'iso_alpha': 'NZL'},
    'Singapore': {'lat': 1.3521, 'lon': 103.8198, 'iso_alpha': 'SGP'},
    'United Arab Emirates': {'lat': 23.4241, 'lon': 53.8478, 'iso_alpha': 'ARE'},
    'Qatar': {'lat': 25.3548, 'lon': 51.1839, 'iso_alpha': 'QAT'},
    'Kuwait': {'lat': 29.3117, 'lon': 47.4818, 'iso_alpha': 'KWT'},
    'Israel': {'lat': 31.0461, 'lon': 34.8516, 'iso_alpha': 'ISR'},
    'Colombia': {'lat': 4.5709, 'lon': -74.2973, 'iso_alpha': 'COL'},
    'Peru': {'lat': -9.1900, 'lon': -75.0152, 'iso_alpha': 'PER'},
    'Chile': {'lat': -35.6751, 'lon': -71.5430, 'iso_alpha': 'CHL'},
    'Nigeria': {'lat': 9.0820, 'lon': 8.6753, 'iso_alpha': 'NGA'},
    'Kenya': {'lat': -0.0236, 'lon': 37.9062, 'iso_alpha': 'KEN'},
}

COUNTRY_NAME_MAPPING = {
    # Mapping between salary data (key) and CPI data (values - list of possible names in CPI data)
    'Monaco': ['Monaco'],
    'Bermuda*': ['Bermuda'], # Asterisk might need handling if it's part of the name in salary data
    'Liechtenstein': ['Liechtenstein'],
    'Norway': ['Norway'],
    'Switzerland': ['Switzerland'],
    'Luxembourg': ['Luxembourg'],
    'Singapore': ['Singapore'],
    'USA': ['United States', 'United States of America'], # 'United States' is common in World Bank CPI data
    'Ireland': ['Ireland'],
    'Australia': ['Australia'],
    'Denmark': ['Denmark'],
    'Iceland': ['Iceland'],
    'Netherlands': ['Netherlands'],
    'United Kingdom': ['United Kingdom', 'Britain'],
    'Germany': ['Germany'],
    'Sweden': ['Sweden'],
    'Belgium': ['Belgium'],
    'Finland': ['Finland'],
    'France': ['France'],
    'Austria': ['Austria'],
    'Israel': ['Israel'],
    'Canada': ['Canada'],
    'Japan': ['Japan'],
    'Italy': ['Italy'],
    'New Zealand': ['New Zealand'],
    'South Korea': ['Korea, Rep.', 'South Korea', 'Korea'], # 'Korea, Rep.' is common in World Bank
    'Spain': ['Spain'],
    'Cyprus': ['Cyprus'],
    'Malta': ['Malta'],
    'Slovenia': ['Slovenia'],
    'Portugal': ['Portugal'],
    'Greece': ['Greece'],
    'Estonia': ['Estonia'],
    'Czech Republic': ['Czech Republic', 'Czechia'],
    'Lithuania': ['Lithuania'],
    'Latvia': ['Latvia'],
    'Slovakia': ['Slovakia', 'Slovak Republic'],
    'Croatia': ['Croatia'],
    'Hungary': ['Hungary'],
    'Poland': ['Poland'],
    'Romania': ['Romania'],
    'Bulgaria': ['Bulgaria'],
    'Russia': ['Russian Federation', 'Russia'],
    'Turkey': ['Turkey', 'Turkiye', 'Türkiye'],
    'Kazakhstan': ['Kazakhstan'],
    'China': ['China'],
    'Mexico': ['Mexico'],
    'Brazil': ['Brazil'],
    'Malaysia': ['Malaysia'],
    'Thailand': ['Thailand'],
    'Indonesia': ['Indonesia'],
    'Philippines': ['Philippines'],
    'India': ['India'],
    'Vietnam': ['Vietnam'], # Might be 'Viet Nam' in some datasets
    'Egypt': ['Egypt, Arab Rep.', 'Egypt'],
    'South Africa': ['South Africa']
}

def load_categories_data(data_dir: str) -> Optional[pd.DataFrame]:
    """Load and process categories data from CSV file (Housing-related-expenditure-of-households.csv)"""
    categories_path = os.path.join(data_dir, "Housing-related-expenditure-of-households.csv")
    try:
        df = pd.read_csv(categories_path, sep=';', encoding='latin1')
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        return df
    except FileNotFoundError:
        print(f"Categories file not found at: {categories_path}")
        return None
    except Exception as e:
        print(f"Error loading categories data: {e}")
        traceback.print_exc()
        return None

def normalize_country_name(country):
    """Normalize country names, handling various data types."""
    if pd.isna(country) or country is None:  # Handle missing values directly
        return "N/A"
    if isinstance(country, (int, float)):
        country = str(int(country))
    elif not isinstance(country, str):
        country = str(country)

    country = country.strip()
    country_variations = {
            'USA': 'United States', 'United States of America': 'United States',
            'UK': 'United Kingdom', 'Great Britain': 'United Kingdom', 'Britain': 'United Kingdom',
            'Czechia': 'Czech Republic', 'České': 'Czech Republic', 'Česká': 'Czech Republic',
            'República Checa': 'Czech Republic', 'République tchèque': 'Czech Republic',
            'Republic of Moldova': 'Moldova',
            'Turkiye': 'Turkey', 'Türkiye': 'Turkey', 'T\x81rkiye': 'Turkey', # Handle encoding issue if present
            'Korea, Rep.': 'South Korea', 'Korea': 'South Korea', 'Republic of Korea': 'South Korea',
            'Russian Federation': 'Russia',
            'Slovak Republic': 'Slovakia',
            'Bosnia and Herzegovina': 'Bosnia-Herzegovina',
            'United Arab Emirates': 'UAE', 'UAE (United Arab Emirates)': 'UAE', # Using UAE as standard here, map COUNTRY_DATA if needed
            'Hong Kong SAR, China': 'Hong Kong',
            'Taiwan, China': 'Taiwan',
            'Macedonia, FYR': 'North Macedonia', 'Macedonia': 'North Macedonia',
            'The Netherlands': 'Netherlands',
            'Egypt, Arab Rep.': 'Egypt'
        # Add more variations as identified from data
    }
    # Case-insensitive direct match first
    normalized = country_variations.get(country, country)
    for k, v in country_variations.items():
        if k.lower() == country.lower():
            normalized = v
            break
    return normalized.strip()

def load_cpi_data(cpi_2_path: str) -> pd.DataFrame:
    """Loads CPI data from CPI_2.csv, handling missing values."""
    try:
        df = pd.read_csv(cpi_2_path, sep=';', encoding='utf-8') # Assuming UTF-8, adjust if needed
        df = df.set_index('Country Name') # Assuming 'Country Name' is the column for country names
        
        year_cols = [str(year) for year in range(1960, datetime.now().year + 2)] # Extend to cover potential future data
        for col in year_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        df = df.ffill(axis=1) # Forward fill missing CPI values
        

        print("Successfully loaded and processed data from CPI_2.csv")
        print(f"Loaded CPI for {len(df)} countries. Sample:")
        print(df.head())
        return df

    except FileNotFoundError:
        print(f"Error: CPI_2.csv not found at {cpi_2_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading or processing CPI_2.csv: {e}")
        traceback.print_exc()
        return pd.DataFrame()

import re

def load_data() -> Optional[Dict[str, pd.DataFrame]]:
    def convert_to_usd(amount, currency, exchange_rates, year):
        if currency == 'USD':
            return amount
        rate = exchange_rates.get(currency, {}).get(str(year), None)
        if rate is None:
            print(f"Warning: No exchange rate found for {currency} in {year}. Using 1.0.")
            rate = 1.0
        return amount * rate

    try:
        global DATA
        if not os.path.exists(DATA_FOLDER):
            raise FileNotFoundError(f"Data folder not found: {DATA_FOLDER}")
        
      
        # Load Monthly salary data
        print("Loading Monthly salary data...")
        monthly_salary_path = os.path.join(DATA_FOLDER, "Monthly_salary.csv")
        monthly_salary_df = pd.read_csv(monthly_salary_path, encoding='utf-8', sep=';', decimal=',')
        
        monthly_salary_df = monthly_salary_df.dropna(axis=1, how='all')
        monthly_salary_df = monthly_salary_df.dropna(how='all')
        first_col_name = monthly_salary_df.columns[0]
        monthly_salary_df = monthly_salary_df.rename(columns={first_col_name: 'Country'})
        
        monthly_salary_df = pd.melt(monthly_salary_df, id_vars=['Country'], var_name='Year', value_name='Monthly_Salary')
        
        monthly_salary_df['Year'] = pd.to_numeric(monthly_salary_df['Year'], errors='coerce')
        monthly_salary_df['Monthly_Salary'] = pd.to_numeric(
            monthly_salary_df['Monthly_Salary'].astype(str).str.replace(',', '.'), errors='coerce'
        )
        monthly_salary_df = monthly_salary_df.dropna(subset=['Year', 'Monthly_Salary'])
        print(f"Salary data loaded. Shape: {monthly_salary_df.shape}. Countries: {monthly_salary_df['Country'].nunique()}")

        # Load categories data (Housing-related expenditure)
        print("\nLoading Categories data...")
        categories_df_raw = pd.read_csv(
           os.path.join(DATA_FOLDER, "Housing-related-expenditure-of-households.csv"),
           encoding='latin1', sep=';', decimal=','
        )
        if categories_df_raw is None: raise ValueError("Failed to load categories data.")

        print("Original category columns:", categories_df_raw.columns.tolist())
        
        column_mapping = {
            # Ensure keys exactly match the column names in the CSV after stripping
            'Country': 'Country', # Assuming first column is Country
            'Housing, water, electricity, gas and other fuels': 'Housing',
            'Food and non-alcoholic beverages': 'Food',
            'Transport': 'Transport',
            'Miscellaneous goods and services': 'Miscellaneous',
            'Recreation and culture': 'Recreation',
            'Restaurants and hotels': 'Restaurants',
            'Furnishings, household equipment and routine maintenance of the house': 'Furnishings',
            'Clothing and footwear': 'Clothing',
            'Alcoholic beverages, tobacco and narcotics': 'Alcohol',
            'Health': 'Health',
            'Communications': 'Communications',
            'Education': 'Education',
            'Sum \'All other items\'?': 'Other' # Question mark might be part of name
        }
        # Strip column names in DataFrame before renaming
        categories_df_raw.columns = categories_df_raw.columns.str.strip()
        categories_df = categories_df_raw.rename(columns=column_mapping)
        
        # Keep only columns that were successfully mapped + 'Country'
        mapped_cols = [col for col in column_mapping.values() if col in categories_df.columns]
        categories_df = categories_df[['Country'] + [col for col in mapped_cols if col != 'Country']]


        for col in categories_df.columns:
            if col != 'Country':
                categories_df[col] = pd.to_numeric(
                    categories_df[col].astype(str).str.replace(',', '.'), errors='coerce'
                )
        print(f"Categories data loaded and processed. Shape: {categories_df.shape}. Countries: {categories_df['Country'].nunique()}")


        # Load Cost of living data
        print("\nLoading Cost of living data...")
        cost_living_path = os.path.join(DATA_FOLDER, "Cost_of_living.csv")
        cost_living_df = pd.read_csv(cost_living_path, encoding='utf-8', sep=';')
        
        cost_living_df['Cost of living'] = pd.to_numeric(
            cost_living_df['Cost of living'].astype(str).str.replace('$', '', regex=False).str.replace(',', ''),
            errors='coerce'
        )
        print(f"Cost of living data loaded. Shape: {cost_living_df.shape}. Countries: {cost_living_df['Country'].nunique()}")

        # Load CPI data
        print("\nLoading CPI data...")
        cpi_2_path = os.path.join(DATA_FOLDER, "CPI_2.csv")
        cpi_df = load_cpi_data(cpi_2_path)
        if cpi_df.empty:
             print("Warning: CPI data is empty. Cost adjustments might not work as expected.")
        else:
            # cpi_df = cpi_df.apply(pd.to_numeric, errors='coerce') # Already done in load_cpi_data
            cpi_df = cpi_df.dropna(how='all').dropna(axis=1, how='all')
            print(f"Final CPI data shape: {cpi_df.shape}")
            # print(f"CPI columns: {cpi_df.columns.tolist()}") # Potentially very long list
            # print(f"Sample of CPI data:\n{cpi_df.head()}")

         # Load VAT data
        print("\nLoading VAT data...")
        vat_data = None
        try:
            vat_path = os.path.join(DATA_FOLDER, "VAT.csv")

        # Pre-process the file content to fix the header row MORE ROBUSTLY
            with open(vat_path, 'r', encoding='utf-8-sig') as f:
                header = f.readline()  # Read the header row separately
                content = f.read()  # Read the rest of the content

            cleaned_header = header.replace('"', '').strip().split(',')
            cleaned_content = content.replace('"', '')

            vat_df = pd.read_csv(StringIO(cleaned_content), header=None, names=cleaned_header, on_bad_lines='skip')

        # Print the actual columns to debug (after cleaning)
            print("Original VAT columns (after cleaning):", vat_df.columns.tolist())

        # Find the country column (case-insensitive)
            country_col = None
            possible_country_cols = ['Country', 'Country Name', 'CountryName', 'Nation']
            for col in possible_country_cols:
                if col.lower() in (c.lower() for c in vat_df.columns):  # Case-insensitive check
                    country_col = next((c for c in vat_df.columns if c.lower() == col.lower()), None)  # Get actual case
                    break

            if not country_col:
                raise ValueError(f"No country column found. Available columns: {vat_df.columns.tolist()}")

            vat_df.rename(columns={country_col: 'Country'}, inplace=True)

        # Find year columns (2015-2024)
            year_columns = [str(year) for year in range(2015, 2025)]  # Updated range
            columns_to_keep = ['Country'] + year_columns
            vat_df = vat_df[columns_to_keep]
            
            # Convert year columns to numeric
            for col in year_columns:
                if col in vat_df.columns:
                    vat_df[col] = pd.to_numeric(
                        vat_df[col].astype(str)
                        .str.replace(',', '.')
                        .str.replace('"', ''),
                        errors='coerce'
                    )
            
            # Create long format dataframe
            vat_df_long = pd.melt(
                vat_df,
                id_vars=['Country'],
                value_vars=year_columns,
                var_name='Year',
                value_name='VAT_Rate'
            )
            
            # Convert Year to numeric and sort
            vat_df_long['Year'] = pd.to_numeric(vat_df_long['Year'])
            vat_df_long = vat_df_long.sort_values(['Country', 'Year'])
            
            # Remove rows with NaN values
            vat_df_long = vat_df_long.dropna(subset=['VAT_Rate'])
            
            # Create VAT rate getter function
            def get_vat_rate(country: str, year: int) -> Optional[float]:
                try:
                    matching_rows = vat_df_long[
                        (vat_df_long['Country'] == country) & 
                        (vat_df_long['Year'] == year)
                    ]
                    if matching_rows.empty:
                        return None
                    rate = matching_rows['VAT_Rate'].iloc[0]
                    return float(rate) if pd.notna(rate) else None
                except (IndexError, ValueError) as e:
                    print(f"Error getting VAT rate for {country} in {year}: {e}")
                    return None
            
            # Store VAT data
            vat_data = {
                'original': vat_df,
                'processed': vat_df_long,
                'get_rate': get_vat_rate
            }
            
            print("Successfully loaded and processed VAT data")
            print(f"Processed VAT data shape: {vat_df_long.shape}")
            print("Sample of processed VAT data:")
            print(vat_df_long.head())
            
        except Exception as e:
            print(f"Error loading VAT data: {str(e)}")
            traceback.print_exc()
            vat_data = None
        

        # Load Crude Oil Prices (Brent) - you can expand this for other benchmarks
        print("\nLoading Crude Oil data...")
        crude_oil_df = pd.read_csv(
            os.path.join(DATA_FOLDER, "Crude_oil.csv"),
            sep=",", encoding='utf-8',
            parse_dates=['observation_date'])
        crude_oil_df = crude_oil_df.rename(columns={'observation_date': 'Date', 'DCOILBRENTEU': 'Brent_Price'})

        # Load Currency Exchange Rates (Corrected)
        currency_path = os.path.join(DATA_FOLDER, "Currency.csv")

        # Use csv module for header parsing with newline=''
        print("\nLoading Currency data...")
        currency_path = os.path.join(DATA_FOLDER, "Currency.csv")
        currency_df = pd.read_csv(
            currency_path,
            encoding='utf-8-sig',
            quotechar='"',
            dtype=str,
            on_bad_lines='skip'
        )

        # Split the data if it's in one column
        if len(currency_df.columns) == 1:
            first_col = currency_df.columns[0]
            new_columns = first_col.split(',')
            new_columns = [col.strip('"') for col in new_columns]
            
            new_data = []
            for _, row in currency_df.iterrows():
                values = row[first_col].split(',')
                values = [v.strip('"') for v in values]
                if len(values) < len(new_columns):
                    values.extend([''] * (len(new_columns) - len(values)))
                new_data.append(values[:len(new_columns)])
            
            currency_df = pd.DataFrame(new_data, columns=new_columns)

        # Clean up column names and data
        currency_df.columns = currency_df.columns.str.strip().str.strip('"')
        
        # Convert year columns to numeric
        year_cols = [col for col in currency_df.columns if str(col).isdigit()]
        year_cols = [str(year) for year in range(2015, 2025)]  # Columns to keep
        cols_to_keep = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'] + year_cols
        currency_df = currency_df[cols_to_keep]
        for col in year_cols:
            currency_df[col] = pd.to_numeric(currency_df[col], errors='coerce')

        # Create a copy of the DataFrame with renamed column
        result_df = currency_df.copy()
        result_df = result_df.rename(columns={'Country Name': 'Country'})

        # Keep only necessary columns
        cols_to_keep = ['Country'] + year_cols
        result_df = result_df[cols_to_keep]

        # Remove any rows where all year values are NaN
        result_df = result_df.dropna(subset=year_cols, how='all')

        print("\nFinal DataFrame info:")
        print("Shape:", result_df.shape)
        print("Columns:", result_df.columns.tolist())
        print("\nSample of final data (first 5 rows, first 5 year columns):")
        sample_cols = ['Country'] + sorted(year_cols)[:5]
        print(result_df[sample_cols].head())
        
        print("\nNumber of countries:", len(result_df))
        print("Year range:", min(year_cols), "to", max(year_cols))

        # Pivot to get exchange rates by country and year
        exchange_rates_df = result_df.set_index('Country')

        print("Loaded and processed Currency data:", exchange_rates_df.shape)


        return {
            "expenses": categories_df, # This seems to be the same as "categories"
            "cost_living": cost_living_df,
            "salary": monthly_salary_df,
            "cpi": cpi_df,
            "categories": categories_df, # Main categories dataframe
            "vat_rates": vat_data,
            "crude_oil": crude_oil_df,
            "currency": exchange_rates_df
        }

    except Exception as e:
        print(f"Error in load_data: {str(e)}")
        traceback.print_exc()
        return None

DATA = load_data() # Call only once
if DATA is None:
    raise Exception("Failed to load required data files. Check file paths and formats.")

# Print info after loading
# for key, df_item in DATA.items():
#     if isinstance(df_item, pd.DataFrame):
#         print(f"\n{key} DataFrame Head:\n{df_item.head()}")
#         print(f"{key} DataFrame Info:")
#         df_item.info()
#     else:
#         print(f"\n{key}: Data is not a DataFrame (type: {type(df_item)})")
    
def get_iso3(country_name):
    try:
        return countries.lookup(country_name).alpha_3
    except LookupError:
        return None

cost_living_df = DATA['cost_living']  # extract it first
COUNTRY_NAME_TO_ISO3 = {name: get_iso3(name) for name in cost_living_df['Country'].unique()}


def get_available_years(country: str, salary_df: Optional[pd.DataFrame]) -> List[int]:
    """Get list of years with available salary data for a country"""
    if salary_df is None or salary_df.empty:
        return []
    try:
        years = sorted(salary_df[salary_df['Country'] == country]['Year'].unique())
        return [int(year) for year in years if pd.notna(year) and isinstance(year, (int, float))]
    except Exception as e:
        print(f"Error getting available salary years for {country}: {e}")
        return []

def get_common_years(countries: List[str], salary_df: Optional[pd.DataFrame]) -> List[int]:
    """Get list of years with salary data available for all selected countries"""
    if not countries or salary_df is None or salary_df.empty:
        return []
    
    common_years = set(get_available_years(countries[0], salary_df))
    for country in countries[1:]:
        country_years = set(get_available_years(country, salary_df))
        common_years = common_years.intersection(country_years)
    return sorted(list(common_years))

AVAILABLE_COUNTRIES_WITH_SALARY = []
if DATA and isinstance(DATA.get("salary"), pd.DataFrame) and not DATA["salary"].empty:
    AVAILABLE_COUNTRIES_WITH_SALARY = sorted(DATA["salary"]["Country"].unique().tolist())
    print(f"Loaded {len(AVAILABLE_COUNTRIES_WITH_SALARY)} countries with salary data.")
else:
    print("Warning: Salary data not loaded or 'Country' column missing. Country list will be empty.")


# --- Helper Functions ---
def create_base_figure(title: str, height: int = 700) -> go.Figure:
    """Create a base figure with common settings"""
    fig = go.Figure()
    fig.update_layout(
        title=title, title_x=0.5,
        hoverlabel=dict(bgcolor="white"),
        height=height, margin=dict(b=100, t=100),
        template="plotly_white"
    )
    return fig

def get_average_salary(country: str, year: int, salary_df: Optional[pd.DataFrame]) -> Optional[float]:
    """Gets the average monthly salary for a country and year."""
    if salary_df is None or salary_df.empty:
        # print(f"Salary data not available for get_average_salary({country}, {year})")
        return None
    try:
        # print(f"\nLooking for salary data for: Country: {country}, Year: {year}")
        # print(f"Available years in salary data for {country}: {salary_df[salary_df['Country'] == country]['Year'].unique()}")
        
        salary_data = salary_df[
            (salary_df['Country'] == country) & 
            (salary_df['Year'] == float(year)) # Ensure year is float for matching if needed
        ]
        
        if not salary_data.empty:
            return salary_data['Monthly_Salary'].iloc[0]
        
        # Try to get the closest available year if exact year not found
        country_data = salary_df[salary_df['Country'] == country]
        if not country_data.empty and not country_data['Year'].dropna().empty :
            available_years_for_country = country_data['Year'].dropna().astype(float)
            closest_year_idx = (available_years_for_country - float(year)).abs().idxmin()
            closest_year = available_years_for_country[closest_year_idx]
            
            salary_data_closest = country_data[country_data['Year'] == closest_year]
            if not salary_data_closest.empty:
                # print(f"No exact salary data for {country} in {year}. Using closest year: {int(closest_year)}")
                return salary_data_closest['Monthly_Salary'].iloc[0]
        
        # print(f"No salary data found for {country} for year {year} or nearby.")
        return None
    except Exception as e:
        print(f"Error getting salary for {country}, {year}: {e}")
        # traceback.print_exc()
        return None

def get_housing_expenditure(country: str, categories_df: Optional[pd.DataFrame]) -> Optional[float]:
    """Gets housing expenditure percentage for a country."""
    # Note: Original function had 'year' argument but didn't use it for categories_df lookup.
    # Assuming categories_df is not year-specific or uses a representative year.
    if categories_df is None or categories_df.empty:
        # print("Categories data not available for get_housing_expenditure.")
        return None
    try:
        # print(f"\nLooking for housing data for: Country: {country}")
        # print(f"Available countries in categories_df: {categories_df['Country'].unique()}")

        country_data = categories_df[categories_df['Country'] == country]
        
        if country_data.empty: # Try normalized name if exact match fails
            normalized_country = normalize_country_name(country)
            # print(f"Trying normalized country name for housing: {normalized_country}")
            country_data = categories_df[categories_df['Country'] == normalized_country]

        if country_data.empty:
            # print(f"No housing data (categories) found for {country} or its normalized form.")
            return None
            
        # Check for 'Housing' column (already renamed)
        if 'Housing' in country_data.columns:
            housing_val = country_data['Housing'].iloc[0]
            if pd.notna(housing_val): return float(housing_val)
        
        # print(f"Housing column not found or NaN for {country}. Available: {country_data.columns.tolist()}")
        return None
    except Exception as e:
        print(f"Error getting housing expenditure for {country}: {e}")
        # traceback.print_exc()
        return None


def get_cost_of_living(country: str, year: int, 
                      cost_living_df: Optional[pd.DataFrame], 
                      categories_df: Optional[pd.DataFrame], 
                      cpi_df: Optional[pd.DataFrame]) -> Optional[Dict[str, float]]:
    """Gets detailed cost breakdown for a country, adjusted by CPI relative to a base year."""
    if cost_living_df is None:  # We allow categories_df to be None now
        return None

    try:
        # First get the total cost from Cost_of_living.csv
        normalized_country_for_cost = normalize_country_name(country)
        cost_data_row = cost_living_df[cost_living_df['Country'] == normalized_country_for_cost]

        if cost_data_row.empty:
            print(f"No base cost data found for {country}")
            return None
        
        base_cost_from_file = cost_data_row['Cost of living'].iloc[0]
        if pd.isna(base_cost_from_file):
            return None
        base_cost_numeric = float(base_cost_from_file)

        # Apply CPI adjustment to total cost
        adjusted_cost = base_cost_numeric
        if cpi_df is not None and not cpi_df.empty:
            cpi_country_name_in_df = None
            for possible_name in [country, normalized_country_for_cost] + COUNTRY_NAME_MAPPING.get(country, []):
                if possible_name in cpi_df.index:
                    cpi_country_name_in_df = possible_name
                    break

            if cpi_country_name_in_df:
                target_year_str = str(year)
                base_year = 2025
                base_year_str = str(base_year)
                
                if target_year_str in cpi_df.columns:
                    try:
                        cumulative_factor = 1.0
                        current_year = int(target_year_str)
                        
                        cpi_values = {}
                        for y in range(current_year, base_year + 1):
                            y_str = str(y)
                            if y_str in cpi_df.columns:
                                cpi_val = pd.to_numeric(str(cpi_df.loc[cpi_country_name_in_df, y_str]).replace(',', '.'), errors='coerce')
                                if pd.notna(cpi_val):
                                    cpi_values[y] = cpi_val / 100.0

                        for y in range(current_year, base_year):
                            if y in cpi_values and y + 1 in cpi_values:
                                cumulative_factor *= (1 + cpi_values[y])
                        
                        adjusted_cost = base_cost_numeric / max(cumulative_factor, 0.1)
                        print(f"CPI Adjustment for {country} ({year}): {base_cost_numeric:.2f} -> {adjusted_cost:.2f}")
                    except Exception as e:
                        print(f"Error in CPI calculation: {e}")

        # Define default percentages based on typical values
        default_percentages = {
            'Housing': 0.25,      # 25% for housing
            'Food': 0.15,        # 15% for food
            'Transport': 0.12,    # 12% for transport
            'Miscellaneous': 0.10, # 10% for miscellaneous
            'Recreation': 0.08,    # 8% for recreation
            'Other': 0.30         # 30% for other expenses
        }

        costs_breakdown = {}
        
        # Try to get actual percentages from categories_df if available
        if categories_df is not None and not categories_df.empty:
            category_row_df = categories_df[categories_df['Country'] == normalized_country_for_cost]
            if category_row_df.empty and country != normalized_country_for_cost:
                category_row_df = categories_df[categories_df['Country'] == country]

            if not category_row_df.empty:
                # Define mapping between our categories and CSV columns
                category_columns = {
                    'Housing': 'Housing, water, electricity, gas and other fuels',
                    'Food': 'Food and non-alcoholic beverages',
                    'Transport': 'Transport',
                    'Miscellaneous': 'Miscellaneous goods and services',
                    'Recreation': 'Recreation and culture',
                    'Other': 'Sum \'All other items\'?'
                }

                total_percentage = 0
                valid_percentages = True

                # Try to get actual percentages
                for our_category, csv_column in category_columns.items():
                    try:
                        if csv_column in category_row_df.columns:
                            percentage_str = str(category_row_df[csv_column].iloc[0]).replace(',', '.')
                            percentage = float(percentage_str) / 100.0
                            
                            if pd.notna(percentage):
                                costs_breakdown[our_category] = float(adjusted_cost * percentage)
                                total_percentage += percentage
                            else:
                                valid_percentages = False
                                break
                        else:
                            valid_percentages = False
                            break
                    except (ValueError, TypeError) as e:
                        print(f"Error processing {country} - {our_category}: {e}")
                        valid_percentages = False
                        break

                # If we got all percentages successfully and they sum reasonably close to 1
                if valid_percentages and total_percentage > 0 and abs(total_percentage - 1.0) <= 0.1:
                    # Normalize if needed
                    if abs(total_percentage - 1.0) > 0.01:
                        for category in costs_breakdown:
                            costs_breakdown[category] = (costs_breakdown[category] / total_percentage)
                    return costs_breakdown

        # If we get here, either:
        # 1. No categories_df provided
        # 2. Country not found in categories_df
        # 3. Invalid percentages in categories_df
        # Use default percentages
        print(f"Using default percentages for {country}")
        for category, percentage in default_percentages.items():
            costs_breakdown[category] = float(adjusted_cost * percentage)

        return costs_breakdown

    except Exception as e:
        print(f"Error calculating costs for {country}, {year}: {e}")
        traceback.print_exc()
        return None

def get_cpi(country, year, cpi_df):
    try:
        year_str = str(year)
        if year_str not in cpi_df.columns:  # Check if the year exists as a column
            if year == 2025:
                year_str = "2024"
                print(f"Using 2024 CPI data for {country} (2025 data not available)")
            else:
                raise KeyError(f"Year {year} not found in CPI data")  # Raise KeyError if not 2025
        cpi = cpi_df.loc[country, year_str]
        return float(cpi) if pd.notna(cpi) else None
    except KeyError as e:
        print(f"CPI data not found for {country} in {year}: {e}")
        return None
    except (TypeError, ValueError) as e:  # Catch type conversion errors
        print(f"Error converting CPI value for {country} in {year}: {e}")
        return None
    
def calculate_correlation(series1, series2, lag=0):
    """
    Calculate Pearson correlation coefficient between two aligned numeric series.
    Optionally apply a lag (e.g. lag=1 means series1[y] vs series2[y+1])
    """
    if series1 is None or series2 is None:
        return None
    try:
        s1 = pd.to_numeric(series1, errors='coerce')
        s2 = pd.to_numeric(series2, errors='coerce')

        # Ensure integer index (years)
        s1.index = s1.index.astype(int)
        s2.index = s2.index.astype(int)

        # Apply lag: shift index of s2 if lag != 0
        if lag != 0:
            s2 = s2.copy()
            s2.index = s2.index - lag

        # Align years and drop missing
        valid_years = s1.index.intersection(s2.index)
        aligned_df = pd.DataFrame({
            's1': s1.loc[valid_years],
            's2': s2.loc[valid_years]
        }).dropna()

        print(f"Correlation data points (lag={lag}): {len(aligned_df)}")
        if len(aligned_df) >= 3:
            print(f"Series 1 values: {aligned_df['s1'].tolist()}")
            print(f"Series 2 values: {aligned_df['s2'].tolist()}")
            correlation = float(aligned_df['s1'].corr(aligned_df['s2']))
            print(f"Calculated correlation (lag={lag}): {correlation}")
            return correlation
        return None

    except Exception as e:
        print(f"Error in correlation calculation: {e}")
        return None

def get_historical_values(country, start_year, end_year, data_df, value_column, date_column='Year'):
    """Get historical values for a country between start_year and end_year."""
    if data_df is None:
        return None
    try:
        country_data = data_df[data_df['Country'] == country]
        if date_column in country_data.columns:
            filtered_data = country_data[
                (country_data[date_column] >= start_year) & 
                (country_data[date_column] <= end_year)
            ]
            return filtered_data[value_column].values
        return None
    except Exception as e:
        print(f"Error getting historical values for {country}: {e}")
        return None

def calculate_vat_inflation_correlation(country, year, vat_data, cpi_df, window=None):
    """Calculate correlation between VAT rates and inflation for 2015-2025."""
    try:
        if vat_data is None or cpi_df is None or 'get_rate' not in vat_data:
            return None
            
        data_points = []
        
        for y in range(2015, 2026):
            rate = vat_data['get_rate'](country, y)
            cpi = get_cpi(country, y, cpi_df)
            
            if rate is not None and cpi is not None:
                data_points.append({
                    'year': y,
                    'vat': rate,
                    'cpi': cpi
                })
                print(f"{country} - Year {y}: VAT = {rate}, CPI = {cpi}")

        if len(data_points) >= 2:
            df = pd.DataFrame(data_points)
            print(f"\nVAT trend data for {country}:")
            print(df)
            correlation = calculate_correlation(df['vat'], df['cpi'])
            print(f"VAT-CPI correlation for {country} (2015-2025): {correlation}")
            return correlation
        else:
            print(f"Not enough VAT data points for {country} over 2015-2025")
            return None

    except Exception as e:
        print(f"Error calculating VAT-Inflation correlation for {country}: {e}")
        traceback.print_exc()
        return None

def calculate_salary_inflation_correlation(country, year, salary_df, cpi_df, window=None):
    """Calculate correlation between salary and inflation for 2015-2025."""
    try:
        # Use fixed range 2015-2025 instead of window
        data_points = []
        
        for y in range(2015, 2026):  # 2015 to 2025 inclusive
            salary = get_average_salary(country, y, salary_df)
            cpi = get_cpi(country, y, cpi_df)
            
            if salary is not None and cpi is not None:
                data_points.append({
                    'year': y,
                    'salary': salary,
                    'cpi': cpi
                })
                print(f"{country} - Year {y}: Salary = {salary}, CPI = {cpi}")

        if len(data_points) >= 2:
            df = pd.DataFrame(data_points)
            print(f"\nSalary trend data for {country}:")
            print(df)
            correlation = calculate_correlation(df['salary'], df['cpi'])
            print(f"Salary-CPI correlation for {country} (2015-2025): {correlation}")
            return correlation
        else:
            print(f"Not enough salary data points for {country} over 2015-2025")
            return None
            
    except Exception as e:
        print(f"Error calculating Salary-Inflation correlation for {country}: {e}")
        traceback.print_exc()
        return None

def calculate_oil_inflation_correlation(country, year, crude_oil_df, cpi_df, window=5):
    """Calculate correlation between oil prices and inflation."""
    try:
        if crude_oil_df is None or cpi_df is None:
            return None
            
        start_year = year - window
        end_year = year
        
        # Get oil prices
        oil_prices = crude_oil_df[
            (crude_oil_df['Date'].dt.year >= start_year) &
            (crude_oil_df['Date'].dt.year <= end_year)
        ]['Brent_Price'].values
        
        # Get CPI values
        cpi_values = []
        for y in range(start_year, end_year + 1):
            cpi = get_cpi(country, y, cpi_df)
            if cpi is not None:
                cpi_values.append(cpi)
        
        return calculate_correlation(pd.Series(oil_prices), pd.Series(cpi_values))
    except Exception as e:
        print(f"Error calculating Oil-Inflation correlation for {country}: {e}")
        return None

def calculate_col_inflation_correlation(country, year, cost_living_df, cpi_df, window=None):
    """Calculate correlation between cost of living and inflation for 2015-2025."""
    try:
        data_points = []
        
        for y in range(2015, 2026):
            costs_breakdown = get_cost_of_living(country, y, cost_living_df, None, cpi_df)
            if costs_breakdown:
                total_cost = sum(costs_breakdown.values())
                cpi = get_cpi(country, y, cpi_df)
                if cpi is not None:
                    data_points.append({
                        'year': y,
                        'cost': total_cost,
                        'cpi': cpi
                    })
                    print(f"{country} - Year {y}: Cost = {total_cost}, CPI = {cpi}")

        if len(data_points) >= 2:
            df = pd.DataFrame(data_points)
            print(f"\nCost of Living trend data for {country}:")
            print(df)
            correlation = calculate_correlation(df['cost'], df['cpi'])
            print(f"Cost-CPI correlation for {country} (2015-2025): {correlation}")
            return correlation
        else:
            print(f"Not enough cost data points for {country} over 2015-2025")
            return None

    except Exception as e:
        print(f"Error calculating Cost-Inflation correlation for {country}: {e}")
        traceback.print_exc()
        return None
    
def get_exchange_rate(country, year, exchange_rates, country_mapping):
    """Retrieve exchange rate with robust country mapping and fallback."""
    try:
        year_str = str(year)
        mapped_country = country_mapping.get(country)  # Map country name

        if not mapped_country or mapped_country == "N/A":  # Handle missing or special cases
            print(f"Skipping exchange rate lookup for {country} (not in mapping)")
            return None

        if year_str not in exchange_rates.columns:
            if year > 2015:
                rate = get_exchange_rate(country, year - 1, exchange_rates, country_mapping)  # Recursive call with mapping
                if rate is not None:
                    print(f"Using exchange rate from {year - 1} for {country} in {year}")
                    return rate
            print(f"Exchange rate not found for {country} in {year} (or previous year). Using default.")
            return 1.0  # Or handle differently
        
        rate_str = exchange_rates.loc[mapped_country, year_str]  # Use mapped country for lookup
        rate = float(rate_str.replace(',', '.')) if rate_str else None
        print(f"Exchange rate for {country} ({year}): {rate}")
        return rate

    except (KeyError, ValueError, TypeError, AttributeError) as e:
        print(f"Error getting exchange rate for {country} ({year}): {e}")
        return None  # Or handle differently


def calculate_currency_inflation_correlation(country, year, currency_df, cpi_df, window=None):
    """Calculate correlation between exchange rates and inflation for 2015-2025."""
    try:
        data_points = []
        normalized_country = normalize_country_name(country)
        
        for y in range(2015, 2026):
            year_str = str(y)
            if year_str in currency_df.columns:
                try:
                    rate = currency_df.loc[normalized_country, year_str]
                    if pd.notna(rate):
                        rate = float(str(rate).replace(',', '.'))
                        cpi = get_cpi(country, y, cpi_df)
                        if cpi is not None:
                            data_points.append({
                                'year': y,
                                'rate': rate,
                                'cpi': cpi
                            })
                            print(f"{country} - Year {y}: Exchange Rate = {rate}, CPI = {cpi}")
                except (KeyError, ValueError) as e:
                    print(f"Error getting exchange rate for {country} in {y}: {e}")
                    continue

        if len(data_points) >= 2:
            df = pd.DataFrame(data_points)
            print(f"\nCurrency trend data for {country}:")
            print(df)
            correlation = calculate_correlation(df['rate'], df['cpi'])
            print(f"Currency-CPI correlation for {country} (2015-2025): {correlation}")
            return correlation
        else:
            print(f"Not enough currency data points for {country} over 2015-2025")
            return None

    except Exception as e:
        print(f"Error calculating Currency-Inflation correlation for {country}: {e}")
        traceback.print_exc()
        return None

# --- UI Definition ---
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style("""
            .title { color: #2a9d8f; margin-bottom: 20px; }
            .error-message { color: red; font-weight: bold; }
            .stats { background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 15px; white-space: pre-wrap; }
            .download-button { margin-top: 15px; }
        """)
    ),
    ui.h2("🌍 Global Cost of Living Comparison", class_="title"),

    ui.row(
        ui.column(3,
            ui.card(
                ui.card_header("Settings"),
                ui.input_select(
                    "primary_country", "Select Primary Country:",
                    choices=AVAILABLE_COUNTRIES_WITH_SALARY if AVAILABLE_COUNTRIES_WITH_SALARY else ["No data loaded"]
                ),
                ui.input_selectize(
                    "comparison_countries", "Add Countries to Compare:",
                    choices=AVAILABLE_COUNTRIES_WITH_SALARY if AVAILABLE_COUNTRIES_WITH_SALARY else [],
                    multiple=True
                ),
                ui.input_select(
                    "year", "Select Year:",
                    choices={str(year): str(year) for year in reversed(YEARS)}
                ),
                ui.div(ui.output_text("error_message"), class_="error-message"),
                ui.div(ui.output_text("country_stats"), class_="stats"),
                ui.download_button("download_data_btn", "📥 Download Selected Data", class_="download-button"), # Renamed ID
                ui.input_action_button("generate_plots", "📊 Generate Plots", class_="btn-primary")
            )
        ),
        ui.column(9,
            ui.navset_pill_list(
                ui.nav_panel("Historical Trends", ui.output_ui("historical_trend_ui")),
                ui.nav_panel("Country Comparison", ui.output_ui("country_comparison_ui")),
                ui.nav_panel("Cost Map", ui.output_ui("cost_map_ui")),
                ui.nav_panel("Inflation Trends", ui.output_ui("inflation_plot_ui")),
                ui.nav_panel("Radar Analysis", ui.output_ui("radar_chart_ui")),
                ui.nav_panel("Data Table", ui.output_data_frame("cost_table_df"))
            )
        )
    )
)

# --- Server Logic ---
def server(input: Inputs, output: Outputs, session: Session):

    salary_df_reactive = reactive.Value(DATA.get("salary"))
    cost_living_df_reactive = reactive.Value(DATA.get("cost_living"))
    categories_df_reactive = reactive.Value(DATA.get("categories"))
    cpi_df_reactive = reactive.Value(DATA.get("cpi"))

    @reactive.Calc
    def selected_countries_list() -> List[str]:
        primary = input.primary_country()
        comparisons = input.comparison_countries() or []
        
        if not primary or not isinstance(primary, str): return []
        
        final_comparisons = [c for c in comparisons if isinstance(c, str) and c != primary]
        return [primary] + final_comparisons

    # REMOVE @reactive.Calc from here:
    # @reactive.Calc # <--- REMOVE THIS LINE
    def aggregated_country_data(year_to_aggregate: int) -> Dict[str, Dict[str, Optional[any]]]:
        """Aggregates all relevant data for selected countries and a specific year.
           This is now a regular helper function, not a reactive.Calc itself.
        """
        data_dict: Dict[str, Dict[str, Optional[any]]] = {}

        salary_df = salary_df_reactive()
        cost_liv_df = cost_living_df_reactive()
        cat_df = categories_df_reactive()
        cpi_data_df = cpi_df_reactive()

        if salary_df is None or cost_liv_df is None or cat_df is None:
            print(f"Aggregated data: One or more core dataframes are None for year {year_to_aggregate}")
            return {}

        # Ensure selected_countries_list() is available and provides a list
        current_selected_countries = selected_countries_list()
        if not isinstance(current_selected_countries, list):
            print("Warning: selected_countries_list() did not return a list.")
            current_selected_countries = []


        for country_name in current_selected_countries: # Use the result of the reactive calc
            try:
                costs_breakdown = get_cost_of_living(country_name, year_to_aggregate, cost_liv_df, cat_df, cpi_data_df)
                current_salary = get_average_salary(country_name, year_to_aggregate, salary_df)
                housing_percentage = get_housing_expenditure(country_name, cat_df)

                if costs_breakdown is None:
                    continue

                total_calculated_cost = sum(costs_breakdown.values())
                
                data_dict[country_name] = {
                    "costs": costs_breakdown,
                    "salary": current_salary,
                    "housing_percentage": housing_percentage,
                    "total_cost": total_calculated_cost
                }

            except Exception as e_agg:
                print(f"Error aggregating data for {country_name}, year {year_to_aggregate}: {e_agg}")
                traceback.print_exc()
                data_dict[country_name] = {"costs": None, "salary": None, "housing_percentage": None, "total_cost": None}
                continue
            
        return data_dict

    @reactive.Calc
    def current_input_year_data() -> dict:
        """Data for the year selected in the input dropdown."""
        # This now calls the regular function 'aggregated_country_data'
        return aggregated_country_data(int(input.year()))

    @reactive.Calc
    def historical_data_for_primary() -> Dict[int, Optional[Dict[str, any]]]:
        """Get historical data (all years in YEARS constant) for the primary country."""
        primary_c = input.primary_country()
        if not primary_c: return {}

        hist_data: Dict[int, Optional[Dict[str, any]]] = {}
        
        salary_df = salary_df_reactive()
        cost_liv_df = cost_living_df_reactive()
        cat_df = categories_df_reactive()
        cpi_data_df = cpi_df_reactive()

        if salary_df is None or cost_liv_df is None or cat_df is None:
             print("Historical data: One or more core dataframes are None.")
             return {}

        for year_val in YEARS:
            costs_b = get_cost_of_living(primary_c, year_val, cost_liv_df, cat_df, cpi_data_df)
            salary_s = get_average_salary(primary_c, year_val, salary_df)
            if costs_b and salary_s is not None:
                hist_data[year_val] = {
                    "costs": costs_b,
                    "salary": salary_s,
                    "total_cost": sum(costs_b.values())
                }
            else:
                hist_data[year_val] = None # Or some indicator of missing data
        return hist_data

          
    @output
    @render.text
    def error_message() -> str:
        primary_c = input.primary_country()
        if not primary_c: return "Please select a primary country."
        
        year_sel = int(input.year())
        agg_data = current_input_year_data() # Data for selected year

        primary_country_data = agg_data.get(primary_c)
        if not primary_country_data or primary_country_data.get("total_cost") is None:
            return f"⚠️ Data may be incomplete for {primary_c} for {year_sel}. Some visualizations might not render."
        
        if input.comparison_countries():
            missing_data_countries = []
            for comp_country in input.comparison_countries():
                if comp_country == primary_c: continue # Already checked
                comp_data = agg_data.get(comp_country)
                if not comp_data or comp_data.get("total_cost") is None:
                    missing_data_countries.append(comp_country)
            if missing_data_countries:
                return f"⚠️ Data may be incomplete for comparison countries ({', '.join(missing_data_countries)}) for {year_sel}."
        return "" # No error

    @output
    @render.text
    def country_stats():
        primary_c = input.primary_country()
        if not primary_c: return "Select a primary country."

        year_sel = int(input.year())
        country_agg_data = current_input_year_data().get(primary_c)
    
        if not country_agg_data or country_agg_data.get("costs") is None or country_agg_data.get("salary") is None:
            return f"Data not fully available for {primary_c} for {year_sel}."

        total_cost_val = country_agg_data.get("total_cost", 0.0)
        salary_val = country_agg_data.get("salary", 0.0)

        stats_text = f"Selected: {primary_c} ({year_sel})\n"
        stats_text += f"Total Monthly Costs: ${total_cost_val:,.2f}\n"
        stats_text += f"Avg Monthly Salary: ${salary_val:,.2f}\n"
        if salary_val and salary_val != 0:
            ratio = total_cost_val / salary_val
            stats_text += f"Cost-to-Salary Ratio: {ratio:.2%}"
        else:
            stats_text += "Cost-to-Salary Ratio: N/A (salary is zero or unavailable)"
        return stats_text

    @output
    @render.ui
    @reactive.event(input.generate_plots)
    def historical_trend_ui():
        primary_c = input.primary_country()
        if not primary_c:
            return ui.p("Please select a primary country and click 'Generate Plots'.")

        hist_data_primary = historical_data_for_primary()
    
        plot_df_data = []
        categories = ['Housing', 'Food', 'Transport', 'Miscellaneous', 'Recreation', 'Other']
    
        for year_val, data_for_year in hist_data_primary.items():
            if data_for_year and data_for_year.get("costs"):
                row = {'Year': year_val, 'Monthly Salary': data_for_year.get('salary')}
                costs = data_for_year.get('costs', {})
                for cat in categories:
                    row[cat] = costs.get(cat, 0.0)
                plot_df_data.append(row)
    
        if not plot_df_data:
            return ui.p(f"Not enough historical data to plot for {primary_c}.")

        df = pd.DataFrame(plot_df_data).sort_values(by='Year')
    
        fig = go.Figure()
    
    # Add stacked bars for each category
        colors = px.colors.qualitative.Set3
        for i, category in enumerate(categories):
            fig.add_trace(go.Bar(
                name=category,
                x=df['Year'],
                y=df[category],
                marker_color=colors[i % len(colors)],
                hovertemplate=f"{category}: %{{y:$,.2f}}<extra></extra>"
        ))

    # Add salary line
        fig.add_trace(go.Scatter(
            name='Monthly Salary',
            x=df['Year'],
            y=df['Monthly Salary'],
            mode='lines+markers',
            line=dict(color='red', width=2),
            hovertemplate="Salary: %{y:$,.2f}<extra></extra>"
    ))

    # Add total costs line
        df['Total Costs'] = df[categories].sum(axis=1)
        fig.add_trace(go.Scatter(
            name='Total Monthly Costs',
            x=df['Year'],
            y=df['Total Costs'],
            mode='lines+markers',
            line=dict(color='blue', width=2, dash='dot'),
            hovertemplate="Total Costs: %{y:$,.2f}<extra></extra>"
    ))

        fig.update_layout(
            title=f'Costs vs. Salary in {primary_c} ({df["Year"].min()}-{df["Year"].max()})',
            barmode='stack',
            xaxis_title="Year",
            yaxis_title="Amount (USD per month)",
            hovermode='x unified',
            showlegend=True,
            height=600
    )
        
        
        return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))


    @output
    @render.ui
    @reactive.event(input.generate_plots)
    def country_comparison_ui():
        sel_countries = selected_countries_list()
        if not sel_countries:
            return ui.p("Please select countries and click 'Generate Plots'.")

        year_sel = int(input.year()) # Use the selected year for comparison
        agg_data_for_comparison = aggregated_country_data(year_sel) 
        
        plot_data = []
        
        for country_name in sel_countries:
            country_specific_data = agg_data_for_comparison.get(country_name)
            if country_specific_data and country_specific_data.get("costs"):
                entry = {'Country': country_name}
                entry.update(country_specific_data['costs']) # Add cost categories
                if country_specific_data.get("salary") is not None:
                    entry['Salary'] = country_specific_data['salary']
                plot_data.append(entry)

        if not plot_data:
            return ui.p(f"No comparable data for selected countries for {year_sel}.")

        df_comp = pd.DataFrame(plot_data)
        if df_comp.empty:
            return ui.p(f"No comparable data for selected countries for {year_sel} (empty DataFrame).")


        fig = go.Figure()
        cost_categories_in_df = [col for col in df_comp.columns if col not in ['Country', 'Salary']]
        colors = px.colors.qualitative.Plotly

        for i, category in enumerate(cost_categories_in_df):
            if category in df_comp: # Ensure category exists
                fig.add_trace(go.Bar(
                    name=category, x=df_comp['Country'], y=df_comp[category],
                    marker_color=colors[i % len(colors)],
                    hovertemplate=f"{category}: %{{y:$,.2f}}<extra></extra>"
                ))

        if 'Salary' in df_comp.columns:
            fig.add_trace(go.Scatter(
                name='Monthly Salary', x=df_comp['Country'], y=df_comp['Salary'],
                mode='lines+markers', line=dict(color='red', width=2), marker=dict(color='red', size=8),
                hovertemplate="Salary: %{y:$,.2f}<extra></extra>"
            ))

        fig.update_layout(
            title=f'Cost of Living Comparison ({year_sel})',
            barmode='stack', xaxis_title="Country", yaxis_title="Amount (USD)",
            hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            margin=dict(b=120, t=80), height=600
        )
        return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))


    @output
    @render.ui
    @reactive.event(input.generate_plots)
    def cost_map_ui():
        sel_countries_for_highlight = selected_countries_list()
        year_sel_map = int(input.year())

        cost_liv_df = cost_living_df_reactive()
        cat_df = categories_df_reactive()
        cpi_data_df = cpi_df_reactive()

        if cost_liv_df is None or cat_df is None:
            return ui.p("Core data for map not loaded (cost_living or categories).")

        all_countries_for_map = cost_liv_df['Country'].unique()

        choropleth_data = []

        for country in all_countries_for_map:
            costs_info = get_cost_of_living(country, year_sel_map, cost_liv_df, cat_df, cpi_data_df)

            if costs_info:
                total_c = sum(costs_info.values())
                iso3 = COUNTRY_NAME_TO_ISO3.get(country) or COUNTRY_NAME_TO_ISO3.get(normalize_country_name(country))
                if iso3:
                    choropleth_data.append({
                        'iso3': iso3,
                        'country_name_display': country,
                        'total_cost': total_c,
                        'is_selected': country in sel_countries_for_highlight or \
                                   normalize_country_name(country) in sel_countries_for_highlight
                })

        if not choropleth_data:
            return ui.p("No data available for the map visualization for the selected year.")

        df_map = pd.DataFrame(choropleth_data)

        fig = go.Figure(data=go.Choropleth(
            locations=df_map['iso3'],
            z=df_map['total_cost'],
            text=df_map['country_name_display'],
            hovertemplate="<b>%{text}</b><br>Cost of Living: <b>$%{z:,.0f}</b><extra></extra>",
    # Custom colorscale with more color steps in the lower range
            colorscale=[
                [0, 'rgb(0, 0, 128)'],         # Dark blue for lowest values (~500)
                [0.1, 'rgb(0, 0, 255)'],       # Blue (~750)
                [0.2, 'rgb(0, 128, 255)'],     # Light blue (~1000)
                [0.3, 'rgb(0, 255, 255)'],     # Cyan (~1250)
                [0.4, 'rgb(0, 255, 128)'],     # Blue-green (~1500)
                [0.5, 'rgb(0, 255, 0)'],       # Green (~1750)
                [0.6, 'rgb(128, 255, 0)'],     # Yellow-green (~2000)
                [0.7, 'rgb(255, 255, 0)'],     # Yellow (~2250)
                [0.8, 'rgb(255, 128, 0)'],     # Orange (~2500)
                [0.9, 'rgb(255, 0, 0)'],       # Red (~2750)
                [1.0, 'rgb(128, 0, 0)']        # Dark red for highest values (~3000)
    ],
            autocolorscale=False,
            reversescale=False,
            marker_line_color='rgb(50, 50, 50)',
            marker_line_width=0.5,
            colorbar_title="Cost of Living ($)",
            zmin=df_map['total_cost'].min(),
            zmax=min(df_map['total_cost'].max(), 3000),
    # Create more tick marks on the color scale
            colorbar=dict(
                tickvals=[500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000],
                ticktext=['$500', '$750', '$1000', '$1250', '$1500', '$1750', '$2000', '$2250', '$2500', '$2750', '$3000']
    )
))

# Make no-data areas very distinct
        fig.update_geos(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth',
            landcolor='rgb(220, 220, 220)',    # Light grey for no data
            showcountries=True,
            countrycolor='rgb(80, 80, 80)',    # Darker grey for borders
            countrywidth=0.5
)

# Improve overall layout
        fig.update_layout(
            title_text=f'Global Cost of Living Map ({year_sel_map})',
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth'
    ),
            height=600, 
            margin={"r":0,"t":50,"l":0,"b":0},
            font=dict(size=14)
)
        return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))


    @output
    @render.ui
    @reactive.event(input.generate_plots)
    def inflation_plot_ui():
        primary_c = input.primary_country()
        if not primary_c:
            return ui.p("Please select a primary country and click 'Generate Plots'.")

        cpi_data = cpi_df_reactive()
        if cpi_data is None or cpi_data.empty:
            return ui.p("CPI data not available for inflation plot.")

    # Find the country in CPI data
        country_in_cpi_df = None
        if primary_c in cpi_data.index:
            country_in_cpi_df = primary_c
        else:
            norm_primary_c = normalize_country_name(primary_c)
            if norm_primary_c in cpi_data.index:
                country_in_cpi_df = norm_primary_c
            else:
                mapped_names = COUNTRY_NAME_MAPPING.get(primary_c, [])
                for mapped_name in mapped_names:
                    if mapped_name in cpi_data.index:
                        country_in_cpi_df = mapped_name
                        break

        if not country_in_cpi_df:
            return ui.p(f"No CPI data found for {primary_c}")

    # Get the inflation rates directly from CPI data
        country_data = cpi_data.loc[country_in_cpi_df]
    
    # Create DataFrame with years and inflation rates
        inflation_rates_list = []
        for year in range(2015, 2025):  # Adjust year range as needed
            year_str = str(year)
            if year_str in country_data.index:
                rate = country_data[year_str]
                if pd.notna(rate):
                    inflation_rates_list.append({
                    'Year': year,
                    'Inflation Rate': float(rate)
                })

        if not inflation_rates_list:
            return ui.p(f"No inflation data available for {primary_c}")

        df_inflation = pd.DataFrame(inflation_rates_list)

    # Create the plot
        fig = go.Figure()
    
    # Add inflation rate line
        fig.add_trace(go.Scatter(
            x=df_inflation['Year'],
            y=df_inflation['Inflation Rate'],
            mode='lines+markers',
            name='Inflation Rate',
            line=dict(color='royalblue'),
            hovertemplate='Year: %{x}<br>Rate: %{y:.2f}%<extra></extra>'
    ))

    # Add trend line if enough points
        if len(df_inflation) >= 2:
            z = np.polyfit(df_inflation['Year'], df_inflation['Inflation Rate'], 1)
            p = np.poly1d(z)
            trend_values = p(df_inflation['Year'])
            fig.add_trace(go.Scatter(
                x=df_inflation['Year'],
                y=trend_values,
                mode='lines',
                name=f'Trend (slope: {z[0]:.2f}%/year)',
                line=dict(color='red', dash='dash'),
                hovertemplate='Year: %{x}<br>Trend: %{y:.2f}%<extra></extra>'
        ))

    # Add average line
        avg_inflation = df_inflation['Inflation Rate'].mean()
        fig.add_trace(go.Scatter(
            x=[df_inflation['Year'].min(), df_inflation['Year'].max()],
            y=[avg_inflation, avg_inflation],
            mode='lines',
            name=f'Average ({avg_inflation:.2f}%)',
            line=dict(color='green', dash='dot'),
            hovertemplate='Average: %{y:.2f}%<extra></extra>'
    ))

        fig.update_layout(
            title=f'Inflation Trend for {primary_c}',
            xaxis_title='Year',
            yaxis_title='Inflation Rate (%)',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            height=500,
            margin=dict(b=100, t=80)
    )
        
        return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))
    
    @output
    @render.ui
    @reactive.event(input.generate_plots)
    def radar_chart_ui():
        """Create radar chart with correlation values."""
        sel_countries = selected_countries_list()
        if not sel_countries:
            return ui.p("Please select countries to compare.")

        year_sel = int(input.year())
        chart_data = []

    # Define features and their display names
        features_config = {
            'vat_correlation': {
                'name': 'VAT-Inflation',
                'getter': lambda x: calculate_vat_inflation_correlation(x, year_sel, DATA.get("vat_rates"), DATA.get("cpi")),
                'formatter': lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        },
            'salary_correlation': {
                'name': 'Salary-Inflation',
                'getter': lambda x: calculate_salary_inflation_correlation(x, year_sel, DATA.get("salary"), DATA.get("cpi")),
                'formatter': lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        },
            'oil_correlation': {
                'name': 'Oil-Inflation',
                'getter': lambda x: calculate_oil_inflation_correlation(x, year_sel, DATA.get("crude_oil"), DATA.get("cpi")),
                'formatter': lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        },
            'col_correlation': {
                'name': 'Cost of Living-Inflation',
                'getter': lambda x: calculate_col_inflation_correlation(x, year_sel, DATA.get("cost_living"), DATA.get("cpi")),
                'formatter': lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        },
            'currency_correlation': {
                'name': 'Currency-Inflation',
                'getter': lambda x: calculate_currency_inflation_correlation(x, year_sel, DATA.get("currency"), DATA.get("cpi")),
                'formatter': lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        }
    }

        for country in sel_countries:
            try:
                row_data = {'Country': country}
                for feature_id, config in features_config.items():
                    try:
                        value = config['getter'](country)
                        if value is not None and not pd.isna(value):
                            row_data[feature_id] = float(abs(value))  
                        else:
                            row_data[feature_id] = np.nan
                    except Exception as e:
                        print(f"Error calculating {feature_id} for {country}: {e}")
                        row_data[feature_id] = np.nan

                if any(pd.notna(row_data[feat]) for feat in features_config.keys()):
                    chart_data.append(row_data)
                    print(f"Data added for {country}: {row_data}")
                else:
                    print(f"No valid data for {country}, skipping.")

            except Exception as e:
                print(f"Error processing data for {country}: {e}")

        if not chart_data:
            return ui.p("No valid correlation data for selected countries.")

        try:
            df = pd.DataFrame(chart_data)
            feature_ids = list(features_config.keys())

            fig = go.Figure()
            colors = px.colors.qualitative.Set3

            for idx, (_, row) in enumerate(df.iterrows()):
                r_values = row[feature_ids].tolist()
                theta_labels = [features_config[f]['name'] for f in feature_ids]
                hover_text = [
                    f"{features_config[f]['name']}: {features_config[f]['formatter'](row[f])}" for f in feature_ids
            ]

                fig.add_trace(go.Scatterpolar(
                    r=r_values,
                    theta=theta_labels,
                    fill='toself',
                    name=row['Country'],
                    line_color=colors[idx % len(colors)],
                    opacity=0.7,
                    text=hover_text,
                    hovertemplate="<b>%{theta}</b><br>Correlation: %{r:.2f}<extra></extra>",
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[-1, 1],
                        tickformat='.2f',
                        gridcolor='rgba(0,0,0,0.1)',
                        showline=False
                )
            ),
                title=dict(
                    text=f"Economic Indicators Correlation with Inflation ({year_sel})",
                    x=0.5,
                    y=0.95,
                    font=dict(size=20)
            ),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
            ),
                margin=dict(t=100, b=150),
                height=700
        )
            
            return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))

        except Exception as e:
            print(f"Error creating radar chart: {e}")
            return ui.p(f"Error creating radar chart: {str(e)}")
    
    def get_formatted_values_for_hover(country, year, features_config):
        """Formats the original values for the hover tooltip."""
        formatted_values = ""
        for feature_id, config in features_config.items():
            value = config['getter'](country)
            formatted_values += f"{config['name']}: {config['formatter'](value)}&lt;br&gt;"
        return formatted_values
    

    @output
    @render.data_frame
    def cost_table_df():
        sel_countries_table = selected_countries_list()
        if not sel_countries_table: return pd.DataFrame()

        year_sel_table = int(input.year())
        agg_data_for_table = aggregated_country_data(year_sel_table) # Use selected year
        
        data_rows_list = []
        
        for country_name_table in sel_countries_table:
            country_data_table = agg_data_for_table.get(country_name_table)
            if country_data_table and country_data_table.get("costs"):
                costs_dict = country_data_table["costs"]
                salary_val_table = country_data_table.get("salary")
                
                row_data = {'Country': country_name_table, 'Year': year_sel_table}
                row_data['Monthly Salary'] = f"${salary_val_table:,.2f}" if salary_val_table is not None else 'N/A'
                row_data['Total Cost'] = f"${country_data_table.get('total_cost', 0.0):,.2f}"
                
                for cat_name in ['Housing', 'Food', 'Transport', 'Miscellaneous', 'Recreation', 'Other']:
                    row_data[cat_name] = f"${costs_dict.get(cat_name, 0.0):,.2f}"
                data_rows_list.append(row_data)
        
        if not data_rows_list: return pd.DataFrame()
        
        df_output = pd.DataFrame(data_rows_list)
        # Order columns nicely
        cols_order = ['Country', 'Year', 'Monthly Salary', 'Total Cost', 'Housing', 'Food', 'Transport', 'Miscellaneous', 'Recreation', 'Other']
        existing_cols_order = [col for col in cols_order if col in df_output.columns]
        return df_output[existing_cols_order]


    @render.download(filename=lambda: f"cost_of_living_data_{input.primary_country()}_{input.year()}.csv")
    async def download_data_btn(): # Matched ID from UI
        # Download data for selected countries and selected year
        sel_countries_dl = selected_countries_list()
        year_sel_dl = int(input.year())
        
        if not sel_countries_dl:
            yield "Country,Year,Status\n"
            yield f",,{datetime.now()},No countries selected\n" # Empty if no countries
            return

        agg_data_for_dl = aggregated_country_data(year_sel_dl)

        output_sio = StringIO()
        header = ['Country', 'Year', 'Monthly_Salary', 'Total_Cost']
        default_cost_cats = ['Housing', 'Food', 'Transport', 'Miscellaneous', 'Recreation', 'Other']
        header.extend(default_cost_cats)
        output_sio.write(",".join(header) + "\n")

        for country_name_dl in sel_countries_dl:
            country_data_dl = agg_data_for_dl.get(country_name_dl)
            if country_data_dl and country_data_dl.get("costs"):
                costs_dl = country_data_dl["costs"]
                salary_dl = country_data_dl.get("salary", np.nan) # Use NaN for missing numeric
                total_cost_dl = country_data_dl.get("total_cost", np.nan)

                row_items = [
                    country_name_dl, str(year_sel_dl), 
                    f"{salary_dl:.2f}" if pd.notna(salary_dl) else "",
                    f"{total_cost_dl:.2f}" if pd.notna(total_cost_dl) else ""
                ]
                for cat_key in default_cost_cats:
                    cat_val = costs_dl.get(cat_key, np.nan)
                    row_items.append(f"{cat_val:.2f}" if pd.notna(cat_val) else "")
                
                output_sio.write(",".join(row_items) + "\n")
            else: # No data for this country/year combination
                 output_sio.write(f"{country_name_dl},{year_sel_dl},{'No data found for this combination'}\n")


        yield output_sio.getvalue()

app = App(app_ui, server)

#app._app.mount("/", StaticFiles(directory="www"), name="www")