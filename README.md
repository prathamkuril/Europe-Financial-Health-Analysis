# Financial Data Analysis for European Industries

This project focuses on analyzing various financial metrics of industries in Europe, using datasets such as Market Capitalization, EVA (Economic Value Added), Gross Profit, and more. The objective is to clean, merge, and analyze these datasets, visualize the relationships between financial metrics, and filter industries based on different financial conditions.



## Data Analysis Topic: Comparative Analysis of Industry Financial Health of Europe in  Year 2022 

**Datasets Used:**
1. Employee Statistics by Industry
2. EVA and Equity EVA by Industry
3. Market Capitalization by Industry
4. Tax Rate by Industry
5. Cost of Capital by Industry
6. Operating Lease by Industry
7. Total Beta by Industry Sector
8. Dollar Value Measures by Industry

**Dataset link:** https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datacurrent.html


**Analytical Questions** 
1. How does the financial health of different industries compare when considering metrics such as Economic Value Added (EVA), market capitalization, revenues, and profitability within a given year?
2. How does the tax efficiency of industries impact their overall performance, and what correlations exist between tax rates, financial metrics, and profitability in that specific year?
3. What insights can be gained by comparing the cost of capital and risk factors (beta, cost of debt) across different industries, and how do these factors influence financial performance and market dynamics in the given year?
4. How do differences in operational aspects, such as operating leases and debt structures, affect operational income, Return on Invested Capital (ROIC), and pre-tax margins among various industries?
5. In comparing dollar value metrics like market capitalization, book equity, enterprise value, and revenues across industries, what strengths and weaknesses unique to each industry emerge?

## Project Structure

The project consists of the following key steps:

### 1. Importing Libraries

We use a variety of Python libraries such as:

- `pandas` for data manipulation and analysis.
- `numpy` for numerical operations.
- `scipy.stats` for statistical analysis.
- `matplotlib` and `seaborn` for visualizations.

```python
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
```

### 2. Loading Datasets

Multiple CSV files are loaded into pandas DataFrames:

- `EVAEurope.csv`
- `EmployeeEurope.csv`
- `MktCapEurope.csv`
- `DollarEurope.csv`
- `taxrateEurope.csv`
- `CostOfCapitalEurope.csv`
- `OperatingLeaseAdjustedValuesForDebtReturnsAndEarnings.csv`
- `totalbetaEurope.csv`

```python
EVAEurope = pd.read_csv('EVAEurope.csv')
EmployeeEurope = pd.read_csv('EmployeeEurope.csv')
MktCapEurope = pd.read_csv('MktCapEurope.csv')
DollarEurope = pd.read_csv('DollarEurope.csv')
taxrateEurope = pd.read_csv('taxrateEurope.csv')
CostOfCapitalEurope = pd.read_csv('CostOfCapitalEurope.csv')
OperatingLease = pd.read_csv('OperatingLeaseAdjustedValuesForDebtReturnsAndEarnings.csv')
totalbetaEurope = pd.read_csv('totalbetaEurope.csv')
```

### 3. Data Preprocessing

The datasets are cleaned by renaming columns for consistency, dropping unnecessary columns, and merging them into a single DataFrame based on the `Industry Name` column.

```python
# Renaming columns for consistency
taxrateEurope.rename(columns={'Industry name': 'Industry Name'}, inplace=True)
DollarEurope.rename(columns={'Industry  Name': 'Industry Name'}, inplace=True)

# Dropping unnecessary columns
EmployeeEurope.drop(columns=['Number of firms'], inplace=True)
MktCapEurope.drop(columns=['Number of firms'], inplace=True)
DollarEurope.drop(columns=['Number of firms'], inplace=True)
taxrateEurope.drop(columns=['Number of firms'], inplace=True)
CostOfCapitalEurope.drop(columns=['Number of Firms'], inplace=True)
OperatingLease.drop(columns=['Number of firms'], inplace=True)
totalbetaEurope.drop(columns=['Number of firms'], inplace=True)

# Merging dataframes based on 'Industry Name'
merged_data = pd.merge(EmployeeEurope, EVAEurope, on='Industry Name')
merged_data = pd.merge(merged_data, MktCapEurope, on='Industry Name')
merged_data = pd.merge(merged_data, taxrateEurope, on='Industry Name')
merged_data = pd.merge(merged_data, CostOfCapitalEurope, on='Industry Name')
merged_data = pd.merge(merged_data, OperatingLease, on='Industry Name')
merged_data = pd.merge(merged_data, totalbetaEurope, on='Industry Name')
merged_data = pd.merge(merged_data, DollarEurope, on='Industry Name')

# Removing duplicate columns
columns_to_remove = [
    'Revenues ($ millions)_x', 'Beta_x', 'Cost of Equity_x', 'Cost of Capital_x', 
    'Std Dev in Stock_x', 'E/(D+E)_x', 'Cost of Debt_x', 'Tax Rate_x', 'Beta_y', 
    'Cost of Equity_y', 'E/(D+E)_y', 'Std Dev in Stock_y', 'Cost of Debt_y', 
    'Tax Rate_y', 'After-tax Cost of Debt_y', 'D/(D+E)_y', 'Cost of Capital_y', 
    'Revenues ($ millions)_y', 'After-tax Cost of Debt_x', 'D/(D+E)_x'
]
merged_data.drop(columns=columns_to_remove, inplace=True)
```

### 4. Cleaning and Converting Data

We ensure the numeric columns are properly formatted by cleaning special characters and converting string-based numeric values to floats.

```python
def clean_and_convert_to_float(value):
    # Remove special characters and convert to float
    try:
        if '%' in value:
            cleaned_value = float(value.replace('%', '')) / 100.0
        else:
            cleaned_value = float(value.replace('$', '').replace('₹', '').replace(',', '').replace(')', '').replace('(', ''))
        return cleaned_value
    except ValueError:
        return float('NaN')  # Handle non-convertible values as NaN

# Columns to convert to float (excluding 'Industry Name')
columns_to_convert = [col for col in merged_data.columns if col != 'Industry Name']

# Convert columns to float after cleaning
for col in columns_to_convert:
    merged_data[col] = merged_data[col].apply(lambda x: clean_and_convert_to_float(x) if isinstance(x, str) else x).astype(float)
```

### 5. Exploratory Data Analysis (EDA)

- **Checking for missing values**: We identify null values in the dataset and inspect the structure using `merged_data.info()`.
- **Correlation Heatmap**: A correlation matrix is generated and visualized to identify relationships between different financial metrics.

```python
# Calculate correlations
correlation_matrix = merged_data.corr()

# Plotting correlation heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Financial Metrics')
plt.show()
```

### 6. Visualization

We provide various visualizations to understand the financial metrics, such as:

- **Correlation heatmap**: Visualizes the relationships between features.
- **Market Capitalization by Industry**: Horizontal bar chart to display the market cap for each industry.
- **Gross Profit by Industry**: A similar bar chart to visualize gross profit across industries.
- **EVA by Industry**: Visualizes Economic Value Added for each industry.

```python
# Plotting Market Capitalization by Industry
sorted_data = merged_data.sort_values(by='Market Capitalization ( ($ millions)', ascending=False).iloc[2:]
industry_names = sorted_data['Industry Name']
market_cap = sorted_data['Market Capitalization ( ($ millions)']

plt.figure(figsize=(16, 14))
plt.barh(industry_names, market_cap, color='skyblue')
plt.xlabel('Market Capitalization ($ millions)')
plt.ylabel('Industry Name')
plt.title('Market Capitalization by Industry')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### 7. Sorting and Filtering Data

We sort and filter industries based on financial metrics like Market Capitalization, Gross Profit, and EVA. These results are displayed for both the top-performing and bottom-performing industries.

```python
# Top 10 industries by Market Cap excluding top 2
sorted_data = merged_data.sort_values(by='Market Capitalization ( ($ millions)', ascending=False)
top_12 = sorted_data.head(12)[['Industry Name', 'Market Capitalization ( ($ millions)']]
top_10_excluding_top_2 = top_12.iloc[2:].reset_index(drop=True)
print("Top 10 Industries as per Market Capitalization:")
print(top_10_excluding_top_2.to_string(index=False))

# Bottom 10 industries by Market Cap
bottom_10 = sorted_data.tail(10)[['Industry Name', 'Market Capitalization ( ($ millions)']]
print("Bottom 10 Industries as per Market Capitalization:")
print(bottom_10.to_string(index=False))
```

### 8. Financial Condition Filtering

Industries are filtered based on combinations of high and low values for Market Capitalization, Gross Profit, and EVA.

```python
# Define quantiles
quantile_75 = merged_data.quantile(0.75)
quantile_25 = merged_data.quantile(0.25)

# High Market Cap, High Gross Profit, High EVA
high_high_high = merged_data[
    (merged_data['Market Capitalization ( ($ millions)'] > quantile_75['Market Capitalization ( ($ millions)']) &
     merged_data['Gross Profit ($ millions)'] > quantile_75['Gross Profit ($ millions)']) &
    (merged_data['EVA'] > quantile_75['EVA'])
]

# Low Market Cap, Low Gross Profit, Low EVA
low_low_low = merged_data[
    (merged_data['Market Capitalization ( ($ millions)'] < quantile_25['Market Capitalization ( ($ millions)']) &
     merged_data['Gross Profit ($ millions)'] < quantile_25['Gross Profit ($ millions)']) &
    (merged_data['EVA'] < quantile_25['EVA'])
]

# Additional conditions can be defined similarly
```

## Visualizations

- **Heatmap**: Correlation between all financial metrics.
- **Bar Graphs**: Market Capitalization, Gross Profit, and EVA by Industry.
- **Sorted Tables**: Top and bottom industries by Market Cap, Gross Profit, and EVA.

## Results

After filtering the data based on various conditions, we have identified industries that perform well or poorly in terms of financial metrics like Market Capitalization, Gross Profit, and EVA. The data and visualizations provide insights into how these industries fare in different aspects of financial performance.

## How to Run

1. Clone the repository.
2. Install the required dependencies (listed below).
3. Run the notebook to see the analysis and visualizations.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scipy

Install the necessary libraries using:

```bash
pip install pandas numpy matplotlib seaborn scipy
```

## Conclusion

This analysis provides insights into the financial health of industries in Europe by visualizing and filtering key metrics such as Market Capitalization, Gross Profit, and EVA. The correlations between different financial metrics help identify industries with strong or weak financial performance.

