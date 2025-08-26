import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import statsmodels.api as sm
import scikit_posthocs as sp
import pymannkendall as mk
from scipy.stats import skew
from linearmodels.panel import PanelOLS, RandomEffects, compare
from statsmodels.tools.tools import add_constant
from matplotlib.font_manager import FontProperties
from scipy.stats import shapiro, mannwhitneyu, kruskal, spearmanr



"""
Data Preprocessing
a. Load data
b. Convert data to the correct format
"""
# a. Load data & drop unused brand
data = pd.read_excel("datacompile.xlsx")
data = data[~data['Brand'].isin(['COUTTS', 'MSBANK'])]

# b. Convert Month to datetime
data['Month'] = pd.to_datetime(data['Month'], format="%Y-%m")



""" 
Code for:
4.1 Descriptive Performance & Usage Trends
a. Summary Statistics:
    a.1. Performance Statistics
    a.2. Usage statistics
b. Plotting Trends for each performance Metrics
    b.1. Availability plot
    b.2. Response Time plot
    b.3. Success Rate plot
c. Plotting Overall Performance Trend
d. Plotting Trend for usage
"""
# a. Summary Statistics
# a.1. Performance Statistics
selected_cols = ['Availability', 'Success Rate', 'Response Time']  
summary_stats = data[selected_cols].describe().T
summary_stats['median'] = data[selected_cols].median()
summary_stats = summary_stats[['mean', 'median', 'std', 'min', 'max']]
summary_stats = summary_stats.round(3)
print(summary_stats)

# a.2. Usage Statistics by Brand
usage_stats_by_brand = (
    data.groupby('Brand')['Total API Call Usage']
        .agg(['mean', 'median', 'std', 'min', 'max'])
        .round(3))
# Classification based on max value
def classify_usage(max_value):
    if max_value > 100_000_000:
        return 'High-volume'
    elif max_value >= 20_000_000:
        return 'Mid-volume'
    else:
        return 'Small-volume'
usage_stats_by_brand['Usage Group'] = usage_stats_by_brand['max'].apply(classify_usage)
# Sort by max value
usage_stats_by_brand = usage_stats_by_brand.sort_values('max', ascending=False)
print(usage_stats_by_brand)



# b. Plotting Trends for each performance Metrics & Usage
# b.1. Availability plot
sns.set(style="whitegrid")
# Compute volatility
brand_volatility = data.groupby('Brand')['Availability'].std().sort_values(ascending=False)
most_fluctuating = brand_volatility.head(10).index
most_stable = brand_volatility.tail(10).index
print(brand_volatility)
# Subplots
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
# Define plot function with colormap option
def plot_to_ax(brands_to_plot, ax, title, colormap):
    cmap = cm.get_cmap(colormap, len(brands_to_plot))
    for i, brand in enumerate(brands_to_plot):
        subset = data[data['Brand'] == brand]
        ax.plot(subset['Month'], subset['Availability'], label=brand, color=cmap(i))
    ax.set_title(title, fontweight='bold', fontsize='large')
    semi_bold_font = FontProperties(family='Calibri', weight='semibold', size='small')
    ax.legend(loc='lower right', fontsize='small', prop=semi_bold_font)
# Plot with separate colormaps
plot_to_ax(most_stable, axes[0], 'Less Volatile Availability (Std. Deviation ≤0.55%)', colormap='tab10')
plot_to_ax(most_fluctuating, axes[1], 'More Volatile Availability (Std. Deviation >0.55%)', colormap='tab20')
# Main y-axis label for the whole figure
fig.text(0.04, 0.5, 'Availability (%)', va='center', rotation='vertical',
         fontsize='medium', fontweight='bold')
# Subplot-specific y-axis labels
axes[0].set_ylabel('Range: 97.8–100%', fontsize='medium', fontweight='bold')
axes[1].set_ylabel('Range: 80–100%', fontsize='medium', fontweight='bold')
# Set x-axis range
start_date = pd.Timestamp('2021-04-30')
end_date = pd.Timestamp('2025-05-31')
axes[0].set_xlim(start_date, end_date)
axes[1].set_xlim(start_date, end_date)
# X-axis label for the bottom plot
axes[1].set_xlabel('Month', fontweight='bold')
plt.tight_layout(rect=[0.05, 0, 1, 1])
plt.show()


# b.2. Response Rate plot
# Compute volatility
brand_volatility = data.groupby('Brand')['Response Time'].std().sort_values(ascending=False)
most_fluctuating = brand_volatility.head(10).index
most_stable = brand_volatility.tail(10).index
print(brand_volatility)
# Subplots (no shared y-axis)
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
# Define plot function with colormap option
def plot_to_ax(brands_to_plot, ax, title, colormap):
    cmap = cm.get_cmap(colormap, len(brands_to_plot))
    for i, brand in enumerate(brands_to_plot):
        subset = data[data['Brand'] == brand]
        ax.plot(subset['Month'], subset['Response Time'], label=brand, color=cmap(i))
    ax.set_title(title, fontweight='bold', fontsize='large')
    semi_bold_font = FontProperties(family='Calibri', weight='semibold', size='small')
    ax.legend(
        loc='upper center',
        ncol=5,
        fontsize='small',
        prop=semi_bold_font,
        frameon=True)
# Plot with separate colormaps
plot_to_ax(most_stable, axes[0], 'Less Volatile Response Time (Std. Deviation ≤75.4 ms)', colormap='tab10')
plot_to_ax(most_fluctuating, axes[1], 'More Volatile Response Time (Std. Deviation >75.4 ms)', colormap='tab20')
# Main y-axis label for the whole figure
fig.text(0.04, 0.5, 'Response Time (ms)', va='center', rotation='vertical',
         fontsize='medium', fontweight='bold')
# Subplot-specific y-axis labels
axes[0].set_ylabel('Range: 0–950 ms', fontsize='medium', fontweight='bold')
axes[1].set_ylabel('Range: 0–1100 ms', fontsize='medium', fontweight='bold')
# Set x-axis range
start_date = pd.Timestamp('2021-04-30')
end_date = pd.Timestamp('2025-05-31')
axes[0].set_xlim(start_date, end_date)
axes[1].set_xlim(start_date, end_date)
# X-axis label for the bottom plot
axes[1].set_xlabel('Month', fontweight='bold')
plt.tight_layout(rect=[0.05, 0, 1, 1])
plt.show()


# b.3. Success Rate plot
# Compute volatility
brand_volatility = data.groupby('Brand')['Success Rate'].std().sort_values(ascending=False)
most_fluctuating = brand_volatility.head(10).index
most_stable = brand_volatility.tail(10).index
print(brand_volatility)
# Subplots (no shared y-axis)
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
# Define plot function with colormap option
def plot_to_ax(brands_to_plot, ax, title, colormap):
    cmap = cm.get_cmap(colormap, len(brands_to_plot))
    for i, brand in enumerate(brands_to_plot):
        subset = data[data['Brand'] == brand]
        ax.plot(subset['Month'], subset['Success Rate'], label=brand, color=cmap(i))
    ax.set_title(title, fontweight='bold', fontsize='large')
    semi_bold_font = FontProperties(family='Calibri', weight='semibold', size='small')
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 0),
        ncol=5,
        fontsize='small',
        prop=semi_bold_font,
        frameon=True)
# Plot with separate colormaps
plot_to_ax(most_stable, axes[0], 'Less Volatile Success Rate (Std. Deviation ≤0.54%)', colormap='tab10')
plot_to_ax(most_fluctuating, axes[1], 'More Volatile Success Rate (Std. Deviation >0.54%)', colormap='tab20')
# Main y-axis label for the whole figure
fig.text(0.04, 0.5, 'Success Rate (%)', va='center', rotation='vertical',
         fontsize='medium', fontweight='bold')
# Subplot-specific y-axis labels
axes[0].set_ylabel('Range: 96.8–100%', fontsize='medium', fontweight='bold')
axes[1].set_ylabel('Range: 80–100%', fontsize='medium', fontweight='bold')
# Set x-axis range
start_date = pd.Timestamp('2021-04-30')
end_date = pd.Timestamp('2025-05-31')
axes[0].set_xlim(start_date, end_date)
axes[1].set_xlim(start_date, end_date)
# Zoom in y-axis for stable brands
axes[0].set_ylim(96.8, 100.1)
axes[1].set_xlabel('Month', fontweight='bold')
# X-axis label for bottom plot
axes[1].set_xlabel('Month', fontweight='bold')
plt.tight_layout(rect=[0.05, 0, 1, 1])
plt.show()


# c. Overall Trends
monthly_avg = data.groupby('Month').agg({
    'Availability': 'mean',
    'Response Time': 'mean',
    'Success Rate': 'mean'
}).reset_index()
# Create the figure and first y-axis
fig, ax1 = plt.subplots(figsize=(14, 6))
# Plot Availability on first y-axis
color1 = 'tab:blue'
ax1.set_xlabel('Month', fontweight='semibold')
ax1.set_ylabel('Availability (%)', color=color1, fontweight='semibold')
ax1.plot(monthly_avg['Month'], monthly_avg['Availability'], color=color1, label='Availability (%)')
ax1.tick_params(axis='y', labelcolor=color1)
# Create second y-axis for Response Time
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Response Time (ms)', color=color2, fontweight='semibold')
ax2.plot(monthly_avg['Month'], monthly_avg['Response Time'], color=color2, label='Response Time (ms)')
ax2.tick_params(axis='y', labelcolor=color2)
# Create third y-axis for Success Rate
ax3 = ax1.twinx()
color3 = 'tab:green'
ax3.spines['right'].set_position(('outward', 60))  
ax3.set_ylabel('Success Rate (%)', color=color3, fontweight='semibold')
ax3.plot(monthly_avg['Month'], monthly_avg['Success Rate'], color=color3, label='Success Rate (%)')
ax3.tick_params(axis='y', labelcolor=color3)
# Set x-axis range
start_date = pd.Timestamp('2021-04-30')
end_date = pd.Timestamp('2025-05-31')
ax1.set_xlim(start_date, end_date)  
# Add title and grid
plt.title('Average API Metrics Over Time (All Brands)',fontweight='semibold')
ax1.grid(True)
fig.tight_layout()
plt.show()


# d. Usage Trend
# Figure + colormaps
usage_groups = ['High-volume', 'Mid-volume', 'Small-volume']
colormaps = {
    'High-volume': 'tab10',
    'Mid-volume': 'Accent',
    'Small-volume': 'tab10_r'}
titles = {
    'High-volume': 'High-volume (max. usage >100M)',
    'Mid-volume': 'Mid-volume (max. usage 20M–100M)',
    'Small-volume': 'Small-volume (max. usage <20M)'}
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
# Plot each group
for idx, group in enumerate(usage_groups):
    brands_in_group = usage_stats_by_brand[usage_stats_by_brand['Usage Group'] == group].index
    cmap = cm.get_cmap(colormaps[group], len(brands_in_group))
    for i, brand in enumerate(brands_in_group):
        subset = data[data['Brand'] == brand]
        axes[idx].plot(subset['Month'], subset['Total API Call Usage'], label=brand, color=cmap(i))
    axes[idx].set_title(titles[group], fontweight='bold')
    axes[idx].set_ylabel('Usage (Millions)', fontweight='bold')
    axes[idx].legend(fontsize='small', ncol=3)
    axes[idx].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1_000_000:.0f}M'))
    axes[idx].grid(True, axis='y', linestyle='--', alpha=0.35)
# X-axis limits for all subplots
start_date = pd.Timestamp('2021-04-30')
end_date   = pd.Timestamp('2025-05-31')
for ax in axes:
    ax.set_xlim(start_date, end_date)
# Common x label
axes[-1].set_xlabel('Month', fontweight='bold')
plt.tight_layout()
plt.show()



""" 
Code for:
4.2 Categorical Group Comparison
a. Bank Type Comparison
    a.1. Boxplot of TDNB & TBAF
    a.2. Shapiro Wilk Test
    a.3. Mann-Whitney U & Cliff's Delta
b. Regulatory Period Comparison
    b.1. Boxplot of each regulatory period
    b.2. Kruskal Wallis & Eta-square
    b.3. Pairwise Dunn's test
"""


# a. Bank type comparison
# a.1. Boxplot of TDNB & TBAF
# Availability by Bank Type
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
ax = sns.boxplot(
    x='Bank Type', y='Availability', data=data,
    palette="Set2", showmeans=True, fliersize=5,
    meanprops={"marker": "D", "markerfacecolor": "lightblue", "markeredgecolor": "black"})
plt.title('API Availability by Bank Type', fontweight="semibold")
plt.ylabel('Availability (%)', fontweight="semibold")
plt.xlabel('Bank Type', fontweight='semibold')
plt.ylim(96, 100.1)
# Mean text box
group_means = data.groupby('Bank Type')['Availability'].mean().round(2)
mean_text = "Mean of Each Bank Type:\n" + '\n'.join([f'{k}: {v:.2f}%' for k, v in group_means.items()])
props = dict(boxstyle='square', facecolor='white', edgecolor='grey', linewidth=1)
ax.text(0.75, 0.08, mean_text, transform=ax.transAxes,
        fontsize=11, fontweight='semibold', ha='center', va='bottom', bbox=props)
plt.tight_layout()
plt.show()


# Success Rate by Bank Type
plt.figure(figsize=(8, 5))
ax = sns.boxplot(
    x='Bank Type', y='Success Rate', data=data,
    palette="Set2", showmeans=True, fliersize=5,
    meanprops={"marker": "D", "markerfacecolor": "lightblue", "markeredgecolor": "black"})
plt.title('API Success Rate by Bank Type', fontweight="semibold")
plt.ylabel('Success Rate (%)', fontweight="semibold")
plt.xlabel('Bank Type', fontweight='semibold')
plt.ylim(89, 100.1)
# Mean text box
group_means = data.groupby('Bank Type')['Success Rate'].mean().round(2)
mean_text = "Mean of Each Bank Type:\n" + '\n'.join([f'{k}: {v:.2f}%' for k, v in group_means.items()])
props = dict(boxstyle='square', facecolor='white', edgecolor='grey', linewidth=1)
ax.text(0.75, 0.08, mean_text, transform=ax.transAxes,
        fontsize=11, fontweight='semibold', ha='center', va='bottom', bbox=props)
plt.tight_layout()
plt.show()


# Response Time by Bank Type
plt.figure(figsize=(8, 5))
ax = sns.boxplot(
    x='Bank Type', y='Response Time', data=data,
    palette="Set2", showmeans=True, fliersize=5,
    meanprops={"marker": "D", "markerfacecolor": "lightblue", "markeredgecolor": "black"})

plt.title('API Response Time by Bank Type', fontweight="semibold")
plt.ylabel('Response Time (ms)', fontweight="semibold")
plt.xlabel('Bank Type', fontweight='semibold')
# Mean text box
group_means = data.groupby('Bank Type')['Response Time'].mean().round(2)
mean_text = "Mean of Each Bank Type:\n" + '\n'.join([f'{k}: {v:.2f} (ms)' for k, v in group_means.items()])
props = dict(boxstyle='square', facecolor='white', edgecolor='grey', linewidth=1)
ax.text(0.75, 0.83, mean_text, transform=ax.transAxes,
        fontsize=11, fontweight='semibold', ha='center', va='bottom', bbox=props)
plt.tight_layout()
plt.show()
data.groupby('Bank Type')['Response Time'].agg(['mean', 'median'])


# a.2. Shapiro Wilk Test
# Check normality for Availability
stat, p = shapiro(data['Availability'])
print(f'Availability Normality p = {p:.2e}')
# Check for Success Rate
stat, p = shapiro(data['Success Rate'])
print(f'Success Rate Normality p = {p:.2e}')
# Check for Response Time
stat, p = shapiro(data['Response Time'])
print(f'Response Time Normality p = {p:.2e}')


# a.3. Mann-Whitney U & Cliff's Delta
# Split data by Bank Type for Mann-Whitney test
t_banks = data[data['Bank Type'] == 'TBAF']
c_banks = data[data['Bank Type'] == 'TDNB']
# Availability
stat, p = mannwhitneyu(t_banks['Availability'], c_banks['Availability'])
print(f"Mann-Whitney U (Availability): p = {p:.2e}")
# Success Rate
stat, p = mannwhitneyu(t_banks['Success Rate'], c_banks['Success Rate'])
print(f"Mann-Whitney U (Success Rate): p = {p:.5f}")
# Response Time
stat, p = mannwhitneyu(t_banks['Response Time'], c_banks['Response Time'])
print(f"Mann-Whitney U (Response Time): p = {p:.5f}")
# Cliff's Delta
def cliffs_delta(a, b):
    """
    Calculate Cliff's Delta effect size between two groups.
    Parameters:
    a (array-like): Group 1 values
    b (array-like): Group 2 values
    Returns:
    delta (float): Cliff's Delta value
    magnitude (str): Effect size magnitude (negligible, small, medium, large)
    """
    n = len(a)
    m = len(b)
    more = sum(x > y for x in a for y in b)
    less = sum(x < y for x in a for y in b)
    delta = (more - less) / (n * m)
    # Interpret magnitude
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        magnitude = "negligible"
    elif abs_delta < 0.33:
        magnitude = "small"
    elif abs_delta < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"  
    return delta, magnitude
# Availability between TBAF and TDNB
delta, magnitude = cliffs_delta(t_banks['Availability'], c_banks['Availability'])
print(f"Cliff's Delta (Availability): {delta} ({magnitude})")
# Success Rate
delta, magnitude = cliffs_delta(t_banks['Success Rate'], c_banks['Success Rate'])
print(f"Cliff's Delta (Success Rate): {delta:} ({magnitude})")
# Response Time
delta, magnitude = cliffs_delta(t_banks['Response Time'], c_banks['Response Time'])
print(f"Cliff's Delta (Response Time): {delta} ({magnitude})")


# b. Regulatory Period Comparison
# b.1. Boxplot of each regulatory period
# Availability by Time Period
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
ax = sns.boxplot(
    x='Time Period', y='Availability', data=data,
    palette="Set2", showmeans=True, fliersize=5,
    meanprops={"marker": "D", "markerfacecolor": "lightblue", "markeredgecolor": "black"})
plt.title('API Availability by Regulatory Period', fontweight="semibold")
plt.ylabel('Availability (%)', fontweight="semibold")
plt.xlabel('Regulatory Period', fontweight="semibold")
plt.xticks(fontweight="semibold")
plt.ylim(96, 100.1)
# Mean text box
group_means = data.groupby('Time Period')['Availability'].mean().round(2)
mean_text = "Mean of Each Reg. Period:\n" + '\n'.join([f'{k}: {v:.2f}%' for k, v in group_means.items()])
props = dict(boxstyle='square', facecolor='white', edgecolor='grey', linewidth=1)
ax.text(0.83, 0.04, mean_text, transform=ax.transAxes,
        fontsize=11, fontweight='semibold', ha='center', va='bottom', bbox=props)
plt.tight_layout()
plt.show()

# Success Rate
plt.figure(figsize=(8, 5))
ax = sns.boxplot(
    x='Time Period', y='Success Rate', data=data,
    palette="Set2", showmeans=True, fliersize=5,
    meanprops={"marker": "D", "markerfacecolor": "lightblue", "markeredgecolor": "black"})
plt.title('API Success Rate by Regulatory Period', fontweight="semibold")
plt.ylabel('Success Rate (%)', fontweight="semibold")
plt.xlabel('Regulatory Period', fontweight='semibold')
plt.xticks(fontweight="semibold")
plt.ylim(89, 100.1)
# Mean text box
group_means = data.groupby('Time Period')['Success Rate'].mean().round(2)
mean_text = "Mean of Each Reg. Period:\n" + '\n'.join([f'{k}: {v:.2f}%' for k, v in group_means.items()])
props = dict(boxstyle='square', facecolor='white', edgecolor='grey', linewidth=1)
ax.text(0.75, 0.08, mean_text, transform=ax.transAxes,
        fontsize=11, fontweight='semibold', ha='center', va='bottom', bbox=props)
plt.tight_layout()
plt.show()

# Response Time
plt.figure(figsize=(8, 5))
ax = sns.boxplot(
    x='Time Period', y='Response Time', data=data,
    palette="Set2", showmeans=True, fliersize=5,
    meanprops={"marker": "D", "markerfacecolor": "lightblue", "markeredgecolor": "black"})
plt.title('API Response Time by Regulatory Period', fontweight="semibold")
plt.ylabel('Response Time (ms)', fontweight="semibold")
plt.xlabel('Regulatory Period', fontweight='semibold')
plt.xticks(fontweight="semibold")
# Mean text box
group_means = data.groupby('Time Period')['Response Time'].mean().round(2)
mean_text = "Mean of Each Reg. Period:\n" + '\n'.join([f'{k}: {v:.2f} (ms)' for k, v in group_means.items()])
props = dict(boxstyle='square', facecolor='white', edgecolor='grey', linewidth=1)
ax.text(0.5, 0.75, mean_text, transform=ax.transAxes,
        fontsize=11, fontweight='semibold', ha='center', va='bottom', bbox=props)
plt.tight_layout()
plt.show()


# b.2. Kruskal Wallis & Eta-square
# Split data by Time Period
pre2023 = data[data['Time Period'] == 'Pre-2023']
transition = data[data['Time Period'] == 'Transition']
post2023 = data[data['Time Period'] == 'Post-2023']
# Compute Eta-squared & Kruskal-Wallis
def eta_squared_kruskal(groups, H_stat):
    n = sum(len(g) for g in groups)
    k = len(groups)
    eta_sq = (H_stat - k + 1) / (n - k)
    return eta_sq
# Loop over metrics
metrics = ['Availability', 'Success Rate', 'Response Time', ]
for metric in metrics:
    # Split data by Time Period for the current metric
    groups = [
        pre2023[metric],
        transition[metric],
        post2023[metric]]    
    # Run Kruskal-Wallis test
    H_stat, p_value = kruskal(*groups)    
    # Compute Eta-squared
    eta_sq = eta_squared_kruskal(groups, H_stat)    
    # Print results
    print(f"\n=== {metric} ===")
    print(f"Kruskal-Wallis H = {H_stat}, p = {p_value:.2e}")
    print(f"Eta-squared (η²) = {eta_sq}")
        # Interpret effect size
    if eta_sq < 0.01:
        effect = "Negligible"
    elif eta_sq < 0.06:
        effect = "Small"
    elif eta_sq < 0.14:
        effect = "Medium"
    else:
        effect = "Large"
    print(f"Effect size: {effect}")

# b.3. Pairwise Dunn's test
# Loop through all metrics
metrics = ['Availability', 'Response Time', 'Success Rate']
for metric in metrics:
    print(f"\n=== Dunn's Test for {metric} ===")
    # Dunn’s test with Bonferroni correction for multiple comparisons
    dunn_results = sp.posthoc_dunn(
        data,
        val_col=metric,
        group_col='Time Period',
        p_adjust='bonferroni')
    print(dunn_results)



""" 
Code for:
4.3 Performance Trend Analysis
a. Trend by Brand
    a.1. Mann Kendall Test
    a.2. Heatmap
b. Trend by Bank Type
    b.1. Mann Kendall Test
"""



# 4.3 Performance Trend Analysis
# a. Trend by Brand
# a.1. Mann Kendall Test
# Sort data
data = data.sort_values(["Brand", "Month"])
# Plot for each brand
brands = data["Brand"].unique()
# Prepare summary dataframe
summary = []
for brand in brands:
    brand_df = data[data["Brand"] == brand]
    brand_df = brand_df.sort_values("Month")
    x_numeric = (brand_df["Month"] - brand_df["Month"].min()).dt.days
    for metric in metrics:
        # Mann-Kendall test
        mk_result = mk.original_test(brand_df[metric])
        mk_trend = mk_result.trend
        mk_tau = mk_result.Tau
        mk_p = mk_result.p
        # Classification
        if mk_p < 0.05:
            if metric == "Response Time":
                # Reverse logic for Response Time
                if mk_trend == "decreasing":
                    classification = "Improving"
                elif mk_trend == "increasing":
                    classification = "Declining"
                else:
                    classification = "Stable"
            else:
                # Normal logic for Availability and Success Rate
                if mk_trend == "increasing":
                    classification = "Improving"
                elif mk_trend == "decreasing":
                    classification = "Declining"
                else:
                    classification = "Stable"
        else:
            classification = "Stable"       
        summary.append({
            "Brand": brand,
            "Metric": metric,
            "Mann-Kendall Trend": mk_trend,
            "MK Tau": round(mk_tau, 4),
            "MK p-value": round(mk_p, 6),
            "Classification": classification})
# Convert to DataFrame
summary_df = pd.DataFrame(summary)
print(summary_df)


# a.2. Heatmap
# Define trend strength labeling 
def classify_trend(trend, tau, p, metric):
    if p < 0.05:
        # Determine strength
        if abs(tau) < 0.3:
            strength = "Weak"
        elif abs(tau) < 0.5:
            strength = "Moderate"
        else:
            strength = "Strong"
        # Label according to metric type
        if metric == "Response Time":
            if trend == "decreasing":
                return f"Improving ({strength})"
            elif trend == "increasing":
                return f"Declining ({strength})"
            else:
                return "Stable"
        else:
            if trend == "increasing":
                return f"Improving ({strength})"
            elif trend == "decreasing":
                return f"Declining ({strength})"
            else:
                return "Stable"
    else:
        return "Stable"
# Apply classification 
summary_df["Trend_Label"] = summary_df.apply(
    lambda row: classify_trend(
        row["Mann-Kendall Trend"], row["MK Tau"], row["MK p-value"], row["Metric"]),
    axis=1)
# Pivot table 
heatmap_data = summary_df.pivot(index="Brand", columns="Metric", values="Trend_Label")
# Map trend to numeric scale for coloring 
def map_to_numeric(label):
    mapping = {
        "Improving (Weak)": 1,
        "Improving (Moderate)": 2,
        "Improving (Strong)": 3,
        "Declining (Weak)": -1,
        "Declining (Moderate)": -2,
        "Declining (Strong)": -3,
        "Stable": 0,}
    return mapping.get(label, 0)
heatmap_numeric = heatmap_data.copy()
# Reorder columns manually
desired_order = ["Availability", "Success Rate", "Response Time"]
heatmap_data = heatmap_data[desired_order]
heatmap_numeric = heatmap_numeric[desired_order]
for col in heatmap_numeric.columns:
    heatmap_numeric[col] = heatmap_data[col].apply(map_to_numeric)
# Custom color palette: red grey green with strength
cmap = sns.color_palette(
    ["#b30000",  # -3: Declining (Strong)
     "#e74c3c",  # -2: Declining (Moderate)
     "#f5b7b1",  # -1: Declining (Weak)
     "#ecf0f1",  #  0: Stable
     "#aed6f1",  #  1: Improving (Weak)
     "#5dade2",  #  2: Improving (Moderate)
     "#154360",  #  3: Improving (Strong)
])
# Plot 
fig, ax = plt.subplots(figsize=(10, 12))
sns.heatmap(
    heatmap_numeric,
    annot=heatmap_data,
    fmt="",
    cmap=cmap,
    cbar=False,
    center=0,
    linewidths=0.5,
    linecolor="white",
    ax=ax,)
# Loop over annotation texts
for i, text in enumerate(ax.texts):
    # Figure out which label this annotation belongs to
    row = i // heatmap_data.shape[1]
    col = i % heatmap_data.shape[1]
    row_text = heatmap_data.values[row][col]
    # Set text color
    color = "white" if row_text == "Improving (Strong)" else "black"
    text.set_color(color)
    # Make text semi-bold
    text.set_fontweight("semibold")
# Title and labels (semi-bold)
plt.title(
    "Brand × Metric Trend Heatmap (Strength-Sensitive)",
    fontsize=16,
    fontweight="semibold",)
plt.ylabel("Brand", fontweight="semibold")
plt.xlabel("Metric", fontweight="semibold")
# Tick labels semi-bold
plt.xticks(fontweight="semibold")
plt.yticks(fontweight="semibold")
plt.tight_layout()
plt.show()


# b. Trend by Bank Type
# b.1. Mann Kendall Test
for metric in ['Availability',  'Success Rate', 'Response Time']:
    for bank_type in ['TBAF', 'TDNB']:
        series = data[data['Bank Type'] == bank_type].sort_values('Month')[metric]
        result = mk.original_test(series)
        print(f"{metric} - {bank_type}: {result.trend}, p = {result.p:.2e}, tau:{result.Tau:.3f}")


# b.2. Trend overtime by Bank type      
# Sort for plotting
data.sort_values(by="Month", inplace=True)
# Create Year-Month for cleaner x-axis
data["YearMonth"] = data["Month"].dt.to_period("M").astype(str)
# Plot Trends for Each Bank
metrics = ["Availability", "Response Time", "Success Rate"]
# Compare TBAF vs TDNB
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="Month", y=metric, hue="Bank Type", ci=None, marker="o")
    plt.title(f"{metric} Over Time: TBAF vs TDNB", fontsize=16, fontweight = 'bold')
    plt.xlabel("Month")
    plt.ylabel(metric)
    plt.xticks(rotation=0, fontweight='semibold')
    plt.yticks(fontweight='semibold')
    legend = plt.legend(title="Bank Type")
    legend.get_title().set_fontweight("semibold")   
    for text in legend.get_texts():                 
        text.set_fontweight("semibold")
    plt.tight_layout()
    plt.show()
    


""" 
Code for:
4.4 Compliance with Performance Standards (Availability and Response Time)
    a. Overtime compliance
    b. brand Heatmap
"""

# 4.4 Compliance with Performance Standards (Availability and Response Time)
# a. Overtime Compliance

# Create a Quarter column
data_copy = data.copy()
data_copy['Quarter'] = data_copy['Month'].dt.to_period('Q')
# Compute average availability per brand per quarter
availability_check = data_copy.groupby(['Brand', 'Quarter'])['Availability'].mean().reset_index()
availability_check['Meets_Standard'] = availability_check['Availability'] >= 99.5
# average compliance per quarter (across brands)
quarterly_avg_availability = availability_check.groupby('Quarter')['Meets_Standard'].mean().reset_index()
quarterly_avg_availability['Compliance_%'] = (quarterly_avg_availability['Meets_Standard'] * 100).round(1)
# Create month period column
data_copy['MonthPeriod'] = data_copy['Month'].dt.to_period('M')
# Mark compliance
data_copy['Meets_Standard_RT'] = data_copy['Response Time'] <= 750
# Average per brand per month
monthly_check = data_copy.groupby(['Brand', 'MonthPeriod'])['Meets_Standard_RT'].mean().reset_index()
#average across brands for each month
monthly_avg_response = monthly_check.groupby('MonthPeriod')['Meets_Standard_RT'].mean().reset_index()
monthly_avg_response['Compliance_%'] = (monthly_avg_response['Meets_Standard_RT'] * 100).round(1)
fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
# Plot availability trend
ax[0].plot(quarterly_avg_availability['Quarter'].astype(str), quarterly_avg_availability['Compliance_%'], marker='o')
ax[0].set_title('Availability Compliance (Quarterly)', fontweight ='semibold', fontsize = 'large')
ax[0].set_ylabel('Avg. Compliance %', fontweight ='semibold')
ax[0].set_ylim(50, 100)
ax[0].grid(True)
# Plot response time trend
ax[1].plot(monthly_avg_response['MonthPeriod'].astype(str), monthly_avg_response['Compliance_%'], marker='o')
ax[1].set_title('Response Time Compliance (Monthly)', fontweight ='semibold', fontsize = 'large')
ax[1].set_ylabel('Avg. Compliance %', fontweight ='semibold')
ax[1].set_xlabel('Time', fontweight ='semibold')
ax[1].set_ylim(88, 101)
ax[1].grid(True)
ax[0].tick_params(axis='x', rotation=90)
ax[1].tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.show()
print(quarterly_avg_availability)
print(monthly_avg_response)

# b. brand heatmap
# Create a Quarter column
data_copy = data.copy()
data_copy['Quarter'] = data_copy['Month'].dt.to_period('Q')
# 1. Availability Compliance (Quarterly) 
availability_check = data_copy.groupby(['Brand', 'Quarter'])['Availability'].mean().reset_index()
availability_check['Meets_Standard'] = availability_check['Availability'] >= 99.5
# Calculate compliance rate per brand (as percentage)
availability_summary = availability_check.groupby('Brand')['Meets_Standard'].mean().reset_index()
availability_summary['Availability_Compliance_%'] = (availability_summary['Meets_Standard'] * 100).round(1)
availability_summary.drop(columns='Meets_Standard', inplace=True)
#2. Response Time Compliance (Monthly) 
data_copy['Meets_Standard'] = data_copy['Response Time'] <= 750
response_summary = data_copy.groupby('Brand')['Meets_Standard'].mean().reset_index()
response_summary['ResponseTime_Compliance_%'] = (response_summary['Meets_Standard'] * 100).round(1)
response_summary.drop(columns='Meets_Standard', inplace=True)
# 3. Merge both summaries 
compliance_summary = pd.merge(availability_summary, response_summary, on='Brand')
# Display
print("\nOpen Banking API Compliance Summary:\n")
print(compliance_summary.sort_values(by='Brand').to_string(index=False))
# Set Brand as index for heatmap format
heatmap_data = compliance_summary.rename(columns={
    'Availability_Compliance_%': 'Availability',
    'ResponseTime_Compliance_%': 'Response Time'
}).set_index('Brand')
# Create the heatmap
plt.figure(figsize=(10, 14))
# Define semi-bold font properties
semibold_font = FontProperties(weight='bold')
# Pass it into heatmap
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".1f",
    cmap="RdBu",
    cbar_kws={'label': 'Compliance %'},
    annot_kws={'fontproperties': semibold_font})
plt.ylabel("Brand", fontweight="semibold")
plt.tight_layout()
plt.show()



""" 
Code for:
4.5 Performance Composite scoring
    a. calculate composite scoring
    b. plot performance ranking
"""

# 4.5 Performance Composite scoring
# a. calculate composite scoring
# Load data
df = data.copy()
df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
df = df.dropna(subset=["Month"])
# Prepare year and decay
df["Year"] = df["Month"].dt.year
most_recent_year = df["Year"].max()
# Time weight function
def calculate_time_weights(years, gamma):
    return np.exp(-gamma * (most_recent_year - years))
# Store results for each gamma
results = []
for gamma in [0.3, 0.5, 0.7]:
    df_temp = df.copy()
    df_temp["Time_Weight"] = calculate_time_weights(df_temp["Year"], gamma)
    # Apply weights
    df_temp["Availability_weighted"] = df_temp["Availability"] * df_temp["Time_Weight"]
    df_temp["SuccessRate_weighted"] = df_temp["Success Rate"] * df_temp["Time_Weight"]
    df_temp["ResponseTime_weighted"] = df_temp["Response Time"] * df_temp["Time_Weight"]
    # Aggregate by Brand only
    weighted_df = (
        df_temp.groupby("Brand")
        .agg({
            "Availability_weighted": "sum",
            "SuccessRate_weighted": "sum",
            "ResponseTime_weighted": "sum",
            "Time_Weight": "sum"
        })
        .reset_index())
    # Compute weighted average
    weighted_df["Availability"] = weighted_df["Availability_weighted"] / weighted_df["Time_Weight"]
    weighted_df["Success Rate"] = weighted_df["SuccessRate_weighted"] / weighted_df["Time_Weight"]
    weighted_df["Response Time"] = weighted_df["ResponseTime_weighted"] / weighted_df["Time_Weight"]
    # Rankings
    weighted_df["Availability_Rank"] = weighted_df["Availability"].rank(method="min", ascending=False)
    weighted_df["SuccessRate_Rank"] = weighted_df["Success Rate"].rank(method="min", ascending=False)
    weighted_df["ResponseTime_Rank"] = weighted_df["Response Time"].rank(method="min", ascending=True)
    # Normalize
    weighted_df["Availability_norm"] = (
        (weighted_df["Availability"] - weighted_df["Availability"].min()) /
        (weighted_df["Availability"].max() - weighted_df["Availability"].min()))
    weighted_df["SuccessRate_norm"] = (
        (weighted_df["Success Rate"] - weighted_df["Success Rate"].min()) /
        (weighted_df["Success Rate"].max() - weighted_df["Success Rate"].min()))
    weighted_df["ResponseTime_norm"] = (
        (weighted_df["Response Time"] - weighted_df["Response Time"].min()) /
        (weighted_df["Response Time"].max() - weighted_df["Response Time"].min()))
    # Invert response time
    weighted_df["ResponseTime_inverted"] = 1 - weighted_df["ResponseTime_norm"]
    # Composite score
    weighted_df["Composite_Score"] = (
        weighted_df["Availability_norm"] +
        weighted_df["SuccessRate_norm"] +
        weighted_df["ResponseTime_inverted"]
    ) / 3
    # Tier classification
    quantiles = weighted_df["Composite_Score"].quantile([0.25, 0.75])
    low_threshold = quantiles[0.25]
    high_threshold = quantiles[0.75]
    def assign_tier(score):
        if score <= low_threshold:
            return "Low"
        elif score >= high_threshold:
            return "High"
        else:
            return "Medium"
    weighted_df["Performance_Tier"] = weighted_df["Composite_Score"].apply(assign_tier)
    weighted_df["Gamma"] = gamma
    results.append(weighted_df)
# Combine all gamma values
final_df = pd.concat(results)
final_df.sort_values(by=["Gamma", "Composite_Score"], ascending=[True, False], inplace=True)
# Preview top 30
print(final_df[["Brand", "Gamma", "Composite_Score", "Performance_Tier",
                "Availability_Rank", "SuccessRate_Rank", "ResponseTime_Rank"]].head(30))


# b. plot performance ranking
# Define custom colors
palette = {
    0.3: "#8ecae6",
    0.5: "#219ebc",
    0.7: "#184e77",
}
# Ensure Brand order (High → Low score)
brand_order = final_df.groupby("Brand")["Composite_Score"].mean().sort_values(ascending=False).index
final_df["Brand"] = pd.Categorical(final_df["Brand"], categories=brand_order, ordered=True)
final_df = final_df.sort_values("Brand", ascending=True)  # For top-down bar chart
# Create figure
fig, ax = plt.subplots(figsize=(8, 12))
# Plot horizontal bar chart
sns.barplot(
    data=final_df,
    y="Brand",
    x="Composite_Score",
    hue="Gamma",
    palette=palette,
    edgecolor="black",
    linewidth=0.5,
    ax=ax
)
# Add annotations (score on bars)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=8, padding=3)
# Get total number of unique brands
n_brands = final_df["Brand"].nunique()
# Define performance regions (top 6 = high, next 8 = medium, rest = low)
high_end = 6
med_end = 6 + 8
# Background color shading (behind bars)
ax.axhspan(-0.5, high_end - 0.5, facecolor="#d3f9d8", alpha=0.3, zorder=0)
ax.axhspan(high_end - 0.5, med_end - 0.5, facecolor="#fffac8", alpha=0.3, zorder=0)
ax.axhspan(med_end - 0.5, n_brands - 0.5, facecolor="#ffd6d6", alpha=0.3, zorder=0)
# Add vertical rotated tier labels
ax.text(0.915, high_end / 2 - 0.5, "High Performance", fontsize=10, fontweight='bold', rotation=-90, va='center')
ax.text(0.915, (high_end + med_end) / 2 - 0.5, "Medium Performance", fontsize=10, fontweight='bold', rotation=-90, va='center')
ax.text(0.915, (med_end + n_brands) / 2 - 0.5, "Low Performance", fontsize=10, fontweight='bold', rotation=-90, va='center')
# Aesthetic settings
plt.title("Composite API Performance Score by Brand and Time Decay (Gamma)", fontsize=13, fontweight='bold')
plt.xlabel("Composite Score", fontsize=11, fontweight='bold')
plt.ylabel("Brand", fontsize=11, fontweight='bold')
plt.xlim(0, 0.90)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title="Gamma", loc="lower right")
plt.tight_layout()
plt.show()




""" 
Code for:
4.6. Correlation Analysis Performance vs Usage
"""

# 4.6 Correlation Analysis Performance vs Usage
correlation_summary = []
# Loop over brands
brands = data["Brand"].unique()
for brand in brands:
    brand_df = data[data["Brand"] == brand]
    metrics = ["Availability", "Success Rate", "Response Time"]
    titles = ["Availability (%)", "Success Rate (%)", "Response Time (ms)"]
    for i, metric in enumerate(metrics):
        # Correlation
        pearson_corr, _ = pearsonr(brand_df[metric], brand_df["Total API Call Usage"])
        spearman_corr, _ = spearmanr(brand_df[metric], brand_df["Total API Call Usage"])
        # Append to summary
        correlation_summary.append({
            "Brand": brand,
            "Metric": titles[i],
            "Spearman Correlation": round(spearman_corr, 3)})    
# Convert summary to DataFrame
summary_df = pd.DataFrame(correlation_summary)
# Display summary
print(summary_df)



""" 
Code for:
4.7 Panel Data Regression
    a. Testing Regression Model 
        a.1. Fixed Effects Model
        a.2. Random Effects Model
        a.3. Hausman Test
    b. Fixed Effects 2 way model
        b.1. Original 2 way model
        b.2. Updated 2 way Fixed Effects Mode
    c. Create month plot
"""


# a. Testing Regression Model 
# Log-transform Total API Call Usage
df = data.copy()
df['Log_Usage'] = np.log(df['Total API Call Usage'])
# Remove zero or negative values to avoid log errors
df = df[(df['Total API Call Usage'] > 0) & (df['Response Time'] > 0)]
# Compute skewness
skewness_results = {
    "Original_Total_API_Call_Usage": skew(df['Total API Call Usage']),
    "Log_Total_API_Call_Usage": skew(np.log(df['Total API Call Usage'])),}
# Display results
for label, val in skewness_results.items():
    print(f"{label}: {val:.4f}")
# Plot only Total API Call Usage
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['Total API Call Usage'], kde=True, ax=axes[0])
axes[0].set_title('Original Total API Call Usage')
sns.histplot(df['Log_Usage'], kde=True, ax=axes[1])
axes[1].set_title('Log-Transformed Total API Call Usage')
axes[0].set_title('Original Total API Call Usage', fontweight='bold')
axes[1].set_title('Log-Transformed Total API Call Usage', fontweight='bold')
plt.tight_layout()
plt.show()
df['Time'] = pd.to_datetime(df['Month'], errors='coerce')
df = df.dropna(subset=['Time'])
# Set multi-index: Brand × Time
df = df.set_index(['Brand', 'Time'])


# a.1. Fixed Effects Model
fe_model = PanelOLS.from_formula(
    formula="Log_Usage ~ Q('Response Time') + Availability + Q('Success Rate') + EntityEffects",
    data=df)
fe_results = fe_model.fit()
print("\nFixed Effects Model Results:\n", fe_results.summary)
# Print full p-values from the fixed effects regression
print("Exact p-values:")
print(fe_results.pvalues)
# Print exact p-values in scientific notation with high precision
for var, pval in fe_results.pvalues.items():
    print(f"{var}: p = {pval:.5e}")

df_ow = data.copy()
df_ow['Log_Usage'] = np.log(df_ow['Total API Call Usage'])
df_ow = df_ow.dropna(subset=['Brand', 'Month', 'Log_Usage',
                             'Response Time', 'Availability', 'Success Rate'])
df_ow = df_ow.set_index(['Brand', 'Month'])

# Estimate one-way FE (brand) using formula (your style)
fe_model = PanelOLS.from_formula(
    "Log_Usage ~ Q('Response Time') + Availability + Q('Success Rate') + EntityEffects",
    data=df_ow
)
fe_results = fe_model.fit()
print("\nOne-way FE (Brand) Results:\n", fe_results.summary)

# Build a clean summary dataframe for key regressors
params = fe_results.params
bse    = fe_results.std_errors
tstats = fe_results.tstats
pvals  = fe_results.pvalues

# Labels as they appear when using Q(...) in the formula
main_vars = ["Q('Response Time')", "Availability", "Q('Success Rate')"]

# If your labels differ (e.g., you didn’t use Q()), fallback:
if not set(main_vars).issubset(params.index):
    main_vars = ['Response Time', 'Availability', 'Success Rate']

summary_ow = pd.DataFrame({
    'Coefficient': params[main_vars],
    'Std. Error': bse[main_vars],
    't-Statistic': tstats[main_vars],
    'p-Value': pvals[main_vars]
})

# Append model metadata rows (same style as your two-way table)
meta = pd.DataFrame({
    'Coefficient': ['—', '—', fe_results.nobs],
    'Std. Error':  ['—', '—', '—'],
    't-Statistic': ['—', '—', 'R² (Within)'],
    'p-Value':     ['—', '—', (fe_results.rsquared_within)]
}, index=['Entity Fixed Effects', 'Time Fixed Effects', 'Observations'])

# Set FE flags
meta.loc['Entity Fixed Effects', 'Coefficient'] = 'Yes'
meta.loc['Time Fixed Effects',   'Coefficient'] = 'No'

summary_ow = pd.concat([summary_ow, meta])
print(summary_ow)




# a.2. Random Effects Model
re_model = RandomEffects.from_formula(
    formula="Log_Usage ~ Q('Response Time') + Availability + Q('Success Rate')",
    data=df)
re_results = re_model.fit()
print("\nRandom Effects Model Results:\n", re_results.summary)


# a.3. Hausman Test
# Run comparison and Hausman-style test
comparison = compare({'FE': fe_results, 'RE': re_results})
print("\nHausman-style Comparison:\n", comparison)

import numpy as np
from scipy import stats

# Extract coefficients (excluding entity effects for FE)
fe_params = fe_results.params
re_params = re_results.params

# Align coefficients (drop intercept/entity dummies if present)
common_coef = fe_params.index.intersection(re_params.index)
b_fe = fe_params[common_coef]
b_re = re_params[common_coef]

# Extract covariance matrices for common coefficients
V_fe = fe_results.cov.loc[common_coef, common_coef]
V_re = re_results.cov.loc[common_coef, common_coef]

# Compute difference
b_diff = b_fe - b_re
V_diff = V_fe - V_re

# Hausman test statistic
chi2_stat = float(b_diff.T @ np.linalg.inv(V_diff) @ b_diff)
df = len(b_diff)  # degrees of freedom
p_value = 1 - stats.chi2.cdf(chi2_stat, df)

print(f"Hausman test statistic: {chi2_stat:.4f}")
print(f"Degrees of freedom: {df}")
print(f"P-value: {p_value:.5e}")



# b.1. Original 2 way model
# Drop rows with missing values
df = data.copy()
df['Log_Usage'] = np.log(df['Total API Call Usage'])
df = df.dropna(subset=[
    'Brand', 'Month', 'Log_Usage', 'Response Time',
    'Availability', 'Success Rate'])
# Set panel index: Brand = entity, Month = time
df = df.set_index(['Brand', 'Month'])
# Create time fixed effects (month dummies), drop first to avoid dummy trap
month_dummies = pd.get_dummies(df.index.get_level_values('Month'), prefix='Month', drop_first=True)
month_dummies.index = df.index  # Align index with df
df = df.join(month_dummies)
# Define regressors
X_vars = ['Response Time', 'Availability', 'Success Rate'] + list(month_dummies.columns)
X = sm.add_constant(df[X_vars])
y = df['Log_Usage']
# Run fixed effects model with entity (brand) and time (month) fixed effects
model = PanelOLS(y, X, entity_effects=True)
results = model.fit()
print(results.summary)
# Extract key variables only (not month dummies)
params = results.params
bse = results.std_errors
tstats = results.tstats
pvals = results.pvalues
# Filter to keep only main variables (not the Month dummies)
main_vars = ['Response Time', 'Availability', 'Success Rate']
summary_df = pd.DataFrame({
    'Coefficient': params[main_vars],
    'Std. Error': bse[main_vars],
    't-Statistic': tstats[main_vars],
    'p-Value': pvals[main_vars]
})
# Add model metadata 
metadata = pd.DataFrame({
    'Coefficient': ['—', '—', '—'],
    'Std. Error': ['—', '—', '—'],
    't-Statistic': ['—', '—', '—'],
    'p-Value': ['—', '—', '—']
}, index=['Entity Fixed Effects', 'Time Fixed Effects', 'Observations'])
# Append metadata
summary_df = pd.concat([summary_df, metadata])
summary_df.loc['Observations', 'Coefficient'] = results.nobs
summary_df.loc['Observations', 't-Statistic'] = 'R² (Within)'
summary_df.loc['Observations', 'p-Value'] = round(results.rsquared_within, 3)
print(summary_df)


# b.2. Updated 2 way Fixed Effects Mode
df = data.copy()
df['Log_Usage'] = np.log(df['Total API Call Usage'])
df = df.dropna(subset=[
    'Brand', 'Month', 'Log_Usage', 'Response Time',
    'Availability', 'Success Rate'])
# Set panel index: Brand = entity, Month = time
df = df.set_index(['Brand', 'Month'])
# Create month dummies (time fixed effects)
month_dummies = pd.get_dummies(df.index.get_level_values('Month'),
                               prefix='Month', drop_first=True)
month_dummies.index = df.index
df = df.join(month_dummies)
# Define regressors (no Success Rate)
X_vars = ['Response Time', 'Availability'] + list(month_dummies.columns)
X = sm.add_constant(df[X_vars])
y = df['Log_Usage']
# Two-way Fixed Effects model
model = PanelOLS(y, X, entity_effects=True)
results = model.fit()
print(results.summary)
# Extract main effects only
main_vars = ['Response Time', 'Availability']
summary_df = pd.DataFrame({
    'Coefficient': results.params[main_vars],
    'Std. Error': results.std_errors[main_vars],
    't-Statistic': results.tstats[main_vars],
    'p-Value': results.pvalues[main_vars]})
# Add model info
summary_df.loc['Observations', 'Coefficient'] = results.nobs
summary_df.loc['Observations', 't-Statistic'] = 'R² (Within)'
summary_df.loc['Observations', 'p-Value'] = round(results.rsquared_within, 3)
print(summary_df)


# c. Create month plot
# Assuming 'results' is your PanelOLS fitted model
params = results.params
pvalues = results.pvalues
# Filter only month fixed effects (those whose name starts with 'Month_')
month_coefs = params[params.index.str.startswith("Month_")]
month_pvals = pvalues[pvalues.index.str.startswith("Month_")]
# Ensure arrays
coefs = np.array(month_coefs.values)
pvals = np.array(month_pvals.values)
months = np.array([col.replace('Month_', '')[:7] for col in month_coefs.index])
# Significance masks
mask_3 = pvals < 0.001
mask_2 = (pvals >= 0.001) & (pvals < 0.01)
mask_1 = (pvals >= 0.01) & (pvals < 0.05)
mask_ns = pvals >= 0.05
plt.figure(figsize=(14, 6))
# Plot line
plt.plot(months, coefs, color='black', label='Coefficient', linewidth=2)
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
# Plot by significance
plt.scatter(months[mask_3], coefs[mask_3], color='#069AF3', marker='o', label='p < 0.001', 
            zorder=3, s=70, edgecolors='black')
plt.scatter(months[mask_2], coefs[mask_2], color='#C1F80A', marker='^', label='p < 0.01', 
            zorder=3, s=70, edgecolors='black')
plt.scatter(months[mask_1], coefs[mask_1], color='skyblue', marker='s', label='p < 0.05', 
            zorder=3, s=70, edgecolors='black')
plt.scatter(months[mask_ns], coefs[mask_ns], color='lightgray', marker='v', label='Not significant', 
            s=70, zorder=3, edgecolors='black')
# Formatting
plt.xticks(rotation=90)
plt.ylabel("Coefficient")
plt.xlabel("Month")
plt.title("Month Fixed Effects Coefficients by Significance Level", fontweight='semibold')
# Both horizontal & vertical grids
plt.grid(True, linestyle='--', alpha=0.5)
# Legend in desired order
handles, labels = plt.gca().get_legend_handles_labels()
order = [0, 1, 2, 3, 4]  # Adjust if needed
plt.legend([handles[i] for i in order], [labels[i] for i in order])
plt.tight_layout()
plt.show()


