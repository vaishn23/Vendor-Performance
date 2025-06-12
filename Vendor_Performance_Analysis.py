#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sqlite3
from scipy.stats import ttest_ind
import scipy.stats as stats


# In[2]:


conn =sqlite3.connect('inventory.db')
df = pd.read_sql_query("select * from vendor_sales_summary", conn)
df.head()


# In[3]:


df.describe().T


# In[4]:


numerical_cols = df.select_dtypes(include=np.number).columns
plt.figure(figsize = (15,10))
for i, col in enumerate(numerical_cols):
    plt.subplot(4,4,i+1)
    sns.histplot(df[col], kde=True, bins =30)
    plt.title(col)
plt.tight_layout()
plt.show()


# In[5]:


plt.figure(figsize = (15,10))
for i, col in enumerate(numerical_cols):
    plt.subplot(4,4,i+1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()


# In[6]:


#removing inconsistencies
df = pd.read_sql_query("""select *
from vendor_sales_summary
where GrossProfit >0
and ProfitMargin > 0
and TotalSalesQuantity >0""",conn)


# In[7]:


df


# In[8]:


numerical_cols = df.select_dtypes(include=np.number).columns
plt.figure(figsize = (15,10))
for i, col in enumerate(numerical_cols):
    plt.subplot(4,4,i+1)
    sns.histplot(df[col], kde=True, bins =30)
    plt.title(col)
plt.tight_layout()
plt.show()


# In[9]:


categorical_cols = ['VendorName', 'Description']
plt.figure(figsize = (12,5))
for i, col in enumerate(categorical_cols):
    plt.subplot(1,2,i+1)
    sns.histplot(y=df[col].value_counts().index[:10])
    plt.title(f'Count Plot of{col}')
plt.tight_layout()
plt.show()


# In[10]:


plt.figure(figsize =(12,8))
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot = True, fmt = ".2f", cmap = "coolwarm", linewidths = 0.5)
plt.title("Correlation Heatmap")
plt.show()


# ## Identify brands which needs Promotional or Pricing Adjustments which exhibits lower Sales Performance but Higher Profit Margins

# In[11]:


brand_performance = df.groupby('Description').agg({
    'TotalSalesDollars':'sum',
    'ProfitMargin':'mean'}). reset_index()


# In[12]:


low_sales_threshold = brand_performance['TotalSalesDollars'].quantile(0.15)
high_margin_threshold = brand_performance['ProfitMargin'].quantile(0.85)


# In[13]:


low_sales_threshold


# In[14]:


high_margin_threshold


# In[15]:


target_brands = brand_performance[
    (brand_performance['TotalSalesDollars']<= low_sales_threshold)&
    (brand_performance['ProfitMargin'] >= high_margin_threshold)
]
print("Brands with low sales but high profit margins:")
display(target_brands.sort_values('TotalSalesDollars'))


# In[16]:


brand_performance = brand_performance[brand_performance['TotalSalesDollars']<10000]


# In[17]:


plt.figure(figsize=(10, 6))

sns.scatterplot(data=brand_performance, x='TotalSalesDollars', y='ProfitMargin', color="blue", label="All Brands", alpha=0.2)
sns.scatterplot(data=target_brands, x='TotalSalesDollars', y='ProfitMargin', color="red", label="Target Brands")

plt.axhline(high_margin_threshold, linestyle='--', color='black', label="High Margin Threshold")
plt.axvline(low_sales_threshold, linestyle='--', color='black', label="Low Sales Threshold")
plt.xlabel("Total Sales ($)")
plt.ylabel("Profit Margin (%)")
plt.title("Brands for Promotional or Pricing Adjustments")
plt.legend()
plt.grid(True)
plt.show()


# ## Which vendors and brands demonstrate the highest sales performance?

# In[18]:


def format_dollars(value):
    if value>= 1_000_000:
        return f'{value/1_000_000:.2f}M'
    elif value >= 1_000:
        return f"{value/1_000:.2f}K"
    else:
        return str(value)


# In[19]:


top_vendors=df.groupby ("VendorName") ["TotalSalesDollars"].sum().nlargest(10)

top_brands=df.groupby("Description") ["TotalSalesDollars"].sum().nlargest(10)

top_vendors


# In[20]:


top_vendors


# In[21]:


top_brands


# In[22]:


top_brands.apply(lambda x :format_dollars(x))


# In[23]:


plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
ax1=sns.barplot(y=top_vendors.index, x=top_vendors.values, palette="Blues_r")
plt.title("Top 10 Vendors by Sales")

for bar in ax1.patches:
    ax1.text(bar.get_width() + (bar.get_width()*0.02),
             bar.get_y() + bar.get_height() / 2,
             format_dollars(bar.get_width()),
             ha='left', va='center', fontsize=10, color='black')

#Plot for Top Brands
plt.subplot(1, 2, 2)
ax2=sns.barplot(y=top_brands.index.astype(str), x=top_brands.values, palette="Reds_r")
plt.title("Top 10 Brands by Sales")

for bar in ax2.patches:
    ax2.text(bar.get_width()+ (bar.get_width()*0.02),
             bar.get_y()+ bar.get_height() / 2,
             format_dollars(bar.get_width()),
             ha='left', va='center', fontsize=18, color='black')
    
plt.tight_layout()
plt.show()


# ## Which vendors contribute the most to total purchase?

# In[24]:


vendor_performance = df.groupby('VendorName').agg({
    'TotalPurchaseDollars':'sum',
    'GrossProfit':'sum',
    'TotalSalesDollars':'sum'
}).reset_index()
vendor_performance


# In[25]:


vendor_performance['PurchaseContribution%'] = vendor_performance['TotalPurchaseDollars']/vendor_performance['TotalPurchaseDollars'].sum()*100


# In[26]:


def format_dollars(value):
    try:
        value = float(value)
        if value >= 1_000_000:
            return f'{value / 1_000_000:.2f}M'
        elif value >= 1_000:
            return f'{value / 1_000:.2f}K'
        else:
            return f'{value:.2f}'
    except (ValueError, TypeError):
        return value


# In[27]:


top_vendors=round(vendor_performance.sort_values('PurchaseContribution%', ascending =False),2).head(10)


# In[28]:


top_vendors['TotalSalesDollars'] = top_vendors['TotalSalesDollars'].apply(format_dollars)
top_vendors['TotalPurchaseDollars'] = top_vendors['TotalPurchaseDollars'].apply(format_dollars)
top_vendors['GrossProfit'] = top_vendors['GrossProfit'].apply(format_dollars)
top_vendors


# In[29]:


top_vendors['PurchaseContribution%'].sum()


# In[30]:


top_vendors['Cumulative_Contribution%'] = top_vendors['PurchaseContribution%'].cumsum()
top_vendors


# In[31]:


fig, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_vendors ['VendorName'], y=top_vendors ['PurchaseContribution%'], palette="mako", ax=ax1)
for i, value in enumerate(top_vendors ['PurchaseContribution%']):
    ax1.text(i, value-1, str(value)+'%', ha='center', fontsize=10, color='white')
ax2= ax1.twinx()
ax2.plot(top_vendors ['VendorName'],top_vendors ['Cumulative_Contribution%'], color='red', marker='o', linestyle='dashed', label = 'CumulativeContribution')
ax1.set_xticklabels(top_vendors ['VendorName'], rotation=90)
ax1.set_ylabel('Purchase Contribution %', color='blue')
ax2.set_ylabel('Cumulative Contribution %', color='red')
ax1.set_xlabel('Vendors')
ax1.set_title('Pareto Chart: Vendor Contribution to Total Purchases')
ax2.axhline(y=100, color='gray', linestyle='dashed', alpha=0.7)
ax2.legend(loc='upper right')
plt.show()


#  ## How much of total procurement is dependent on the top vendors?

# In[32]:


print(f"Total Purchase Contribution of top 10 vendors is {round(top_vendors['PurchaseContribution%'].sum(),2)}%")


# In[33]:


vendors = list(top_vendors ['VendorName'].values)
purchase_contributions = list(top_vendors ['PurchaseContribution%'].values)
total_contribution= sum(purchase_contributions)
remaining_contribution= 100-( total_contribution)
#Append "Other Vendors category
vendors.append("Other Vendors")
purchase_contributions.append(remaining_contribution)
#Donut Chart
fig, ax=plt.subplots(figsize=(8, 8))
wedges, texts, autotexts= ax.pie(purchase_contributions, labels=vendors, autopct='%1.1f%%',
                                startangle =140, pctdistance =0.85, colors= plt.cm.Paired.colors)
centre_circle=plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)
plt.text(0, 0, f"Top 10 Total:\n{total_contribution:.2f}%", fontsize=14, fontweight='bold', ha='center', va='center')
plt.title("Top 10 Vendor's Purchase Contribution(%)")
plt.show()


# ## Does purchasing in bulk reduce the unit price; if yes what is the optimal purchase volume for cost savings?

# In[34]:


df['UnitPurchasePrice'] = df ['TotalPurchaseDollars']/df['TotalPurchaseQuantity']


# In[35]:


df['OrderSize']= pd.qcut(df["TotalPurchaseQuantity"], q=3, labels = ["Small", "Medium", "Large"])


# In[36]:


df.groupby ('OrderSize')[['UnitPurchasePrice']].mean()


# In[37]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="OrderSize", y="UnitPurchasePrice", palette="Set2")
plt.title("Impact of Bulk Purchasing on Unit Price")
plt.xlabel("Order Size")
plt.ylabel("Average Unit Purchase Price")
plt.show()


# ## What is 95% Confidence Interval for profit margins of top-performing and low-performing Vendors?

# In[38]:


top_threshold = df['TotalSalesDollars'].quantile(0.75)
low_threshold = df['TotalSalesDollars'].quantile(0.25)


# In[39]:


top_vendors = df[df["TotalSalesDollars"]>=top_threshold]["ProfitMargin"].dropna()
low_vendors = df[df["TotalSalesDollars"]<=low_threshold]["ProfitMargin"].dropna()


# In[40]:


top_vendors


# In[41]:


low_vendors


# In[42]:


def confidence_interval(data, confidence=0.95):
    mean_val = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))
    t_critical=stats.t.ppf((1+ confidence) / 2, df=len(data) - 1)
    margin_of_error = t_critical * std_err
    return mean_val, mean_val-margin_of_error, mean_val + margin_of_error


# In[43]:


top_mean, top_lower, top_upper= confidence_interval(top_vendors)
low_mean, low_lower, low_upper= confidence_interval(low_vendors)

print(f"Top Vendors 95% CI: ({top_lower:.2f}, {top_upper:.2f}), Mean: {top_mean:.2f}")
print(f"Low Vendors 95% CI: ({low_lower:.2f}, {low_upper:.2f}), Mean: {low_mean:.2f}")

plt.figure(figsize=(12, 6))

# Top Vendors Plot

sns.histplot(top_vendors, kde=True, color="blue", bins=30, alpha=0.5, label="Top Vendors")
plt.axvline(top_lower, color="blue", linestyle="--", label=f"Top Lower: {top_lower:.2f}")
plt.axvline(top_upper, color="blue", linestyle="--", label=f"Top Upper: {top_upper:.2}")
plt.axvline(top_mean, color="blue", linestyle="--", label=f"Top Mean: {top_mean:.2f}")

# Low Vendors Plot

sns.histplot(low_vendors, kde=True, color="red", bins=30, alpha=0.5, label="Low Vendors")
plt.axvline(low_lower, color="red", linestyle="--", label=f"Low Lower: {low_lower:.2f}")
plt.axvline(low_upper, color="red", linestyle="--", label=f"Low Upper: {low_upper:.2f}")
plt.axvline(low_mean, color="red", linestyle="--", label=f"Low Mean: {low_mean:.2f}")
plt.title("Confidence Interval Comparison: Top vs. Low Vendors (Profit Margin)")
plt.xlabel("Profit Margin (%)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()

