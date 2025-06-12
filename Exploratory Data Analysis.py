#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis
# Understanding the dataset to explore how the data is present in the Database and if there's any necessity to create aggregated tables that can help with:
# 
# * Vendor Selection for Profitability
# * Product Pricing Optimization

# In[2]:


import pandas as pd
import sqlite3


# In[3]:


#Creating database connection
conn = sqlite3.connect('inventory.db')


# In[4]:


#checking tables present in the database
tables = pd.read_sql_query("SELECT name From sqlite_master WHERE type = 'table'",conn)
tables


# In[5]:


for table in tables['name']:
    print('-'*50, f'{table} ','-'*50)
    print('Count of records:',pd.read_sql(f"select count(*) as count from {table}",conn)['count'].values[0])
    display(pd.read_sql(f"select * from {table} limit 5",conn))


# In[6]:


purchases = pd.read_sql_query("select * from purchases where VendorNumber = 4466",conn)
purchases


# In[7]:


purchase_prices = pd.read_sql_query("select * from purchase_prices where VendorNumber = 4466",conn)
purchase_prices


# In[8]:


vendor_invoice = pd.read_sql_query("select * from vendor_invoice where VendorNumber =4466",conn )
vendor_invoice


# In[9]:


sales = pd.read_sql_query("select * from sales where VendorNo =4466", conn)
sales


# In[10]:


purchases.groupby(['Brand','PurchasePrice'])[['Quantity','Dollars']].sum()


# In[11]:


sales.groupby('Brand')[['SalesDollars','SalesPrice','SalesQuantity']].sum()


# In[12]:


freight_summary = pd.read_sql_query("select VendorNumber, sum(Freight) as FreightCost from vendor_invoice group by VendorNumber",conn)


# In[13]:


freight_summary


# In[14]:


pd.read_sql_query("""select 
p.VendorNumber, 
p.VendorName,
p.Brand,
p.PurchasePrice,
pp.Volume,
pp.Price as ActualPrice,
sum(p.Quantity) as TotalPurchaseQuantity,
sum(p.Dollars) as TotalPurchaseDollars
from Purchases p
join purchase_prices pp on p.brand = pp.brand
where p.PurchasePrice>0
group by p.VendorNumber, p.VendorName, p.Brand
order by TotalPurchaseDollars""" ,conn)


# In[15]:


pd.read_sql_query("""select
VendorNo,
Brand,
sum(SalesDollars) as TotalSalesDollars,
sum(SalesPrice) as TotalSalesPrice,
sum(SalesQuantity) as TotalSalesQuantity,
sum(ExciseTax) as TotalExciseTax
from Sales
group by VendorNo, Brand
order by TotalSalesDollars""",conn)


# In[19]:


import time
start = time.time()
final_table=pd.read_sql_query("""select 
pp.VendorNumber,
pp.Brand,
pp.Price as ActualPrice,
pp.PurchasePrice,
sum(s.SalesQuantity) as TotalSalesQuantity,
sum(s.SalesDollars) as TotalSalesDollars,
sum(s.SalesPrice) as TotalSalesPrice,
sum(s.ExciseTax) as TotalExciseTax,
sum(vi.Quantity) as TotalPurchaseQuantity,
sum(vi.Dollars) as TotalPurchaseDollars,
sum(vi.Frieght) as TotalFreightCost
from purchase_prices pp
join sales s
    on pp.VendorNumber =s.VendorNo
    and pp.Brand = s.Brand
join vendor_invoice vi
    on pp.VendorNumber = vi.VendorNumber
group by pp.VendorNumber, pp.Brand, pp.Price, pp.PurchasePrice""",conn)
end = time.time()


# In[20]:


vendor_sales_summary = pd.read_sql_query("""
WITH FreightSummary AS (
    SELECT
        VendorNumber,
        SUM(freight) AS FreightCost
    FROM vendor_invoice
    GROUP BY VendorNumber
),
PurchaseSummary AS (
    SELECT
        p.VendorNumber,
        p.VendorName,
        p.Brand,
        p.Description,
        p.PurchasePrice,
        pp.Price AS ActualPrice,
        pp.Volume,
        SUM(p.Quantity) AS TotalPurchaseQuantity,
        SUM(p.Dollars) AS TotalPurchaseDollars
    FROM purchases p
    JOIN purchase_prices pp
        ON p.Brand = pp.Brand
    WHERE p.PurchasePrice > 0
    GROUP BY p.VendorNumber, p.VendorName, p.Brand, p.Description, p.PurchasePrice, pp.Price, pp.Volume
),
SalesSummary AS (
    SELECT
        VendorNo,
        Brand,
        SUM(SalesQuantity) AS TotalSalesQuantity,
        SUM(SalesDollars) AS TotalSalesDollars,
        SUM(SalesPrice) AS TotalSalesPrice,
        SUM(ExciseTax) AS TotalExciseTax
    FROM sales
    GROUP BY VendorNo, Brand
)
SELECT
    ps.VendorNumber,
    ps.VendorName,
    ps.Brand,
    ps.Description,
    ps.PurchasePrice,
    ps.ActualPrice,
    ps.Volume,
    ps.TotalPurchaseQuantity,
    ps.TotalPurchaseDollars,
    ss.TotalSalesQuantity,
    ss.TotalSalesDollars,
    ss.TotalSalesPrice,
    ss.TotalExciseTax,
    fs.FreightCost
FROM PurchaseSummary ps
LEFT JOIN SalesSummary ss
    ON ps.VendorNumber = ss.VendorNo
    AND ps.Brand = ss.Brand
LEFT JOIN FreightSummary fs
    ON ps.VendorNumber = fs.VendorNumber
ORDER BY ps.TotalPurchaseDollars DESC
""", conn)


# In[21]:


vendor_sales_summary


# In[22]:


vendor_sales_summary.dtypes


# In[23]:


vendor_sales_summary.isnull().sum()


# In[24]:


vendor_sales_summary['Volume'] = vendor_sales_summary['Volume'].astype('float64')


# In[25]:


vendor_sales_summary.fillna(0, inplace = True)


# In[26]:


vendor_sales_summary['VendorName'] = vendor_sales_summary['VendorName'].str.strip()


# In[27]:


vendor_sales_summary['GrossProfit'] = vendor_sales_summary['TotalSalesDollars'] -vendor_sales_summary['TotalPurchaseDollars']


# In[28]:


vendor_sales_summary


# In[29]:


vendor_sales_summary ['ProfitMargin'] = (vendor_sales_summary['GrossProfit']/vendor_sales_summary['TotalSalesDollars']) *100


# In[30]:


vendor_sales_summary['Stock Turnover'] = vendor_sales_summary ['TotalSalesQuantity']/vendor_sales_summary['TotalPurchaseQuantity']


# In[31]:


vendor_sales_summary['SalestoPurchaseRatio'] = vendor_sales_summary['TotalSalesDollars']/vendor_sales_summary['TotalPurchaseDollars']


# In[32]:


cursor = conn.cursor()


# In[33]:


cursor.execute("""create table vendor_sales_summary(
    VendorNumber int
    VendorName varchar(100),
    Brand int
    Description varchar(100),
    PurchasePrice decimal(10,2),
    ActualPrice decimal(10,2),
    Volume,
    TotalPurchaseQuantity int,
    TotalPurchaseDollars Decimal(15,2),
    TotalSalesQuantity int,
    TotalSalesDollars decimal(15,2),
    TotalSalesPrice decimal(15,2),
    TotalExciseTax decimal(15,2),
    FreightCost decimal(15,2),
    GrossProfit decimal(15,2),
    ProfitMargin decimal(15,2),
    StockTurnover decimal(15,2),
    SalestoPurchaseRatio decimal(15,2),
    primary key (VendorNumber, Brand)
);
""")


# In[34]:


pd.read_sql_query("select *from vendor_sales_summary", conn)


# In[35]:


vendor_sales_summary.to_sql('vendor_sales_summary', conn, if_exists = 'replace', index = False)


# In[36]:


pd.read_sql_query("select *from vendor_sales_summary", conn)


# In[37]:


import pandas as pd
import sqlite3
import logging
from db_ingestion import ingest_db

logging.basicConfig(
    filename="Logs/get_vendor_summary.log",
    level = logging.DEBUG,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    filemode ="a"

)
def create_vendor_summary(conn):
    vendor_sales_summary = pd.read_sql_query("""WITH FreightSummary AS (
        SELECT
            VendorNumber,
        SUM(freight) AS FreightCost
        FROM vendor_invoice
        GROUP BY VendorNumber
    ),
    PurchaseSummary AS (
        SELECT
            p.VendorNumber,
            p.VendorName,
            p.Brand,
            p.Description,
            p.PurchasePrice,
            pp.Price AS ActualPrice,
            pp.Volume,
            SUM(p.Quantity) AS TotalPurchaseQuantity,
            SUM(p.Dollars) AS TotalPurchaseDollars
        FROM purchases p
        JOIN purchase_prices pp
            ON p.Brand = pp.Brand
        WHERE p.PurchasePrice > 0
        GROUP BY p.VendorNumber, p.VendorName, p.Brand, p.Description, p.PurchasePrice, pp.Price, pp.Volume
    ),
    SalesSummary AS (
        SELECT
            VendorNo,
            Brand,
            SUM(SalesQuantity) AS TotalSalesQuantity,
            SUM(SalesDollars) AS TotalSalesDollars,
            SUM(SalesPrice) AS TotalSalesPrice,
            SUM(ExciseTax) AS TotalExciseTax
        FROM sales
        GROUP BY VendorNo, Brand
    )
    SELECT
        ps.VendorNumber,
        ps.VendorName,
        ps.Brand,
        ps.Description,
        ps.PurchasePrice,
        ps.ActualPrice,
        ps.Volume,
        ps.TotalPurchaseQuantity,
        ps.TotalPurchaseDollars,
        ss.TotalSalesQuantity,
        ss.TotalSalesDollars,
        ss.TotalSalesPrice,
        ss.TotalExciseTax,
        fs.FreightCost
    FROM PurchaseSummary ps
    LEFT JOIN SalesSummary ss
        ON ps.VendorNumber = ss.VendorNo
        AND ps.Brand = ss.Brand
    LEFT JOIN FreightSummary fs
        ON ps.VendorNumber = fs.VendorNumber
    ORDER BY ps.TotalPurchaseDollars DESC""", conn)
    
    return vendor_sales_summary


def clean_data(df):
    df['Volume'] = df['Volume'].astype('float64')
    df.fillna(0, inplace = True)
    df['VendorName'] = df['VendorName'].str.strip()
    df['Description'] = df['Description'].str.strip()
    vendor_sales_summary['GrossProfit'] = vendor_sales_summary['TotalSalesDollars'] -vendor_sales_summary['TotalPurchaseDollars']
    vendor_sales_summary ['ProfitMargin'] = (vendor_sales_summary['GrossProfit']/vendor_sales_summary['TotalSalesDollars']) *100
    vendor_sales_summary['Stock Turnover'] = vendor_sales_summary ['TotalSalesQuantity']/vendor_sales_summary['TotalPurchaseQuantity']
    vendor_sales_summary['SalestoPurchaseRatio'] = vendor_sales_summary['TotalSalesDollars']/vendor_sales_summary['TotalPurchaseDollars']
    
    return df

if __name__ =='__main__':
    conn = sqlite3.connect('inventory.db')
    
    logging.info('Creating Vendor Summary Table..')
    summary_df = create_vendor_summary(conn)
    logging.info(summary_df.head())
    
    logging.info('Cleaning Data..')
    clean_df = clean_data(summary_df)
    logging.info(clean_df.head())
    
    logging.info('Ingesting data..')
    ingest_db(clean_df, 'vendor_sales-summary',conn)
    logging.info('Completed')


# In[38]:


vendor_sales_summary.to_csv('vendor_sales_summary.csv', index=False)


# In[39]:


brand_performance.to_csv('brand_performance.csv', index=False)


# In[ ]:




