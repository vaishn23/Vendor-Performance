#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
from sqlalchemy import create_engine
import logging
logging.basicConfig(
    filename="Logs/ingestion_db.log",
    level = logging.DEBUG,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    filemode ="a"

)


# In[2]:


engine = create_engine('sqlite:///inventory.db')


# In[5]:


def load_raw_data():
    for file in os.listdir('Data'):
        if'.csv' in file:
            df = pd.read_csv('data/'+file)
            logging.info(f'Ingesting{file}in db')
            ingest_db(df, file[:-4], engine)
    end = time.time()
    total_time = (end - start)/60
    logging.info('Ingestion Complete')
    logging.info(f'Total Time Taken:{total_time} minutes')
if __name__ =='__main__':
    load_raw_data()


# In[6]:


def ingest_db (df, table_name, engine):
    df.to_sql(table_name, con = engine, if_exists ='replace', index = False)


# In[7]:


import pandas as pd
import os
from sqlalchemy import create_engine
import logging
logging.basicConfig(
    filename="Logs/ingestion_db.log",
    level = logging.DEBUG,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    filemode ="a"

)
engine = create_engine('sqlite:///inventory.db')
def load_raw_data():
    for file in os.listdir('Data'):
        if'.csv' in file:
            df = pd.read_csv('data/'+file)
            logging.info(f'Ingesting{file}in db')
            ingest_db(df, file[:-4], engine)
    end = time.time()
    total_time = (end - start)/60
    logging.info('Ingestion Complete')
    logging.info(f'Total Time Taken:{total_time} minutes')
if __name__ =='__main__':
    load_raw_data()


# In[ ]:




