#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


sns.set_style("darkgrid")


# In[8]:


df = pd.read_csv("E:\spotify\data.csv")
df.drop("Unnamed: 0", axis=1,inplace=True)
df.head()


# In[ ]:


##Data Cleaning


# In[9]:


df.isna().sum()


# In[10]:


df.info()


# In[11]:


df.shape


# In[12]:


df.columns


# In[13]:


len(df.columns)


# In[14]:


df.describe()


# In[15]:


#Data analysis


# In[16]:


###Top 5 most popular artists


# In[19]:


top_five_artists = df.groupby("artist").count().sort_values(by="song_title", ascending=False)["song_title"][:5]
top_five_artists                                                           


# In[20]:


top_five_artists.plot.barh()


# In[21]:


#top five loudest tracks


# In[26]:


top_five_loudest_tracks= df[["loudness","song_title"]].sort_values(by="loudness",ascending=True)[:5]
top_five_loudest_tracks


# In[28]:


plt.figure(figsize=(12,7))
sns.barplot(x="loudness", y="song_title",data=top_five_loudest_tracks)
plt.title("Top 5 loudest tracks")
plt.show()


# In[29]:


###Artist with most danceability song


# In[32]:


top_five_artists_danceable_songs =df[["danceability", "song_title","artist"]].sort_values(by="danceability",ascending=False)[:5]
top_five_artists_danceable_songs


# In[33]:


plt.figure(figsize=(12,7))
sns.barplot(x="danceability",y="artist",data=top_five_artists_danceable_songs)
plt.title("Artist with the most danceability songs")
plt.show()


# In[34]:


##top instrumental tracks


# In[36]:


top_ten_instrumental_tracks=df[["instrumentalness","song_title","artist"]].sort_values(by="instrumentalness",ascending=False)[:5]
top_ten_instrumental_tracks


# In[40]:


plt.figure(figsize=(12,7))
plt.pie(x="instrumentalness", data=top_ten_instrumental_tracks,autopct='%1.2f%%',labels=top_ten_instrumental_tracks.song_title)
plt.title("Top 10 instrumental tracks")
plt.show()


# In[56]:


### multiple feauture plots


# In[58]:


interest_feature_cols = ["tempo","loudness","danceability","duration_ms",
                        "energy","instrumentalness","liveness","speechiness","valence"]


# In[62]:


for feature_col in interest_feature_cols:
    pos_data =df[df["target"]==1][feature_col]
    neg_data =df[df["target"]==0][feature_col]
    
    plt.figure(figsize=(12,7))
    
    sns.distplot(pos_data,bins=30,label="positive",color="green")
    sns.distplot(neg_data,bins=30,label="Negative",color="red")
    
    plt.legend(loc="upper right")
    plt.title(f"Positive And Negative Histogram Plot For{feature_col}")
    plt.show()
    


# In[ ]:





# In[ ]:




