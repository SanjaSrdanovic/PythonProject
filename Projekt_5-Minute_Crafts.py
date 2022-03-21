# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd 
import numpy as np 
import random
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
#import PIL.ImageOps
from os import path
import os
from wordcloud import WordCloud
import string 
from collections import Counter 
import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words("english")
punctuation = string.punctuation
# print(stopwords[:10])

#%%
"""Loading the data
"""
#using getcwd() is needed to support running example in generated IPython notebook

d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
data = pd.read_csv("/Users/sanjasrdanovic/Desktop/Data Science/Python/Projekt/5-Minute Crafts.csv")
# I put the whole path because I hade some issues with opening the picture for wordcloud later
#%%
"""Inspecting the data, how many items, are there any missing values etc.
"""
# seting printing values to see all data in the console
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None)

print(data.head()) # to see first 5 item
print(data.shape) # (4978, 15)
data.shape[0] # number of rows
data.shape[1] #number of columns

print(data.info()) # no null values
data.describe()  # descriptive statistics

#%%
""" Check outliers with boxplot
"""
# sns.boxplot(data = data.describe(), palette = "coolwarm", orient = "h", width=0.3) 
# difficult to see anything because for some variables the numbers are much higher than for the others, 
# so the range difference is huge to comapre anything
# therefore, I need to create subplots

sns.set(font_scale=1.5) 
fig, axes = plt.subplots(nrows= data.describe().shape[1], figsize = (20,25)) 
# I create the figure with the subplots, where axes is an array with each subplot.
# with shape[1] I fetch the columns of pandas dataframe (columns with numerical values)
# with figsize I set the size of the graph
fig.tight_layout(pad = 3) # to make it tighter

# Then we tell each plot in which subplot we want them with the argument ax.
n = 0 # axis counter
for i in data.describe().columns:
    
    sns.boxplot(x = data[i], orient='h', ax = axes[n], color='skyblue')
    n += 1

plt.savefig("boxplot_data.png") # save figure   

    
# From the box plot, here are a few observations:
# for the activity, it is mostly from 400 to 1000 active since days
# for the duration of the videos there is many outliers, and in general most videos are relatively short.
# The total number of views: most videos only have less than millions of veiws, but there also a few above that 
# Number of characters:average around 40, 50, some outliers with more than 80
# number of words: short, average from 6 to 10, there are only a few outliers
# numer of punct: very low, usually wither none or only till 2, with a few exceptions
# number of words upperrcase there are some, for lowercase just outliers
# number of stopwords very little
# avg. word length around 5,6
# contain digits and starts with digits are categorial values, 0 or 1
# The sentiment of the titles is mostly positive. negative sentiments are outliers

#%%
""" Distribution of nummerical variables
"""
# fig, axes = plt.subplots(nrows= data.describe().shape[1], figsize = (40,40))
# fig.subplots_adjust(hspace=0.4, wspace=0.4)

# n = 0 # axis counter
# for i in data.describe().columns:
    
#     g = sns.histplot(x=data[i], ax = axes[n], color='skyblue')
#     n += 1
#     plt.savefig("histograms.png")
  
# but this gives histograms for each variable in rows in one plot, it is difficult to read off the values, 

# to get each historgram separately
n = 0 # axis counter
for i in data.describe().columns:
    sns.displot(x=data[i], color='skyblue')
    plt.ticklabel_format(style='plain', axis='x',useOffset=False) 
    n += 1
    plt.savefig(i)
   
#%%
""" Correlation plot
""" 
sns.set(font_scale=1.8) 
plt.figure(figsize=(20,20))
sns.heatmap(data.corr(),annot = True, cmap = "magma", linecolor = "white", linewidth = 1.5)
plt.savefig("correlation_plot_data_2.png") # save figure  

# we can observe from the correlation matrix that none of the features strongly correlate with the total number of views.
# num_words and num_char have a strong correlation (0.93) which totally makes sense
# and num_words also has pretty high correlation with num_stopwords (0.71)
# also contains_digits and starts_with_digits (0.75) but there are not so important variables


#%%
""" Data pre-procesing to get clean textual data in order to find most common keywords
"""


# cleaning data  
        
text = " ".join(data['title'])

#coverting to lowercase
text = text.lower()

# removing punctuation
text = "".join(t for t in text if t not in punctuation) 

# removing stopwords
text = [t for t in text.split() if t not in stopwords] # removing the stopwords

# removing digits
text = [t for t in text if not t.isdigit()] 


#%%
"""
Plot for the frequency of 10 most popular keywords
"""

x = [count for word, count in Counter(text).most_common(10)]
y = [word for word, count in Counter(text).most_common(10)]

plt.figure(figsize=(15,10));
axes = sns.barplot(x=x, y=y, palette="pastel")
plt.title("10 Most Frequent Keywords used in 5-Minute Craft Titles");
plt.xlabel("Frequency", fontsize=18);
plt.yticks(fontsize=16);
plt.xticks(fontsize=16);
plt.ylabel("Keywords", fontsize=18);
plt.savefig("most_freq_keywords.png") #save fig

#%% # plot for most common keywords according to the average total views
# none of the variable was significant predictor for the total views, 
# so it might be something for the title that draw attention of the audience, 
# so I can examine the average views of the top 100 keywords that I filtered out and  
# see which keywords from the titles had most views

#top 100 
top100 = Counter(text).most_common(100)

for word in top100:
    word =word[0]
    data[word] = data['title'].apply(lambda x : 1 if word.lower() in x.lower() else 0) #apply it to every row and column
    
# I create an empty dictionary keyword_views where I will put keys(k) keywords from top100 and values(v)   
keyword_views = {} 
for word in top100:
    word = word[0]
    # returning grouped lists in a column as a dictionary
    # df.groupby("id")["val"].agg(lambda x: Counter([a for b in x for a in b]))
    # df.groupby("grouper").agg({"val1": [min,max]}).my_flatten_cols("last")

    # I need to aggregate total views and mean and to add this grouped value as a column to dict
    # dg = data.groupby(word).agg({"total_views" : "sum"}).to_dict()['total_views'] #add to dict
    # but I think it makes more sense to get the average of total views than the sum:
    dg = data.groupby(word).agg({"total_views" : "mean"}).to_dict()['total_views'] #add to dict
    if 1 in dg:
        keyword_views[word] = dg[1]
        
keyword_views = {k: v for k, v in sorted(keyword_views.items(), key=lambda item: item[1])}
x = [_ for _ in (keyword_views.values())][::-1] # x axis are values(average views)
y = [_ for _ in (keyword_views.keys())][::-1] # y axis are keys(keywords)


plt.figure(figsize=(20,10));
axes = sns.barplot(x=x[:15], y=y[:15], palette = "pastel")

plt.title("Keywords with the highest number of views");
plt.xlabel("Average Views", fontsize=18);
plt.yticks(fontsize=16);
plt.xticks(fontsize=16);
plt.ticklabel_format(style='plain', axis='x',useOffset=False)  # to specify numeric onset
plt.savefig("keywords_x_avg_views.png") #save fig

#%%
"""visualisation of most common words in a Wordcloud
"""
def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)

saved_column = data["title"]
# print(saved_column)

lst =[]
for i in saved_column:
    lst.append(i)
# print(type(lst))
keywords = " ".join(lst)
print(type(keywords))

# Insight

print(keywords) 
print(len(keywords))
print(len(set(keywords)))

stopwords = nltk.corpus.stopwords.words("english")

# setting up the mask

mask = Image.open("/Users/sanjasrdanovic/Desktop/Data Science/Python/Projekt/lightbulb.jpg")
print(mask)


# Wordcloud
wordcloud = WordCloud(font_path="/Users/sanjasrdanovic/Desktop/Data Science/Python/Projekt/XeroxSansSerifWideBoldOblique.ttf",
                      stopwords = stopwords,
                      mask = np.array(mask), #np.array
                      background_color="blue",
                      max_words=500,
                      max_font_size=200,
                      width=1500,
                      height=2500,
                      ).generate(keywords)

plt.axis("off")

plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3),interpolation='bilinear')
#plt.title("Most common 5-Minute Crafts Keywords")
plt.savefig("wordcloud_5min_crafts.png", dpi=300)
plt.show()

#%%

""" Top 10 trending videos in the last week
"""
trending = data[data['active_since_days'] <=7]
trending = trending.sort_values(by =['total_views'] , ascending = False).head(10)
trending = trending.reset_index()
print(trending)
trending.drop(trending.columns.difference(['title','total_views']), 1, inplace = True)
position = range(1,11)
trending["position"] = position
trending = trending.set_index('position')
print(trending)

#%% 
#save the new trending videos dataframe
top10 = pd.ExcelWriter('trending_last_week.xlsx')
# write dataframe to excel
trending.to_excel(top10)
# save the excel
top10.save()
print('DataFrame is written successfully to Excel File.')



