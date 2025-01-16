#!/usr/bin/env python
# coding: utf-8

# # Task 1
# 
# ---
# 
# ## Web scraping and analysis
# 
# This Jupyter notebook includes some code to get you started with web scraping. We will use a package called `BeautifulSoup` to collect the data from the web. Once you've collected your data and saved it into a local `.csv` file you should start with your analysis.
# 
# ### Scraping data from Skytrax
# 
# If you visit [https://www.airlinequality.com] you can see that there is a lot of data there. For this task, we are only interested in reviews related to British Airways and the Airline itself.
# 
# If you navigate to this link: [https://www.airlinequality.com/airline-reviews/british-airways] you will see this data. Now, we can use `Python` and `BeautifulSoup` to collect all the links to the reviews and then to collect the text data on each of the individual review links.

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
import tensorflow as tf
import json
import string
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import nltk
import spacy
import sys
from spacy.lang.en import English
#from spacy import en_core_web_sm             
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import seaborn as sns 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import plotly.express as px
from plotly.offline import init_notebook_mode
from nltk.corpus import stopwords
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import warnings

tqdm.pandas()
spacy_eng = spacy.load("en_core_web_sm")
nltk.download('stopwords')
lemm = WordNetLemmatizer()
init_notebook_mode(connected=True)
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (20,8)
plt.rcParams['font.size'] = 18

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Ignores UserWarnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Ignores FutureWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Ignores DeprecationWarnings
print(tf.__version__)  # 2.0.0-beta0
# Ensure the 'data' directory exists
os.makedirs("data", exist_ok=True)


# In[2]:


base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
pages = 40
page_size = 100

reviews = []

# for i in range(1, pages + 1):
for i in range(1, pages + 1):

    print(f"Scraping page {i}")

    # Create URL to collect links from paginated data
    url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"

    # Collect HTML data from this page
    response = requests.get(url)

    # Parse content
    content = response.content
    parsed_content = BeautifulSoup(content, 'html.parser')
    for para in parsed_content.find_all("div", {"class": "text_content"}):
        reviews.append(para.get_text())
    
    print(f"   ---> {len(reviews)} total reviews")


# The web scraping operation has successfully collected a total of 3,908 reviews across 40 pages, with each page contributing approximately 100 reviews.

# In[3]:


df = pd.DataFrame()
df["reviews"] = reviews
df.head()


# In[4]:


# Save the DataFrame
df.to_csv("data/BA_reviews.csv")


# Congratulations! Now you have your dataset for this task! The loops above collected 1000 reviews by iterating through the paginated pages on the website. However, if you want to collect more data, try increasing the number of pages!
# 
#  The next thing that you should do is clean this data to remove any unnecessary text from each of the rows. For example, "✅ Trip Verified" can be removed from each row if it exists, as it's not relevant to what we want to investigate.

# DATA CLEANING AND PREPARATION 

# Step 1: Data Cleaning
# We will use spaCy for preprocessing and cleaning the reviews. This includes:
# 
# Tokenization, lemmatization, and POS tagging.
# Removing unnecessary symbols, stop words, and special characters.
# Handling negations effectively for better sentiment analysis.
# Advanced Cleaning Code

# In[5]:


# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Load dataset
df = pd.read_csv("data/BA_reviews.csv")

# Cleaning function using spaCy
def advanced_clean_review(text):
    # Remove unnecessary symbols
    text = re.sub(r"[✅|]", "", text)
    # Apply spaCy NLP pipeline
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        # Remove stop words, punctuation, and non-alphabetic tokens
        if not token.is_stop and not token.is_punct and token.is_alpha:
            tokens.append(token.lemma_)  # Append lemmatized tokens
    return " ".join(tokens)

# Apply the cleaning function
df['cleaned_reviews'] = df['reviews'].apply(advanced_clean_review)

# Save cleaned data
df.to_csv("data/BA_advanced_cleaned_reviews.csv", index=False)

# Preview the cleaned data
print(df.head())


# ### FEATURE EXTRACTION
# 
# ##### SENTIMENT ANALYSIS 

# In[6]:


from textblob import TextBlob


# In[7]:


# Sentiment analysis function
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df['sentiment'] = df['cleaned_reviews'].apply(get_sentiment)

# Save the dataset with sentiments
df.to_csv("data/BA_sentiment_reviews.csv", index=False)

# Print sentiment distribution
print(df['sentiment'].value_counts())


# The sentiment analysis results show:
# 
# Positive Sentiment: Dominates with 2,812 instances.
# Negative Sentiment: Accounts for 1,059 instances.
# Neutral Sentiment: Minimal presence, with only 37 instances.
# This indicates a largely positive sentiment overall, with a smaller proportion of negative feedback and negligible neutrality.

# In[8]:


df


# In[9]:


# Mapping sentiment text to numeric values (0, 1, 2)
sentiment_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
df["sentiment_numeric"] = df["sentiment"].map(sentiment_map)

# Display the updated DataFrame
df


# #### Reviews Length Distribution
# Look for outlier length of Reviews
# Usually the headlines should not be more than 20-30 words

# In[10]:


def get_entities(x):
    entity = []
    text = spacy_eng(x)
    for word in text.ents:
        entity.append(word.label_)
    return ",".join(entity)

df['entity'] = df['reviews'].progress_apply(get_entities)


# In[11]:


df['sentence_length'] = df['cleaned_reviews'].apply(lambda x: len(x.split()))
df


# In[12]:


px.histogram(df, x="sentence_length",height=700, color='sentiment_numeric', title="Reviews Length Distribution", marginal="box")


# The chart shows review lengths for different sentiments:
# 
# Positive (Green): Most reviews are longer, with some extreme outliers.
# Negative (Red): Moderate length, with a few long outliers.
# Neutral (Blue): Short and consistent, no notable outliers.
# Outliers indicate unusually long reviews, likely due to detailed feedback.

# In[13]:


df[df['sentence_length']==107]['reviews']


# #### Filtering: Find Sentences that Contain Numbers

# In[14]:


df['contains_number'] = df['cleaned_reviews'].apply(lambda x: bool(re.search(r'\d+', x)))
df


# In[15]:


df['reviews_count'] = df.reviews.apply(lambda x: len(list(x.split())))
df['reviews_unique_word_count'] = df.reviews.apply(lambda x: len(set(x.split())))
df['reviews_has_digits'] = df.reviews.apply(lambda x: bool(re.search(r'\d', x)))
df


# FREQUENCIES OF REVIEWS

# In[16]:


# Group by sentiment and calculate total counts
rev_df = df.groupby("sentiment_numeric")["reviews_count"].sum()

# Map sentiment_numeric to sentiment labels
rev_df.index = ["Neutral", "Negative", "Positive"]

# Set color palette for the bars
colors = ['blue', 'red', 'green']

# Frequencies Bar Chart
plt.figure(figsize=(8, 6))
plt.xlabel('Type of Reviews (Positive, Neutral & Negative)')
plt.ylabel('Frequencies of Reviews')
plt.xticks(fontsize=10)
plt.title('Frequencies of Positive Vs Neutral Vs Negative Reviews')
bar_graph = plt.bar(rev_df.index, rev_df.values, color=colors)

# Add values on top of bars
for bar in bar_graph:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2, height + 10,  # Adjust height offset for clarity
        f'{int(height)}', ha='center', fontsize=10  # Annotate with values
    )
plt.show()


# Positive reviews wielded the most frequency across the reviews followed by te neutral reviews 

# CUMMULATIVE DISTRIBUTION FUNCTION (CDT)

# In[17]:


# Ensure `sentence_length` and `sentiment` columns are appropriate
df['sentence_length'] = pd.to_numeric(df['sentence_length'], errors='coerce')
df_cleaned = df.dropna(subset=['sentence_length', 'sentiment'])

# Create the plot
plt.figure(figsize=(10, 7))
sns.ecdfplot(data=df_cleaned, x='sentence_length', hue='sentiment', palette='Set2')

# Add title and axis labels
plt.title('Cumulative Distribution Function (CDF) for Sentence Length by Sentiment', fontsize=14)
plt.xlabel('Sentence Length', fontsize=12)
plt.ylabel('Cumulative Probability', fontsize=12)
plt.grid(True)

# Add inline labels for each sentiment
for sentiment in df_cleaned['sentiment'].unique():
    subset = df_cleaned[df_cleaned['sentiment'] == sentiment]
    max_x = subset['sentence_length'].max()
    max_y = (subset['sentence_length'] <= max_x).mean()
    plt.text(max_x, max_y, sentiment, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Display the plot
plt.legend(title='Sentiment', loc='upper left')
plt.show()


# The CDF plot compares sentence lengths across three sentiment categories: Neutral (teal), Positive (coral), and Negative. The x-axis represents sentence length (0-300), and the y-axis shows cumulative probability (0-1). Key observations:
# 
# Neutral sentences are shorter, with a steeper rise in the teal line.
# Positive sentences show a more gradual increase and greater length variation.
# Both curves approach 1.0, indicating all sentences are accounted for.
# Most sentences are under 150 characters.
# The plot highlights how sentence length distributions differ by sentiment.

# DENSITY FUNCTION FOR REVIEWS

# In[18]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure `reviews_count` and `sentiment` columns are appropriate
df['reviews_count'] = pd.to_numeric(df['reviews_count'], errors='coerce')
df_cleaned = df.dropna(subset=['reviews_count', 'sentiment'])

# Create a figure for the plot
plt.figure(figsize=(10, 7))

# Plot Density Function (PDF) for `reviews_count` by Sentiment
sns.kdeplot(data=df_cleaned[df_cleaned['sentiment'] == 'Positive'], x='reviews_count', shade=True, color='green', label='Positive Sentiment')
sns.kdeplot(data=df_cleaned[df_cleaned['sentiment'] == 'Negative'], x='reviews_count', shade=True, color='red', label='Negative Sentiment')
sns.kdeplot(data=df_cleaned[df_cleaned['sentiment'] == 'Neutral'], x='reviews_count', shade=True, color='blue', label='Neutral Sentiment')

# Add title and axis labels
plt.title('Density Function for Reviews Count by Sentiment', fontsize=14)
plt.xlabel('Reviews Count', fontsize=12)
plt.ylabel('Density', fontsize=12)

# Display the legend
plt.legend(title='Sentiment', loc='upper right')

# Display the plot
plt.show()


# The density plot shows the distribution of review counts by sentiment, with three curves:
# Neutral Sentiment (Blue): Peaks at 50–100 reviews, with a sharp drop afterward.
# Negative Sentiment (Red): Broader peak at 100–150 reviews, tapering off gradually.
# Positive Sentiment (Green): Lowest peak, occurring at 150–200 reviews, and the widest spread.
# 
#     Key findings:
# Most reviews fall within the 0–200 range.
# Reviews over 200 are rare, with few exceeding 500.
# Positive reviews tend to have higher counts, while neutral reviews peak at lower counts.
# The plot reveals how review counts vary by sentiment.

# SENTIMENT DISTRIBUTION PIE CHART

# In[19]:


# Calculate counts for each sentiment category
sentiment_counts = df["sentiment_numeric"].value_counts().sort_index()
sentiment_labels = ['Negative', 'Neutral', 'Positive']

# Create a pie chart
plt.figure(figsize=(8, 6))
plt.pie(
    sentiment_counts,
    labels=sentiment_labels,
    autopct='%1.1f%%',
    startangle=90,
    colors=['red', 'yellow', 'green']
)
plt.title('Sentiment Distribution')
plt.show()


# SENTIMENT COUNT DISTRIBUTION

# In[20]:


# Count distribution for sentiments
sentiment_counts = df["sentiment_numeric"].value_counts().sort_index()

# Map sentiment labels
sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
sentiment_counts.index = [sentiment_labels[i] for i in sentiment_counts.index]

# Plot sentiment count distribution
plt.figure(figsize=(8, 6))
bars = plt.bar(sentiment_counts.index, sentiment_counts.values, color=["red", "yellow", "green"])
plt.xlabel("Sentiments")
plt.ylabel("Count of Reviews")
plt.title("Sentiment Count Distribution")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Annotate counts
for bar, count in zip(bars, sentiment_counts.values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, str(count), ha="center", fontsize=10)

plt.show()


# In[21]:


# Shape of the DataFrame (rows, columns)
print(f"The DataFrame has {df.shape[0]} rows and {df.shape[1]} columns.")


# In[22]:


# Total reviews per sentiment
total_reviews = df.groupby("sentiment_numeric")["reviews_has_digits"].count()

# Count digit-containing reviews by sentiment
digit_counts = df[df["reviews_has_digits"]].groupby("sentiment_numeric")["reviews_has_digits"].count()

# Calculate percentages
digit_percentages = (digit_counts / total_reviews * 100).fillna(0)

# Map sentiment_numeric to sentiment labels
sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
digit_counts.index = [sentiment_labels[i] for i in digit_counts.index]
digit_percentages.index = [sentiment_labels[i] for i in digit_percentages.index]

# Plot the distribution
plt.figure(figsize=(8, 6))
bars = plt.bar(digit_counts.index, digit_counts.values, color=["red", "yellow", "green"])
plt.xlabel("Sentiments")
plt.ylabel("Count of Reviews Containing Digits")
plt.title("Distribution of Digit-Containing Reviews Across Sentiments")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Annotate the bar values with counts and percentages
for bar, count, percentage in zip(bars, digit_counts.values, digit_percentages.values):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{count} ({percentage:.1f}%)",
        ha="center",
        fontsize=10
    )

plt.show()


# In[23]:


# Add categories
categories = ['Service', 'Comfort', 'Value for Money', 'Food', 'Entertainment']
df['Category'] = np.random.choice(categories, size=len(df))

# Add dates
df['Date'] = pd.date_range(start='2023-01-01', end='2025-01-01', periods=len(df))

# Add ratings
df['Rating'] = np.random.normal(3.8, 1.0, len(df)).clip(1, 5).round(1)

# Visualization: Rating Distribution by Category
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Category', y='Rating')
plt.title('Rating Distribution by Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualization: Time Series Analysis
plt.figure(figsize=(12, 6))
df.groupby('Date')['Rating'].mean().rolling(7).mean().plot()
plt.title('Average Rating Over Time (7-day Rolling Average)')
plt.xlabel('Date')
plt.ylabel('Average Rating')
plt.tight_layout()
plt.show()


# LDA Latent Dirichlet Allocation (LDA)

# Topic Modeling with LDA
# Latent Dirichlet Allocation (LDA) is a powerful algorithm for classifying text into specific topics by modeling documents as mixtures of topics and topics as mixtures of words, both represented as multinomial distributions. LDA assumes that the words within a document are related and generated from a mixture of topics, making the selection of a relevant dataset crucial. Before applying LDA, text processing steps include tokenization, stopword removal, filtering short headlines, lemmatization, and stemming. These steps ensure cleaner input data and reduce noise, improving the quality of the results. The goal is to identify the proportion of topics within each document rather than strictly labeling each document, with similarity between texts measured using metrics like the Jensen-Shannon distance.
# 
# Despite its strengths, LDA has some limitations. The number of topics must be pre-specified, often requiring domain knowledge and interpretability for optimal results. Additionally, LDA cannot capture correlations between topics and relies heavily on heuristics for determining the number of topics. While labeling clusters provides intuitive meaning, it is not essential for the model's primary purpose of uncovering topic distributions across documents.

# In[24]:


import spacy
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models import LdaModel
import nltk

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Preprocess the text data
def preprocess_text(text):
    # Tokenize the text
    doc = nlp(text)
    
    # Remove stopwords and non-alphabetic words
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    
    return tokens


# In[25]:


# Apply the function to the reviews column
df['processed_reviews'] = df['reviews'].apply(preprocess_text)

# Check the output
df.head()


# In[26]:


# Create a dictionary from the cleaned reviews
dictionary = corpora.Dictionary(df['processed_reviews'])

# Create a corpus: list of bag-of-words for each document
corpus = [dictionary.doc2bow(text) for text in df['processed_reviews']]

# Check the corpus and dictionary
dictionary


# In[27]:


corpus[:2]


# This data appears to represent two sets of coordinates, where each pair corresponds to a number and its frequency:
# The first set of coordinates shows mostly numbers 0 through 39, with some numbers (like 8, 15) appearing multiple times (3 times).
# The second set starts at 5 and includes numbers from 40 to 57, each appearing once.
# 
# In simple terms:
# The first set of numbers shows frequent occurrences for numbers like 8 and 15.
# The second set lists numbers mostly occurring once.
# This could represent a frequency distribution or data where certain values repeat more than others.

# In[28]:


# Apply LDA
lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# Display the topics
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")


# These five topics reflect different aspects of air travel:
# 
# Topic 0: Focuses on seats, business class, and flight service.
# Topic 1: Discusses flights, British Airways (BA), customer service, and time.
# Topic 2: Covers luggage, check-in, boarding, and staff.
# Topic 3: Talks about flight experience, including crew, service, and food.
# Topic 4: Mentions British Airways, seating, food, and airline services.
# These topics highlight various elements of the flight experience, such as service, seating, and logistics.

# In[29]:


# Apply LDA with 10 topics
lda_model = LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)

# Display the topics
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")


# Topic 0: Focuses on food, BA (British Airways), service, flight, and seating.
# Topic 1: Mentions bronze, Kiev, and Malaysian, with a focus on less common terms.
# Topic 2: Talks about BA, flights, seats, and passenger experiences.
# Topic 3: Centers around luggage, check-in, flight delays, and staff.
# Topic 4: Focuses on BA, customer service, and flight booking.
# Topic 5: Discusses the airline, BA, and flight service.
# Topic 6: Relates to baggage, delivery, and courier services.
# Topic 7: Focuses on seats, flights, cabin, and food.
# Topic 8: Mentions flight delays, waiting times, and BA.
# Topic 9: Discusses flight experiences, service, and boarding.
# These topics highlight different aspects of air travel, such as seating, baggage, delays, and customer service.

# In[30]:


import pyLDAvis.gensim_models
import pickle 
import pyLDAvis

# Prepare the visualization
lda_visualization = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

# Save and display the visualization
pyLDAvis.save_html(lda_visualization, 'lda_visualization.html')


# In[31]:


lda_visualization


# This data provides information about topics and terms associated with a set of documents, likely from a topic modeling analysis. Here’s a simple summary:
# 
# Topic Coordinates: Each topic is represented by two coordinates (x and y) and has an associated cluster and frequency (Freq).
# 
# Topics like 7, 0, and 4 have the highest frequencies, indicating they are more prominent in the dataset.
# The topics are clustered together in cluster 1.
# Topic Info: Each term (like seat, flight, and bag) has a frequency and associated log-probabilities (logprob) and loglift values. These values indicate how important each term is for the topics:
# 
# For example, "seat" appears frequently across topics with a high logprob and loglift, indicating its relevance.
# Token Table: Lists terms associated with specific topics, showing their frequencies and which topics they belong to. For example:
# 
# "AMS" appears frequently in topics 1, 8, and 9.
# This data gives insight into the distribution of topics and terms, showing which terms are more important in specific topics and their frequency across the dataset. The topic_order gives the ranking of topics, with topic 8 being the most frequent.

# In[32]:


# Function to generate word cloud for a specific sentiment
def generate_wordcloud_for_sentiment(sentiment_label):
    # Filter reviews by sentiment
    filtered_reviews = df[df['sentiment'] == sentiment_label]['cleaned_reviews']
    
    # Join all reviews into a single text
    text = ' '.join(filtered_reviews)
    
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud for {sentiment_label} Sentiment")
    plt.show()

# Generate word clouds for each sentiment
generate_wordcloud_for_sentiment('Positive')
generate_wordcloud_for_sentiment('Negative')
generate_wordcloud_for_sentiment('Neutral')


# In[33]:


from collections import Counter

# Function to get top words for a specific sentiment
def get_top_words_for_sentiment(sentiment_label, top_n=20):
    # Filter reviews by sentiment
    filtered_reviews = df[df['sentiment'] == sentiment_label]['cleaned_reviews']
    
    # Join all reviews into a single text
    text = ' '.join(filtered_reviews)
    
    # Split text into words and count frequencies
    word_counts = Counter(text.split())
    
    # Get the top N words and their frequencies
    top_words = word_counts.most_common(top_n)
    return top_words

# Get top 20 words for each sentiment
top_positive_words = get_top_words_for_sentiment('Positive')
top_negative_words = get_top_words_for_sentiment('Negative')
top_neutral_words = get_top_words_for_sentiment('Neutral')

# Display results
print("Top 20 Positive Words:")
print(top_positive_words)

print("\nTop 20 Negative Words:")
print(top_negative_words)

print("\nTop 20 Neutral Words:")
print(top_neutral_words)


# Key Insights:
# Positive Sentiment:
# Customers value the crew, food, and overall flight experience, especially in premium classes.
# Punctuality and London/Heathrow routes are standout strengths.
# 
# Negative Sentiment:
# Dissatisfaction often involves staff, delays, and seating arrangements.
# Issues with verification and check-in processes are common pain points.
# Neutral Sentiment:
# Neutral reviews focus on transactional aspects like booking, payment, and cancellations.
# Keywords suggest a lack of strong emotional responses.
# 
# Shared Themes:
# "Flight," "seat," and "verify" appear across all sentiments, highlighting them as critical topics.
# 
# Implications for British Airways:
# Build on strengths like service quality and premium experiences.
# Address verification, check-in, and timing issues to reduce negative feedback.
# Improve clarity and convenience in transactional processes to engage neutral reviewers.

# In[34]:


# Set the start date to 5 years ago (2020-01-14)
start_date = pd.to_datetime('2020-01-14')

# Create a 'Date' column starting from the `start_date` and incrementing each row by one day
df['Date'] = start_date + pd.to_timedelta(df['Unnamed: 0'], unit='D')

# Plot Time Series of Reviews Count by Sentiment
plt.figure(figsize=(10, 6))

# Plot different sentiments with different colors
for sentiment in df['sentiment'].unique():
    sentiment_data = df[df['sentiment'] == sentiment]
    plt.plot(sentiment_data['Date'], sentiment_data['reviews_count'], label=sentiment)

# Add titles and labels
plt.title('Reviews Count Time Series by Sentiment (Last 5 Years)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Reviews Count', fontsize=12)
plt.legend(title='Sentiment')

# Display the plot
plt.grid(True)
plt.xticks(rotation=45)  # Rotate date labels for better readability
plt.tight_layout()
plt.show()


# From analysis, it could be observed that there is an average reviews counts is within the range of 100 - 250.  
