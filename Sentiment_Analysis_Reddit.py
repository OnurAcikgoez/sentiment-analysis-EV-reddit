# Data Collection


# Commented out IPython magic to ensure Python compatibility.
# %%capture
# pip install praw

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# pip install gensim

#!pip install numpy==1.24.3
#!pip install --upgrade gensim

# import libraries and download necessary nltk resources
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("vader_lexicon")

# import praw
import praw


user_agent = "EVscraper 1.0 by /u/HedgehogCultural8431"
reddit = praw.Reddit(
    client_id="WhiVjecBCjWuzx6PiTTp1Q",
    client_secret="aCbMTlswoSQGBk3yOtXY3yxdV8t2Wg",
    user_agent=user_agent,
    check_for_async=False
)

# select subreddit
subreddit = reddit.subreddit("cars+electricvehicles")

# define keywords
keywords_1 = ["EV", "EVs"]
keywords_2 = ["electric", "electric car", "electric vehicle", "electric truck", "phev", "plug-in hybrid", "plug-in ev"]

# fetch top posts from last year about EVs
data = []

for submission in subreddit.top(time_filter="year", limit=None):
  title = submission.title
  id = submission.id # stores post ID
  subreddit_name = submission.subreddit.display_name # stores subreddit name

  # post filtering via keywords
  if (any(keyword in title for keyword in keywords_1) or
      any(keyword in title.lower() for keyword in keywords_2)):

      data.append({
          "post_id": id,
          "subreddit": subreddit_name,
          "text_type": "headline",
          "text": title
      })

    # fetch comments
      submission.comments.replace_more(limit=0)
      for comment in submission.comments:
        # skip short comments
        if len(comment.body) > 20:
          data.append({
            "post_id": id,
            "subreddit": subreddit_name,
            "text_type": "comment",
            "text": comment.body
          })

print(len(data))

# export to DataFrame
df = pd.DataFrame(data)

# to csv
df.to_csv("reddit_ev_posts_comments.csv", encoding="utf-8", index=False)

# Approximate length or average number of words per post/comment
df.text.str.split().str.len().mean()

df.describe()
df

"""# Data Cleaning and Preprocessing"""

df = pd.read_csv("/reddit_ev_posts_comments.csv")
df.info()

# drop duplicates if they appear in the same subreddit
df.drop_duplicates(subset=["text", "subreddit"], inplace=True)

# drop NaN values
df.dropna(subset=["text"],inplace=True)

# removing spaces
df["text"] = df["text"].str.strip()

# deleting urls
def delete_url(text):
  return re.sub(r"http\S+|https\S+|www\S+", "", text)

df["text"] = df["text"].apply(delete_url)

# removing unwanted tokens and punctuation
df["tokens"] = df["text"].apply(lambda x: [w for w in nltk.word_tokenize(str(x).lower()) if w.isalpha()])

print(df.head())
df.info()

"""# Sentiment Analysis"""

# import sentiment analysis class
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# create sentiment analysis instance
sid = SentimentIntensityAnalyzer()

# generate sentiment scores and column
sentiment_scores = df["text"].apply(sid.polarity_scores)
print(sentiment_scores)

df["sentiment_scores"] = sentiment_scores

# extract the compund score using lambda function
df["compound_score"] = df["sentiment_scores"].apply(lambda x: x["compound"])

# assigning sentiment label based on sentiment score
def label_sentiment(compound):
  if compound >= 0.05:
    return "positive"
  elif compound <= -0.05:
    return "negative"
  else:
    return "neutral"

# create label column
sentiment_label = df["compound_score"].apply(label_sentiment)
df["sentiment_label"] = sentiment_label

df.head()

"""# Data Visualization"""

# compound distribution
plt.hist(df["compound_score"], bins=10)
plt.xlabel("Compound Score")
plt.ylabel("Frequency")
plt.title("Distribution of Compound Scores")
plt.show()

# average compound by sub
sub_sentiment = df.groupby("subreddit")["compound_score"].mean()

plt.bar(sub_sentiment.index, sub_sentiment.values)
plt.xlabel("Subreddit")
plt.ylabel("Average Compound Score")
plt.title("Average Compound Score by Subreddit")
plt.show()

# distribution of sentiment labels
label_counts = df["sentiment_label"].value_counts()
plt.pie(label_counts.values, labels=label_counts.index)
plt.title("Distribution of Sentiment Labels")
plt.show()

# distribution of labels by sub
grouped_sentiment = df.groupby(["subreddit", "sentiment_label"]).size().reset_index(name="count")
sns.barplot(x="subreddit", y="count", hue="sentiment_label", data=grouped_sentiment)
plt.title("Sentiment Distribution by Subreddit")
plt.show()

# sentiment by sub stacked
grouped = df.groupby(["subreddit", "sentiment_label"]).size().unstack(fill_value=0)
grouped.plot(kind="bar", stacked=True, figsize=(10,6))
plt.title("Stacked Bar Chart of Sentiment by Subreddit")
plt.xlabel("Subreddit")
plt.ylabel("Count")
plt.show()

"""# WordCloud"""

# generate wordcloud
from wordcloud import WordCloud, STOPWORDS

combined_text = " ".join(df["text"])

word_cloud = WordCloud(background_color="white", width=800, height=400, stopwords = STOPWORDS).generate(combined_text)

# display wordcloud
plt.figure(figsize=(9,8))
plt.imshow(word_cloud, interpolation="bilinear")
plt.axis("off")
plt.show

"""## Topic Modelling
Applying Latent Dirichlet Allocation (LDA) model using gensim for topic analysis and modelling.
"""

# import necessary libraries for topic modelling
nltk.download("wordnet")
nltk.download("stopwords")
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import random
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# set random seed for reproducability
random.seed(42)
np.random.seed(42)

# lemmatization and stopword removal for tokenized text

english_stops = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

# preprocessing for topic modelling
def preprocess_topic(tokens):

  # remove stopwords
  no_stops = [t for t in tokens if t not in english_stops]

  #lemmatize tokens
  lemmatized = [lemmatizer.lemmatize(t) for t in no_stops]

  return lemmatized

# update tokens column
df = df.sort_values(by="text").reset_index(drop=True)  # enforce deterministic ordering
df["tokens"] = df["tokens"].apply(preprocess_topic)

# dictionary and creating gensim corpus for LDA
dictionary = corpora.Dictionary(df["tokens"])
dictionary.save("lda_dictionary.dict") # saves dictionary


corpus = [dictionary.doc2bow(text) for text in df["tokens"]] # list of BoW representation
corpora.MmCorpus.serialize("lda_corpus.mm", corpus) # this avoids recomputing BoW every run

# load (for consistency in future runs)
dictionary = corpora.Dictionary.load("lda_dictionary.dict")
corpus = corpora.MmCorpus("lda_corpus.mm")

# define the LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=dictionary,
                                            num_topics=4,
                                            passes=10,
                                            random_state=42,
                                            per_word_topics=True,
                                            minimum_probability=0.0
                                            )

# print the topics
topics = lda_model.print_topics(num_words=10)
for topic in topics:
  print(topic)

# evaluate coherence score to determine optimal number of topics
coherence_model = CoherenceModel(model=lda_model, texts=df["tokens"], dictionary=dictionary, coherence="c_v")
coherence_score = coherence_model.get_coherence()
print("Coherence Score:", coherence_score)

"""# Visualize LDA with pyLDAvis"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install pyLDAvis

# create interactive visualization with pyLDAvis
import pyLDAvis.gensim

lda_display = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)

# save as html
pyLDAvis.save_html(lda_display, "topic_modelling_vis.html")