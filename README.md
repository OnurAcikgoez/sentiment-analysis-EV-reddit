
# Reddit Sentiment Analysis on Electric Vehicles (EVs)

This project analyzes public sentiment toward electric vehicles (EVs) by scraping Reddit data from relevant subreddits and applying Natural Language Processing (NLP) techniques, sentiment analysis, and topic modeling.

📌 Overview

- Platform: Reddit  
- Focus Subreddits: `r/cars`, `r/electricvehicles`  
- Data Collected: Headlines and comments  
- Techniques Used:  
  - Sentiment Analysis using VADER  
  - Topic Modeling using LDA (Gensim)  
  - Data Cleaning & Tokenization  
  - WordCloud & Multiple Visualizations

 🧰 Technologies Used

- Python (pandas, numpy, nltk, matplotlib, seaborn, gensim, praw, pyLDAvis)
- NLP with VADER (for sentiment classification)
- Topic Modeling with Gensim LDA
- Reddit API via `praw`

📈 Project Workflow

1. Data Collection
   - Scrapes posts and comments related to electric vehicles using specific keywords
   - Targets popular posts from the past year across selected subreddits

2. Data Cleaning
   - Removes duplicates, NaNs, URLs, short comments, and unnecessary tokens
   - Applies tokenization and lemmatization

3. Sentiment Analysis
   - Applies VADER sentiment analyzer
   - Labels text as `positive`, `neutral`, or `negative`

4. Visualization
   - Histograms and bar charts to show sentiment distribution
   - Pie charts for sentiment proportions
   - WordCloud for frequently used terms

5. Topic Modeling
   - Prepares data for Latent Dirichlet Allocation (LDA)
   - Visualizes topic distribution using pyLDAvis

 📊 Sample Visuals

- Sentiment distribution by subreddit  
- Compound score histograms  
- WordCloud of EV-related Reddit discussions  
- Interactive LDA topic explorer (saved as HTML)

 🚀 How to Run

1. Install required libraries:
   ```bash
   pip install praw nltk gensim pyLDAvis matplotlib seaborn pandas
