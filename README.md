Sentiment Analysis of Electric Vehicle Discussions on Reddit

Project Description

This project analyzes public sentiment and major discussion themes surrounding Electric Vehicles (EVs) on Reddit. The corpus data for the study was collected through Reddit's PRAW (The Python Reddit API Wrapper) on two subreddits: r/cars and r/electricvehicles. The primary goal was to evaluate the public’s attitude towards EV technology, which can help determine the effectiveness of policy-making and understanding consumer stance from a marketing perspective.

The study employs sentiment analysis using VADER and topic modeling with Latent Dirichlet Allocation (LDA), utilizing tools from Python’s NLP ecosystem. Visualization of the findings was performed using Matplotlib, Seaborn, and pyLDAvis.

Motivation

Due to environmental concerns and the aim for greener transportation options, EV adoption has seen a great increase, particularly after the COVID-19 pandemic. Platforms like Reddit have become hotspots for heated EV discussions, making them valuable for exploring public sentiment towards EVs.


Tools & Technologies
Python
PRAW (Reddit API Wrapper)
NLTK (Natural Language Toolkit)
VADER (Sentiment Analysis)
Gensim (Topic Modeling)
Matplotlib and Seaborn (Visualization)
pyLDAvis (Interactive topic model visualization)
Pandas (Data manipulation)


Data Collection:

Used PRAW to fetch top posts and comments from the past year from r/cars and r/electricvehicles.

Applied keyword filters to isolate EV discussions using terms like "EV", "electric vehicle", "plug-in hybrid".

Data Preprocessing:

Removed URLs, extra whitespace, and duplicates.

Applied tokenization, lemmatization, and stop word removal for topic modeling.

Sentiment Analysis:

Used VADER to generate sentiment scores and assign positive, neutral, or negative labels with a compound threshold of 0.05.

Topic Modeling:

Applied LDA with an optimal number of 4 topics, determined by maximizing coherence scores.

Visualization:

Created sentiment distribution charts and used pyLDAvis for interactive topic exploration.

Findings

The sentiment towards EVs was mostly positive, helped by the fact that r/electricvehicles made up most of the data retrieved.

A substantial amount of neutral sentiment indicates the existence of balanced and factual discussions.

Topic modeling revealed themes around charging infrastructure, battery technology, global market dynamics, and consumer orientations.

Future Work

Expand to include more subreddits and a longer temporal scope.

Apply advanced transformer-based models to enhance sentiment and topic extraction.

Conduct cross-platform analysis comparing Reddit sentiment with other platforms like Twitter.
