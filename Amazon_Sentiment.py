# Required packages
import plotly
import random
import numpy as np
from os import path
from PIL import Image
import turicreate as tc
from functions import *
import plotly.offline as py
import plotly.graph_objs as go
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

# Getting data
df = getDF('reviews_Electronics_5.json.gz')

# Reading data as SFrame
reviews = tc.SFrame.read_json('Electronics_5.json', orient='lines')
cols = ['helpful', 'unixReviewTime', 'reviewTime', 'summary']
reviews = reviews.remove_columns(cols)

# Applying the String Converter Algorithm to our data
comments_transformed = [str(i) for i in reviews['reviewText']]
reviews.remove_column('reviewText')
reviews['reviewText'] = comments_transformed

# Remove punctuation for text analytics
reviews['reviewText'] = reviews['reviewText'].apply(remove_punctuation)

# Applying the Language Identifier Algorithm to our data
language = languages(reviews['reviewText'])
language_sarray = tc.SArray(language)
reviews['language'] = language_sarray

# Checking total number of reviews
print('Number of reviews: %d' % len(reviews['reviewText']))

# More specific analysis
print('Number of distinct reviewers: %d'
      % len(reviews['reviewerID'].unique()))
print('Number of distinct reviewed products: %d'
      % len(reviews['asin'].unique()))
print('Minimum rating score: %d' % reviews['overall'].min())
print('Maximum rating score: %d' % reviews['overall'].max())

# Check if there are languages other than English
english_identifier(reviews['language'])

# Sentiment threshold
reviews['sentiment'] = reviews['overall'].apply(
    lambda x: -1 if x <= 2 else (0 if (x > 2 and x < 4) else +1))

# Distribution of sentiment
print('Number of positive reviews: %d' % (reviews['sentiment'] == 1).sum())
print('Number of neutral reviews: %d' % (reviews['sentiment'] ==0).sum())
print('Number of negative reviews: %d' % (reviews['sentiment'] == -1).sum())

# Disaggregating reviews by sentiment class
positive = (reviews['sentiment'] == 1).sum()
neutral = (reviews['sentiment'] == 0).sum()
negative = (reviews['sentiment'] == -1).sum()

# Plotting a barplot with the number of reviews as a function of sentiment type
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
classes = ['Positive', 'Neutral', 'Negative']
instances = [positive, neutral, negative]
ax.bar(classes, instances)
ax.set_facecolor('#eafff5')
plt.show()

# Exploring review nature in percentage terms
percent_positive = round(100 * positive / len(reviews['sentiment']), 2)
percent_neutral = round(100 * neutral / len(reviews['sentiment']), 2)
percent_negative = round(100 * negative / len(reviews['sentiment']), 2)

print('Percent of neutral reviews: ' + str(percent_positive))
print('Percent of neutral reviews: ' + str(percent_neutral))
print('Percent of negative reviews: ' + str(percent_negative))

# Prepare features for BoW and TF-IDF models
reviews['word_count'] = tc.text_analytics.count_words(reviews['reviewText'])
reviews['tf_idf'] = tc.text_analytics.tf_idf(reviews['reviewText'])

# Creating a list via grouping reviews by product
reviews_by_product = reviews.groupby(['asin'])['reviewText'].apply(list)
text = ' '.join(reviews_by_product[1])

# Selecting stop words to be avoided
stopwords = set(STOPWORDS)

# Selecting Amazon logo as a mask so we can fit the Word Cloud into it
wave_mask = np.array(Image.open('logo.jpg'))
wordcloud = WordCloud(stopwords = stopwords,
                      background_color = 'white', mask = wave_mask).generate(text)

# Displaying the generated image
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

reviews= reviews.remove_columns(['reviewerID', 'language'])

# Indexing reviews by id
reviews['number'] = [i for i in range(len(reviews['comments']))]

# Holdout validation data disaggregation
training_validation, test_data = reviews.random_split(0.9, seed=42)
train_data, validation_data = training_validation.random_split(0.83, seed=42)

# Studying class imbalance
ratings = train_data['overall'].value_counts()
labels = ratings.index
size = ratings.values
colors = ['green', 'lightgreen', 'gold', 'crimson', 'red']
rating = go.Pie(labels = labels, values = size, marker=dict(colors=colors),
                name='Ratings Piechart', hole=0.3)
df = [rating]
layout = go.Layout(title='Percentage Ratings for Amazon Electronic Items')
fig = go.Figure(data = df,layout = layout)
py.iplot(fig)

# Defining training classes
positive_train = (train_data['sentiment'] == 1).sum()
neutral_train = (train_data['sentiment'] == 0).sum()
negative_train = (train_data['sentiment'] == -1).sum()

# Setting up the barplot
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
classes = ['Positive', 'Neutral', 'Negative']
instances = [positive_train, neutral_train, negative_train]
ax.bar(classes, instances)
ax.set_facecolor('#eafff5')
plt.show()

# Splitting the training set reviews into the three sentiment classes
train_positive = train_data[train_data['sentiment'] == 1]
train_neutral = train_data[train_data['sentiment'] == 0]
train_negative = train_data[train_data['sentiment'] == -1]

# Oversampling minority classes
train_negative1 = train_negative.append(train_negative)
train_negative1 = train_negative1.sample(frac=1).reset_index(drop=True)
train_negative2 = train_negative1.append(train_negative.head(61496))
train_negative = train_negative2.sample(frac=1).reset_index(drop=True)

train_neutral1 = train_neutral.append(train_neutral)
train_neutral1 = train_neutral1.sample(frac=1).reset_index(drop=True)
train_neutral2 = train_neutral1.append(train_neutral)
train_neutral = train_neutral2.sample(frac=1).reset_index(drop=True)

# Number of synthetic negative and neutral reviews
print(len(train_negative['sentiment']))
print(len(train_neutral['sentiment']))

# Random uniform distribution discrete number generator
undersample = random.sample(range(1, 1013664), 568789)

# Shuffle the data first and drop 568789 reviews at random
train_positive = train_positive.sample(frac=1).reset_index(drop=True)
train_positive = train_positive.drop(undersample)

# Emsembling resampled data
reviews1 = train_positive.append(train_neutral2)
reviews = reviews1.append(train_negative)
reviews = reviews.sample(frac=1).reset_index(drop=True)

# Examing rebalanced dataframe
reviews.drop(columns=['word_count', 'tf_idf'])

# Defining resampled classes
positive = (reviews['sentiment'] == 1).sum()
neutral = (reviews['sentiment'] == 0).sum()
negative = (reviews['sentiment'] == -1).sum()

# Visualizing cleaned data
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
classes = ['Positive', 'Neutral', 'Negative']
instances = [positive, neutral, negative]
ax.bar(classes, instances)
ax.set_facecolor('#eafff5')
plt.show()

# Applying Feature Engineering to replicate TF-IDF
reviews['word_count'] = tc.text_analytics.count_words(reviews['reviewText'])
reviews['tf_idf'] = tc.text_analytics.tf_idf(reviews['reviewText'])

# Updating the identifying number label for resampled training data
reviews['number'] = [i for i in range(len(reviews['comments']))]

# Machine Learning Modeling
ml_model = tc.logistic_classifier.create(reviews, target='sentiment',
                                         features = ['tf_idf'], l1_penalty=1284.43, l2_penalty=356.87,
                                         validation_set=None)

# Model summary
print(ml_model)

# Assessing model performance
ml_model.evaluate(test_data)

# Conditional probabilities for top positive and negative comments
ml_model.predict_topk(test_data)

# Most positive reviews
ml_model.predict_topk(test_data).sort(
    key_column_names=['probability', 'class'],
    ascending=False).print_rows(num_rows=5)

# Most negative reviews
ml_model.predict_topk(test_data)[
    ml_model.predict_topk(test_data)['class'] == 1].sort(
    key_column_names='probability',
    ascending=True).print_rows(num_rows=5)

# k-NN Search algorithm
knn_search = tc.nearest_neighbors.create(reviews, label='number',
                                         features=['tf_idf'], method='brute_force', distance='cosine', verbose=False)

# Top positive neighbors
top_positive = reviews[(reviews['asin'] == 'B007BYLLNI') &
                      (reviews['overall'] == 5) &
                      (reviews['number'] == 1290580)]
# Finding 3-NN
knn_search.query(top_positive, label='number', k=4)

# Most negative review k-NN Search algorithm
top_negative = reviews[(reviews['asin'] == 'B005FPT38A') &
                      (reviews['overall'] == 1) &
                      (reviews['number'] == 1127600)]
# Finding 3-NN
knn_search.query(top_negative, label='number', k=4)

# Model comparison with imbalanced data
ml_model_imb = tc.logistic_classifier.create(train_data, target = 'sentiment',
                                             features = ['tf_idf'], l1_penalty=10, l2_penalty=10, validation_set=None)

# Model summary
print(ml_model_imb)

# Assessing model performance
ml_model_imb.evaluate(test_data)