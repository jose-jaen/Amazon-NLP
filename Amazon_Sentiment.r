# Setting up the working directory where the data is stored
setwd('/home/jose/Desktop/Mis cosas/Machine Learning Project')

# Firing up the relevant packages for data exploration and visualization
library(tidyverse) ; library(reshape2) ; library(ggthemes) ; library(ggridges); 
library(fBasics) ; library(ggExtra) ; library(e1071) ; library(BSDA) ; library(nortest) 

# Reading data
reviews <- read.csv('amazon_reviews.csv', header = T, sep = ',', dec = '.')
attach(reviews)
names(reviews)


# Grouping the data to show reviewed products in descending order
reviewed_products <- reviews %>% 
                      group_by(asin) %>% 
                      summarize(occurrences = n()) %>% 
                      arrange(desc(occurrences)) 


# Filtering top 10 products
top_reviewed_products <- head(reviewed_products, 10)

# Reduced data frame with targeted features
labels <- c(top_reviewed_products$asin)
count <- c(top_reviewed_products$occurrences)

# Getting the products' name
new_labels <- c('SanDisk 64GB Memory', 'HDMI Amazon Cable', 'Chrome -cast HDMI', 
                'Media -bridge HDMI', 'Trascend SDHC Card',
                'ErgoFit Head -phones', 'DVI HDMI Cable', 
                'USB Apple Cable', 'Roku3 Player', 'eneloop AA Batteries')

top_reviews_df <- data.frame(labels, new_labels, count)

# Barplot of top 10 reviewed products
top_reviews <- ggplot(top_reviews_df, aes(x = reorder(new_labels, -count), y = count)) + 
                    geom_bar(stat = 'identity', fill = '#1663BE') +
                    geom_text(aes(label = count), vjust = -1, size = 4.2) +
                    theme_economist() +
                    labs(title = 'TOP 10 REVIEWED PRODUCTS', x = '', y = '# REVIEWS') +
                    theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5, vjust = -2)) +
                    scale_x_discrete(labels = function(x) stringr::str_wrap(x, width = 1))

top_reviews

# Average rating of products
ratings <- reviews %>% 
            select(asin, overall) %>% 
            group_by(asin) %>% 
            summarize(occurrences = n(), avg_rating = mean(overall)) %>% 
            arrange(desc(occurrences)) 

top_ratings <- head(ratings, 10)
top_ratings_df <- data.frame(top_ratings)


# Plotting the average rating of the top 10 reviewed products
top_ratings <- ggplot(top_ratings_df, aes(x = reorder(new_labels, -occurrences), y = avg_rating)) + 
                    geom_bar(stat = 'identity', fill = '#1663BE') +
                    geom_text(aes(label = round(avg_rating, 2)), vjust = -1, size = 4.2) +
                    theme_economist() + ylim(c(0, 6)) +
                    geom_hline(yintercept = 4, color = 'red', size = 1) +
                    labs(title = 'TOP 10 PRODUCTS AVERAGE RATING', x = '', y = 'RATING') +
                    theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5, vjust = -2)) +
                    scale_x_discrete(labels = function(x) stringr::str_wrap(x, width = 1))

top_ratings

# Let us introduce a threshold rule 
reviews_query <- reviews %>% 
                  mutate (
                    sentiment = case_when(
                    overall <= 2 ~ 'Negative',
                    overall > 2 & overall < 4 ~ 'Neutral',
                    overall >= 4 ~ 'Positive'
                  )
                 ) %>% 
                  select(asin, overall, sentiment) %>% 
                  group_by(asin) %>% 
                  summarize(occurrences = n(), overall, avg_rating = mean(overall), sentiment, asin)

reviews_query <- as.data.frame(reviews_query)

# Total number of positive reviews
length(reviews_query$asin[reviews_query$sentiment == 'Positive'])

# Total number of neutral reviews
length(reviews_query$asin[reviews_query$sentiment == 'Neutral'])

# Total number of negative reviews
length(reviews_query$asin[reviews_query$sentiment == 'Negative'])


# Let us now focus on average rating
segmented_reviews <- reviews %>% 
                 select(asin, overall) %>% 
                 group_by(asin) %>% 
                 summarize(occurrences = n(), avg_rating = mean(overall)) %>% 
                 mutate (
                      sentiment = case_when(
                      avg_rating <= 2 ~ 'Negative',
                      avg_rating > 2 & avg_rating < 4 ~ 'Neutral',
                      avg_rating >= 4 ~ 'Positive'
                      )
                  ) 

# Distribution of products by average rating 
general_distribution <- ggplot(data = segmented_reviews, aes(x = avg_rating)) +
  geom_histogram(aes(y = ..density..), position = 'identity', alpha = 0.5, color = '#BA55D3', fill = '#BA55D3') + geom_density(alpha = 0.6, color = 'red') +
  scale_color_manual(values = c('#999999', '#E69F00', '#56B4E9')) +
  scale_fill_manual(values = c('#999999', '#E69F00', '#56B4E9')) +
  labs(title = 'RATING DISTRIBUTION', x = 'AVERAGE RATING', y = 'DENSITY') + theme_economist() +
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5, vjust = -2)) 

general_distribution

# As seen in the graph, the distribution is left-skewed
mean(segmented_reviews$avg_rating)
median(segmented_reviews$avg_rating)
skewness(segmented_reviews$avg_rating)
skewness(segmented_reviews$occurrences)

# Checking lack of normality
lillie.test(segmented_reviews$avg_rating)
lillie.test(segmented_reviews$occurrences)

# Identify correlation between ratings and number of comments
# Scatterplot
scatter_plot <- ggplot(data = segmented_reviews, aes(x = avg_rating, y = occurrences, fill = sentiment, color = sentiment)) + 
    geom_point() + theme_economist() +
    scale_fill_manual(values = c('#E3242B', '#E69F00', '#56B4E9')) +
    scale_color_manual(values = c('#E3242B', '#E69F00', '#56B4E9')) +
    theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5, vjust = -2)) +
    labs(title = 'OCCURRENCES BY SENTIMENT', x = 'AVERAGE RATING', y = 'OCCURENCES') +
    coord_cartesian(ylim = c(0, 5000))

scatter_plot
cor(segmented_reviews$avg_rating, segmented_reviews$occurrences)

# A quick glance at the graph and the low correlation value are indicative of a low relationship between both variables
# Let's test if the correlation is statistically significant
correlationTest(segmented_reviews$avg_rating, segmented_reviews$occurrences, 'pearson')

# We reject the null hypothesis that there is none, however, there does not seem to be a linear relationship
# Let us use rank correlation coefficients:
correlationTest(segmented_reviews$avg_rating, segmented_reviews$occurrences, 'spearman')
correlationTest(segmented_reviews$avg_rating, segmented_reviews$occurrences, 'kendall')

# Variables' distributions are not independent
# Due to a strong presence of outliers, let us visualize a boxplot
boxplot <- ggplot(data = segmented_reviews, aes(x = avg_rating, y = occurrences, fill= sentiment)) + 
    geom_boxplot(alpha = 0.8) +
    scale_fill_manual(values = c('#E3242B', '#E69F00', '#56B4E9')) +
    scale_color_manual(values = c('#E3242B', '#E69F00', '#56B4E9')) +
    theme(legend.position = 'none') + theme_economist() +
    labs(title = 'Nº REVIEWS DISTRIBUTION', x = 'AVERAGE RATING', y = 'OCCURRENCES') +
    theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5, vjust = -2)) +
    coord_cartesian(ylim = c(0, 50)) + xlim(c(0, 6.5))

boxplot

# Let's test median equivalency hypotheses with sign test 
positive_data <- segmented_reviews$occurrences[segmented_reviews$sentiment == 'Positive']
SIGN.test(positive_data, md = median(segmented_reviews$occurrences[segmented_reviews$sentiment == 'Neutral']), 
          alternative = 'greater')

# Identifying top positive and negative reviews for K-NN Algorithm in Python given the retrieved labels
# Searching for comments that fulfill conditions
reviews$reviewText[(reviews$asin == 'B007BYLLNI') & (reviews$overall == 5)]

# Selecting the objective among potential candidates
reviews$reviewText[(reviews$asin == 'B007BYLLNI') & (reviews$overall == 5)][1]

# Most negative review
reviews$reviewText[(reviews$asin == 'B005FPT38A') & (reviews$overall == 1)]

# Choosing the target
reviews$reviewText[(reviews$asin == 'B005FPT38A') & (reviews$overall == 1)][1]