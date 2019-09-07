# Reddit Post Classification

# Problem Statement

While scraping data from Reddit and intern at Bon Appétit realized that they had accidentally deleted the `subreddit` tag from their data.  As a result, Bon Appétit's marketing team approach us to help them figure out if a post came from [r/Cooking](https://www.reddit.com/r/Cooking/) or [r/AskCulinary](https://www.reddit.com/r/AskCulinary/).  The goal of this project is to predict the subject of origin as accurately as possible.

-------

# Executive Summary

We began the project by scraping r/Cooking and r/AskCulinary for data.  Reddit limits the amount of posts to 25 per request for a total of 1,000 posts which works out to 40 requests; we ended up with roughly 2,000 posts in total.  Luckily for us, we had some older posts stored in .csv files, so we were able to double the number of posts we had access to.

The first thing we had to do was extract features we wanted from the dictionaries since the dictionary storing a given post's data was stored in a single cell of the dataframes.  To do that we used list comprehensions to extract the `id`, `title`, `selftext`, `author`, and `source`.  Once we extracted those five features we were then able to concatenate together the new and old post .csv files for each subreddit.  Once they were appended we began the process of cleaning up the data.  The cleaning process was fairly simple, as it mostly consisted of dropping null values, stripping all text that was not composed of letters, removing duplicate posts by `id`, and making sure there were no cross-posts from one subreddit to another. The last major cleaning step we did was to concatenate together the title and self-text of each post.  We did that because we wanted to have one set of text to work with, though in the future we will consider not concatenating the text together.  Then we were able to remove some unnecessary columns from the data.  We then saved the dataframes to a new file so we could model cleaned data.

Before we began the process of modeling, we looked at the distribution of the lengths of each post and who was posting most frequently.  At this point we noticed that one post contained the remnants of a URL, so we decided to run a regular expression to remove such traces from all posts.  We then looked at what words occurred most frequently and added those words to our list of stop words.  The last preprocessing step we took was to lemmatize the text; lemmatizing seeks to reduce every word to its lemma, or dictionary entry.  We chose not to use a stemming algorithm because there are a "rougher" process and often leave behind text that is not really English, i.e. was could be reduced to `wa`.  Once we ran the lemmatizer, we were ready to begin modeling.

We will model using four different models: a logistic regression, support vector classifier, random forest classifier, and an XGBoost classifier. In addition to the four models we will be making use of a count vectorizer and a TFIDF classifier.  Because there are many hyperparameters (parameters the algorithm cannot determine) which need to be tuned, we will be using grid searching each model.  Once we have our best hyperparameters, we will then evaluate the models on six metrics and a confusion matrix: represent different aspects of the model's performance and the confusion gives the clearest visualization of how the model treats each class in the data. Once we determine what our best model is, we will generate an ROC (receiver operating characteristic) and plot the individual metric scores for the model.

-------

# Conclusions & Recommendations

After looking over the metric scores and confusion matrices, we were able to identify our best model as a logistic regression with the TFIDF vectorizer. Despite this model being the "best" model, it is far from being a good model: it had an accuracy of only 68.7%, a balanced accuracy of 66.4%, and an ROC-AUC (area under the curve) score of 0.66419.  Of those three scores, the most important is the ROC-AUC because it represents how distinct the two classes are: since a score of 0.5 is the lowest possible score, our model's score of 0.66419 is very poor.  In addition to the scores being low, the models are overfit but no severely.

While our models on the whole did a good job at predicting posts from r/Cooking, we cannot recommend that Bon Appétit use our models to predict whether or not a post came from r/Cooking because our best model's scores are not high enough for us to be confident in the model's performance.

Going forward, we would want to continue preprocessing and try different modeling methods: we want to try different vectorizers and different sets of stop words as well as trying more advanced classification methods such as neural networks.