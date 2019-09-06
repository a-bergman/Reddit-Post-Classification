# Reddit Post Classification

# Problem Statement

While scraping data from Reddit and intern at Bon Appétit realized that they had accidentally deleted the `subreddit` tag from their data.  As a result, Bon Appétit's marketing team approach us to help them figure out if a post came from [r/Cooking](https://www.reddit.com/r/Cooking/) or [r/AskCulinary]([r/Cooking](https://www.reddit.com/r/AskCulinary/).  The goal of this project is to predict the subject of origin as accurately as possible.

-------

# Executive Summary

We began the project by scraping r/Cooking and r/AskCulinary for data.  Reddit limits the amount of posts to 25 per request for a total of 1,000 posts which works out to 40 requests; we ended up with roughly 2,000 posts in total.  Luckily for us, we had some older posts stored in .csv files, so we were able to double the number of posts we had access to.

The first thing we had to do was extract features we wanted from the dictionaries since the dictionary storing a given post's data was stored in a single cell of the dataframes.  To do that we used list comprehensions to extract the `id`, `title`, `selftext`, `author`, and `source`.  Once we extracted those five features we were then able to concatenate together the new and old post .csv files for each subreddit.  Once they were appended we began the process of cleaning up the data.  The cleaning process was fairly simple, as it mostly consisted of dropping null values, stripping all text that was not composed of letters, removing duplicate posts by `id`, and making sure there were no cross-posts from one subreddit to another. The last major cleaning step we did was to concatenate together the title and self-text of each post.  We did that because we wanted to have one set of text to work with, though in the future we will consider not concatenating the text together.  Then we were able to remove some unnecessary columns from the data.  We then saved the dataframes to a new file so we could model cleaned data.

Before we began the process of modeling, we looked at the distribution of the lengths of each post and who was posting most frequently.  At this point we noticed that one post contained the remnants of a URL, so we decided to run a regular expression to remove such traces from all posts.  We then looked at what words occurred most frequently and added those words to our list of stop words.  The last preprocessing step we took was to lemmatize the text; lemmatizing seeks to reduce every word to its lemma, or dictionary entry.  We chose not to use a stemming algorithm because there are a "rougher" process and often leave behind text that is not really English, i.e. was could be reduced to wa.  Once we ran the lemmatizer, we were ready to begin modeling.

-------

# Conclusions & Recommendations
