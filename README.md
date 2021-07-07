 ## Author:  
 * Shoval Tayro

## About the project:
My research question is- given a new post- whether it is possible to predict that the writer will commit suicide after the post or not.

### In this project I used 5 machine learning techniques-    
•	Logistic Regreession    
•	Adaboost   
•	Random Forest    
•	KNN    
•	SVM
  
  
### About the data set:
The dataset is a collection of posts from "SuicideWatch" and "depression" subreddits of the Reddit platform.
*	'text' - the text of the post. to embedding the text I had to remove 'stop words' and do stemming on the text.
*	'class' - The labal (if the writer commit suicide or not).
<img src ="https://user-images.githubusercontent.com/57803367/124820873-b2924f00-df76-11eb-9713-67dc4ac61d4d.png" width="350">

### Problems & Solutions:
•	The biggest problem I encountered during the project is how to clear the text of unwanted characters/ words that can affect on the results and can impair the learning process-  
Stop words like 'I', 'am', 'that', 'this' and etc.. are available in abundance in any human language. By removing these words, I remove the low-level information from our text in order to give more focus to the important information.  
I used 'nltk' library to remove the stop words from the text, and I removed the punctuation and the leading/ trailing whitespace of each line.
in addition I did stemming -'PorterStemmer' from 'nltk' because stemming makes the training data more dense. It reduces the size of the dictionary two or three-fold and the learning process will work better.  
I did all these steps on the text to try to create an optimal learning dataset.

•	How to extract the features-  
The techniques I used couldn't perform learning on text, only on numbers so I had to find a way to convert the text into numbers. I chose to use Unicode Conversion after trying to convert to integers and doubles that ended without success.

## RESULST :    

### Logistic Regression:
<img src ="https://user-images.githubusercontent.com/57803367/123545928-48520100-d763-11eb-9034-3c2590479f64.png" width="450">

______________________________________________________________________________________________________________________________  

### Adaboost:
<img src ="https://user-images.githubusercontent.com/57803367/123545992-8c450600-d763-11eb-8669-8c64577804ff.png" width="450">  

______________________________________________________________________________________________________________________________  

### RANDOM FOREST:  
<img src ="https://user-images.githubusercontent.com/57803367/123546016-a7b01100-d763-11eb-9751-4fff5f40cf87.png" width="450">

______________________________________________________________________________________________________________________________    

### KNN:
With CountVectorizer and with TD-IDF:  
<img src ="https://user-images.githubusercontent.com/57803367/123546061-cc0bed80-d763-11eb-9d57-d1cc95e31845.png" width="450">
__________________________________________________________________________________________________________________________      


### SVM :  
With CountVectorizer and with TD-IDF:  
<img src ="https://user-images.githubusercontent.com/57803367/124783725-f45acf80-df4d-11eb-824a-54440d48ea9f.png" width="450">

______________________________________________________________________________________________________________________________    


## Conclusions :  
From this results it can be deduced that it is possible to predict whether the writer of the post committed suicide or not.
The average of the techniques I used is 83.5%.  
The techniques that gived the highest results was Logistic Regression - 89.6%, Logistic regression is a classification algorithm used to find the probability of event success and event failure. It is used when the dependent variable is binary(0/1, True/False, Yes/No in my case suicide/ non-suicide) in nature. It supports categorizing data into discrete classes by studying the relationship from a given set of labelled data.   
KNN give the lowest results - 73.7%, in my opnion KNN give the lowest result beacuse KNN works well with smaller dataset because it is a lazy learner. It needs to store all the data and then makes decision only at run time. So if dataset is large like i used (10,000 posts from Reddit), there will be a lot of processing which may adversely impact the performance of the algorithm.
  

## Source: 
https://www.kaggle.com/nikhileswarkomati/suicide-watch
