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
<img src ="https://user-images.githubusercontent.com/57803367/123545893-1e98da00-d763-11eb-8b28-f8dcf3f6ef7c.png" width="350">

### Problems:
•	The biggest problem I encountered during the project is how to clear the text of unwanted characters/ words that can affect on the results.
•	how to extract the features

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
<img src ="https://user-images.githubusercontent.com/57803367/123546087-dcbc6380-d763-11eb-8fc7-bf5429eb700b.png" width="450">

______________________________________________________________________________________________________________________________    


## Conclusions :  
From this results it can be deduced that it is possible to predict whether the writer of the post committed suicide or not.
The average of the techniques I used is 83.5%
The techniques that gived the highest results was Logistic Regression - 89.6% and KNN give the lowest results - 73.7%
  

## Source: 
https://www.kaggle.com/nikhileswarkomati/suicide-watch
