# Text Difficulty Classifier Project :books:

## What is this Project? 🤔
The Text Difficulty Classifier is a machine learning project aimed at classifying text into various difficulty levels according to the Common European Framework of Reference for Languages (CEFR). 
Our model processes text excerpts from books to determine if it is suited for beginners (A1, A2), intermediate (B1, B2), or advanced (C1, C2) learners.

## Why You Won't Regret Reading Until the End? 🌟
By the end of this README, you'll have a comprehensive understanding of how machine learning can be used for text difficulty classification. 
Plus, we've included a special video demonstration and insights from our latest Kaggle competition and a cool application of text difficulty classification!

### A note on Text Difficulty :sweat_smile:
Language proficiency into levels A1, A2, B1, B2, C1, and C2, relies on various linguistic and structural features such as:
1. **Vocabulary** :books:
   - **Frequency and Commonality**: Texts for beginners (A1, A2) tend to use high-frequency vocabulary and common words. As the level increases, the frequency of usage decreases, and the rarity or specificity of vocabulary increases.
   - **Lexical Density**: Advanced texts (C1, C2) often have a higher density of lexical items (nouns, adjectives, adverbs, and verbs), meaning there is a greater variety of words used with fewer repetitions.

2. **Grammatical Complexity** :memo:
   - **Sentence Structure**: Basic texts use simple sentences, intermediate texts introduce compound sentences, and advanced texts use complex sentences with multiple clauses.
   - **Grammatical Features**: Advanced texts employ a wider range of verb tenses, moods (such as subjunctive), passive constructions, and conditionals.
   
3. **Conceptual Complexity and Abstractness** :bulb:
   - Topics covered in texts for higher proficiency levels often involve abstract concepts, specialized knowledge areas, and philosophical debates, as opposed to the concrete, everyday topics typically discussed in beginner texts.

4. **Cultural and Idiomatic Elements** :earth_americas:
   - Advanced texts may include idioms, cultural references, and expressions that are challenging for lower-level learners to understand without specific cultural knowledge or advanced language skills.

### Why Text Difficulty Classification is Relevant :mag:
Learning a new language can easily become overwhelming without the appropriate material. Text difficulty classification can serve as an educational guide for content creators to:

1. :mortar_board: **Design appropriate exercises** and reliable language proficiency tests.

2. :open_book: **Select suitable reading materials**. Too easy texts won't aid learning effectively, whereas too difficult texts could demotivate learners.

4. :robot: **Automatically adjust content** depending on the user’s proficiency, enhancing personalized learning. 

5. :wheelchair: **Ensuring content is accessible for readers of various proficiency levels**, particularly in multilingual contexts.

## Meet the Team! 👥
- **Camille** - The Logistic regression and decision tree expert, corageous enough to dive into the advanced CamemBERT techniques! and a worderful video producer.
- **Mariana** - The data preprocessing, kNN and Random Forest expert that loves writting read me files and implementing text classifiers into streamlit apps.

## The Menu 📖
- [The Data We Used](#the-data-we-used)
- [Coding: Basic Machine Learning Techniques](#coding-using-machine-learning-techniques)
- [Coding: Advanced Machine Learning Techniques](#Now-the-moment-you-have-been-waiting-for...The-camemBERT)
- [Video Demonstration](#video-demonstration)
- [Kaggle Competition Rank](#kaggle-competition-rank)
- [Results](#results)
- [Conclusion](#conclusion)

## The Data We Used 📊
We utilized a corpus consisting of 4800 sentences ranging from A1 to C2 levels.
Then we asked ourselves:
**What defines a sentence complexity?** ✍❓
The length of the sentence, The richness of the vocabulary, The grammar or the syntax?
Before diving into machine learning models we will translate sentences into metrics and numbers.
We selected the following features to explore:
1. Word count: total tokens considered as words in the sentence.
2. Average word length
3. Sentence_length: total number oh characters
4. Rare words count
5. Number of syllables
6. Vocabulary richness
7. POS Tags Distribution: Proportion of different parts of speech (nouns, verbs, adjectives, etc.).
8. Number of clauses
9. Punctuation count
10. Named entities: words that are names of persons, organizations, locations, dates etc.
11. Conjugations
12. Word similarity: mean, median and variance of the pairwise cosine similarity between the word vectors 

Are ALL these features useful though? To answer this we will perform an RFE (recursive feature elimination) ranking with the Random Forest model. This will give us a meassure of the features' importance so that we can reduce dimensionality, simplifying the model and reducing overfitting.
![Feature Importances](Results/Text_features.png)
Interesting isn't it? :eyes: this gives us an innitial guiding light to train machine learning models on french sentences classifications. We will select the features: ['word_count', 'mean_similarity', 'median_similarity', 'num_syllables', 'variance_similarity', 'average_word_length', 'sentence_length', 'rare_words_count']
and a word frequency metric called TF-IDF to build basic and advanced models. Keep reading!:point_down:

## Coding: Using Machine Learning Techniques to Classify Text Difficulty 🖥️
We first started training models with the features listed above and using the following techniques:
- Logistic Regresssion
- KNN
- Decision Tree
- Random Forest
  
![Classification Report](Results/Classification_report_01.png)

Random Forest performed the best with a 0.427 accuracy (Far away from what we want to achieve though). We will now reflect on what the strengths and weaknesses of model are to explain these results and define next steps.

**Logistic Regression** is a linear model used for binary and multiclass classification by estimating class belonging probabilities with a logistic (sigmoid) function that is applied to the weighted sum of the input features. It is simple, interpretable, and works well with sparse data like TF-IDF features but as it is a linear model, it might struggle with complex, non-linear relationships in the data for example contextual information.
Also the boundaries between A1-A2, B1-B2, and C1-C2 seem to be difficult to find as we can see from the confusion matrix highliting the limitation of a Logistic Regression linear boundary.

**Decision Tree** splits the data into subsets based on feature values in a way that nodes represents features, branches decision rules, and leaves a class labels. The tree is built by choosing the feature that best separates the data according to a criterion like Gini impurity or information gain. It can capture non-linear relationships, are easy to interpret but they can easily overfit, especially with high-dimensional data like TF-IDF vectors. For example in the case of text classification, a Decision Tree might split the data first on the presence of a specific word or word frequency, then further split on other words or features until it reaches a decision.

**Random Forest** is an ensemble of Decision Trees. Each tree is trained on a random subset of the data and features, and their predictions are combined (usually by voting) to make the final prediction which reduces overfitting and improves generalization. They handle well non-linear relationships, are less prone to overfitting and can manage high-dimensional data but they are more complex and harder to interpret. In our example Random Forests would build multiple Decision Trees, each potentially focusing on different features or frequencies. The ensemble would combine these to provide a robust classification.

**K-Nearest Neighbors (KNN)** is a non-parametric, instance-based learning algorithm thet classifies a data point based on the majority class of its `k` nearest neighbors in the feature space. It is simple and intuitive, can capture local patterns in the data but can be computationally expensive with large datasets, as it requires storing and comparing all training samples. It can also struggle with high-dimensional data due to the curse of dimensionality.

Knowing this we will follow 2 main strategies, we will introduce **TF-IDF vectorization** to the training features and we will use **cross validation** and **hyperparameter fine-tunning** to improve our models.  

## Results 📈

After introducing TF-IDF (Term Frequency Inverse Document Frequency of records) we found out that for almost all the models the highest accuracies were obtained when training the models with both text features and word frequencies which makes sense since language difficulty is much more complicated than just word frequencies. Here a classification report for the 4 models trained with this combined approach:

![Classification Report](Results/Classification_report.png)

And here the confusion matrices for the four models. Do you find something interesting?🧐

![Confusion matrices](Results/Confusion_matrix.png)

It was stricking to see such a poor performance from the KNN classifier... 😢 And this is not only for the KNN classifier of course, overall we believe our models could do better so we decided to dive a bit deeper and try to find the way to improve them. It could be that default number of neighboors for the KNN Classifier (5) is not the best or, maybe giving more weight to closer neighbors could help or the euclidean distance (the line length between two points) might not be appropriate...so many questions.
Let's use Gris Search, Bayesian optimisation and Cross validation to find answers.🤓

...Spoiler ALERT ⛔

This is the best we got:

![Improved Models](Results/Confusion_matrix_improved.png)
![Classification Report Improved Models](Results/Classification_report_improved.png)

Optimizing parameters is quite computational expensive and time consuming but did lead to a slight improvement. If you are curious and want to know all the details behind the optimizations we tried check out the full code here 👉 https://github.com/cvermno/ML-Project/tree/main/Code

Maybe you are asking yourself as well as we did: In what sentences are the models still failing?
Well we have an answer for you. See below as an example few sentences extracted fron the mistakes done by the improved Logistic regression on the test labelled data:

![Improved Logistic Regression mistakes](Results/Mistakes_lr.png)

You can find the full table in the code shared above and here 👉 https://github.com/cvermno/ML-Project/tree/main/Results

Most of the mistakes are classifying as A1 sentences that are actually A2 level. The boundaries between these two levels seem very difficult to define even for simple human beings like us. Take the example of "Je ne fais pas grand-chose à la maison." simple easy sentence, not too long, uses quite common words...still it is not A1 level it is A2 because it already implies an exchange of information related to familiar and routine matters which is considered by 📚 [Cambridge](https://www.cambridgeenglish.org/Images/126011-using-cefr-principles-of-good-practice.pdf) 📚 as already level A2. The only way to know this subtle details is with context and deep knowledge of the french language that is why...🥁


## Now the moment you have been waiting for...The camemBERT ##

![Description of the image](https://img-9gag-fun.9cache.com/photo/amg9MGy_460s.jpg)

Facing the reality that classic ML models were limited in achieving a high accuracy, we embarked on a quest for innovative solutions 💡. In our pursuit, we encountered the CamemBERT model, a cutting-edge neural network architecture tailored for natural language understanding tasks. ⚙️And how does it work ?⚙️ First, CamemBERT undergoes a pre-training phase where it familiarizes itself with the nuances of the French language by digesting vast amounts of text data. During this stage, it learns to comprehend relationships between words and sentences, leveraging a technique called self-attention to capture contextual dependencies effectively. Once pre-training is complete, CamemBERT can be fine-tuned for specific tasks, such as predicting the difficulty of French sentences.

Our approach involved three key steps to enhance the accuracy score:
1. We adjusted the model parameters to improve performance. Let's quickly define each of the parameters:
   - Maximum length: Maximum number of words the model processes in a sentence. Longer sequences are truncated, shorter ones are padded.
   - Batch size: Number of sentences processed simultaneously during training or evaluation. Balances memory use and training speed.
   - Learning rate: Step size for updating model parameters during optimization. Balances between convergence speed and training stability.
   - Number of epochs: Number of times the model iterates over the entire dataset. More epochs can improve learning but risk overfitting.

Below is a summary table presenting the accuracy levels achieved for various parameter configurations. This table was generated by importing the predictions dataframe for the model trained on 100% of the training data to Kaggle.

![CamemBERT parameters Optimization](Results/CamemBERT_parameters.png)

2. We tried preprocessing the text by lemmatizing it. However, it did not lead to any improvement compared to the basic CamemBERT model with optimized parameters.
3. We experimented with data augmentation techniques involving "synonym" replacement. In this approach, words in the text are substituted with their synonyms to diversify the training data. The objective is to improve the robustness and generalization of machine learning models trained on the enhanced dataset. We investigated two different methods:
   - Embedding-based substitution: substitute words with similar ones based on their embeddings. This method was highly successful, achieving the best accuracy of all models: 0.5802.
   - WordNet-based substitution: replace words with synonyms retrieved from NLTK's WordNet. Although this method performed well, it did not surpass the others.

## Video Demonstration 🎥
This model is so cool that it deserves a multimedia explanation so we will stop the bla bla in this text and invite you to check out this video showing an amazing application of our text classifier:
https://youtu.be/ROy3S0kd6yE

## Kaggle Competition Rank 🏆
Our model ranked 18
With a score of 0.587
You can view the full competition details here. 👉[Competition](https://www.kaggle.com/competitions/predicting-the-difficulty-of-a-french-text-e4s/overview)
We congratulate all our colleagues for their incredible job it was definitely a challenging first competition!


## Conclusion 🎉

## Now we invite you to discover our app
https://lepetitprofcodespaces-nkb4dbpgbmujkcuhf7sey5.streamlit.app/

# Embed a GIF from Giphy

<div style="text-align: center;">
    <img src="https://media.giphy.com/media/SIaHSy7gMCCYru0bbu/giphy.gif" alt="The Little Prince Invisible Essence">
    <p><a href="https://giphy.com/gifs/orchfilms-orchard-films-the-little-prince-invisible-essence-SIaHSy7gMCCYru0bbu">via GIPHY</a></p>
</div>
