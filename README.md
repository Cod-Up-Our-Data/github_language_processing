# github_language_processing

#### Welcome to this initial exploration of Github Data Science repository data from Github!  The primary purpose of this project is to make accurate mid-season predictions about which language (R or Python) was used in a DATA SCIENCE repository based upon the text elements present in the repository README. The data was compiled by filtering publicly available Git repos based upon the topic of DATA SCIENCE and the two languages: R or PYTHON.  The repos were then sorted by "MOST STARS" and thus, reflect the rankings on 14 May, 2023.  

#### The goals of this initial exploration are as follows:
- PRIMARY: Create a classification model which will predict which language is utilized in the repo based upon the README text.
- SECONDARY: Potentially assist a beginner decide which Data Science language to learn first, based upon which repos correspond with the individual's domain experience/preferences.

#### Project Planning:
- Plan: Questions and Hypotheses
- Acquire: Curated dataset from https://docs.github.com/en/search-github by searching Git repositories.  README text strings sorted by topic: data science and language: R or Python.
- Prepare: All strings were transformed into lower-case, non-ASCII chars were removed.  Strings were tokenized and lemmatized.  English Stopwords and Custom Stopwords like "R" or "python" were removed. Split into ML subsets (Train/Validate/Test) with "language" as the defined target.
- Explore: Created a feature measuring the token count of each README, explored value_counts of R_words/Python_words/All_words,isolated words exclusive to R or Python, compared ratios of occurance of mutual-words in each language, explored word groupings (bi-grams/tri-grams), explored statistical significance of difference in mean token counts for each language.
- Model: (tf-idf --> Classification) Established a baseline "Accuracy" for Positive class of 57.4% using the most frequent target occurance of "language: Python".  Then with a DecisionTreeClassifier with MaxDepth set to 4, established a new Accuracy floor of 82.1%. After creating models with different tree-based and non-tree-based algorithms and multiple tunings, findings indicated the DecisionTreeClassifier with maxDepth=4 yielded best validation results without significant OVERFITTING (67.8% Accuracy on Test).  When Classifying rows based upon bigrams, the performance of the DecisionTreeClassifier fell to 62.7% Accuracy on Test, after initially showing similar results as with single-tokens
- XXXXXXXXXXXDeliver: Please refer to this doc as well as the Final_XXXXXXXXXXXX.ipynb file for the finished version of the presentation, in addition to each of the underlying exploratory notebooks.

#### Initial hypotheses and questions:
* ?  
* ?  
* ? 
* ?
* Will classifying samples based upon the tf-idf of Bigrams and various ngrams lead to an improvement in model performance?
* ?

#### Data Dictionary: 

|Feature |  Data type | Definition |
|---|---|---|
| repo: | str object | user_name/repo_name of Git repository |
| language: | str object | TARGET: specifc language used in repository ( R or Python) |
| readme_contents: | str object | raw markdown contents of repo readme |
| stopped: | str object | prepared readme contents: prepped via process listed above |

#### Findings, Recommendations, and Takeaways:

- Modeling was optimized for 
- Tree-Based models performed 
- This implies
- Along with 
- In the future, 

#### Applications:

- For the purposes of 
- Further evaluation is necessary to 

#### Instructions for those who wish to reproduce this work or simply follow along:
You Will Need (ALL files must be placed in THE SAME FOLDER!):
- 1. final_nba_project.ipynb file from this git repo
- 2. wranglerer.py file from this git repo
- 3. modeling.py file from this git repo
- 4. nba.csv from this git repo

Ensure:
- CATboost library required in the working environment, however, the code in the Final_Notebook can be removed or commented out in order to run the notebook.
- All files are in the SAME FOLDER
- wranglerer.py and modeling.py each have the .py extension in the file name

Any further assistance required, please email me at myemail@somecompany.com.

