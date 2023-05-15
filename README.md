# github_language_processing

#### Welcome to this initial exploration of Github Data Science repository data from Github!  The primary purpose of this project is to make accurate mid-season predictions about which language (R or Python) was used in a DATA SCIENCE repository based upon the text elements present in the repository README. The data was compiled by filtering publicly available Git repos based upon the topic of DATA SCIENCE and the two languages: R or PYTHON.  The repos were then sorted by "MOST STARS" and thus, reflect the rankings on 14 May, 2023.  

#### The goals of this initial exploration are as follows:
- PRIMARY: Create a classification model which will predict which language is utilized in the repo based upon the README text.
- SECONDARY: Potentially assist a beginner decide which Data Science language to learn first, based upon which repos correspond with the individual's domain experience/preferences.

#### Project Planning:
- Plan: Questions and Hypotheses
- Acquire: Curated dataset from https://docs.github.com/en/search-github by searching Git repositories.  README text strings sorted by topic: data science and language: R or Python.
- Prepare: All strings were transformed into lower-case, non-ASCII chars were removed.  Strings were tokenized and lemmatized.  English Stopwords and Custom Stopwords like "R" or "python" were removed. Split into ML subsets (Train/Validate/Test) with "language" as the defined target.
- xxxxxxExplore: xxxUnivariate and multi-variate analysis, correlation matrix, 2D visualization, correlation significance testing, 2-sample T-testing for significant differences in means.
- xxxxxxxModel: xxxEstablished a baseline "Precision" for Positive class of 57.1% using the most frequent target occurance of "yes: playoffs".  Then with a DecisionTreeClassifier with MaxDepth set to 4, established a new Precision floor of 86.0%. After creating models with different tree-based and non-tree-based algorithms and multiple tunings, findings indicated a Multi-Layer Perceptron with a three hidden layers (256,128,64 nodes) yielded best validation results (90.0% Precision on Test).
- xxxxxxDeliver: xxxPlease refer to this doc as well as the Final_NBA.ipynb file for the finished version of the presentation, in addition to each of the underlying exploratory notebooks.

#### Initial hypotheses and questions:
* ?  
* ?  
* ? 
* ?
* ?
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

