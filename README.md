# github_language_processing

#### Welcome to this initial exploration of Github Data Science repository data from Github!  The primary purpose of this project is to make accurate mid-season predictions about which language (R or Python) was used in a DATA SCIENCE repository based upon the text elements present in the repository README. The data was compiled by filtering publicly available Git repos based upon the topic of DATA SCIENCE and the two languages: R or PYTHON.  The repos were then sorted by "MOST STARS" and thus, reflect the rankings on 14 May, 2023.  

#### The goals of this initial exploration are as follows:
- PRIMARY: Create a classification model which will predict which language is utilized in the repo based upon the README text.
- SECONDARY: Potentially assist a beginner decide which Data Science language to learn first, based upon which repos correspond with the individual's domain experience/preferences.

#### Project Planning:
- Plan: Questions and Hypotheses
- Acquire: Curated dataset from https://docs.github.com/en/search-github by searching Git repositories.  README text strings sorted by topic: data science and language: R or Python.
- Prepare: All strings were transformed into lower-case, non-ASCII chars were removed.  Strings were tokenized and lemmatized.  Stopwords (en) were removed. Split into ML subsets (Train/Validate/Test) with
- Explore: xxxUnivariate and multi-variate analysis, correlation matrix, 2D visualization, correlation significance testing, 2-sample T-testing for significant differences in means.
- Model: xxxEstablished a baseline "Precision" for Positive class of 57.1% using the most frequent target occurance of "yes: playoffs".  Then with a DecisionTreeClassifier with MaxDepth set to 4, established a new Precision floor of 86.0%. After creating models with different tree-based and non-tree-based algorithms and multiple tunings, findings indicated a Multi-Layer Perceptron with a three hidden layers (256,128,64 nodes) yielded best validation results (90.0% Precision on Test).
- Deliver: xxxPlease refer to this doc as well as the Final_NBA.ipynb file for the finished version of the presentation, in addition to each of the underlying exploratory notebooks.