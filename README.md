### Presentation Slideshow: 
#### https://docs.google.com/presentation/d/1n3lpqqD19QFO9YwdnG5WksnYBGubXKYgjVOcOVILz6w/edit#slide=id.g2448708e995_0_135

# github_language_processing

#### Welcome to this initial exploration of Github Data Science repository data from Github!  The primary purpose of this project is to make accurate mid-season predictions about which language (R or Python) was used in a DATA SCIENCE repository based upon the text elements present in the repository README. The data was compiled by filtering publicly available Git repos based upon the topic of DATA SCIENCE and the two languages: R or PYTHON.  The repos were then sorted by "MOST STARS" and therefore, reflect the rankings on 14 May, 2023.  

#### The goals of this initial exploration are as follows:
- PRIMARY: Create a classification model which will predict which language is utilized in the repo based upon contents of the README text.
- SECONDARY: Potentially assist a beginner decide which Data Science language to learn first, based upon which repos correspond with the individual's domain experience/preferences.

#### Project Planning:
- Plan: Questions and Hypotheses
- Acquire: Curated dataset from https://docs.github.com/en/search-github by searching Git repositories.  README text strings sorted by topic: data science and language: R or Python.
- Prepare: All strings were transformed into lower-case, non-ASCII chars were removed.  Strings were tokenized and lemmatized.  English Stopwords and Custom Stopwords like "R" or "python" were removed. Split into ML subsets (Train/Validate/Test) with "language" as the defined target.
- Explore: Created a feature measuring the token count of each README, explored value_counts of R_words/Python_words/All_words,isolated words exclusive to R or Python, compared ratios of occurance of mutual-words in each language, explored word groupings (bi-grams/tri-grams), explored statistical significance of difference in mean token counts for each language.
- Model: (tf-idf --> Classification) Established a baseline "Accuracy" of 57.4% using the most frequent target occurance of "language: Python".  Then with a DecisionTreeClassifier with MaxDepth set to 4, established a new Accuracy floor of 82.1%. After creating models with different tree-based and non-tree-based algorithms and multiple tunings, findings indicated the DecisionTreeClassifier with maxDepth=4 yielded best validation results without significant OVERFITTING (67.8% Accuracy on Test).  When Classifying rows based upon bigrams, the performance of the DecisionTreeClassifier fell to 62.7% Accuracy on Test, after initially showing similar results to single-tokens.
- XXXXXXXXXXXDeliver: Please refer to this doc, as well as the Final_XXXXXXXXXXXX.ipynb file for the finished version of the presentation, in addition to each of the underlying exploratory notebooks.  The presentation slideshow can be accessed via the link at the top of the page.

#### Initial hypotheses and questions:
* Which words exist with relative parity within the README files of both languages? 
* Of these common words, are there any words which are highly representative of either language? 
* Are there any words which exist specifically for R or Python README files?  If so, what is the predictive/inferential impact of these words?
* Does the mean token_count of either language README file provide any amplifying information? 
* Does a statistically significant difference exist between the mean token count of each cleaned repo for each of the R and Python repos? 
* Will the classification of samples based upon the tf-idf of Bigrams and various ngrams lead to an improvement in model performance?

#### Data Dictionary: 

|Feature |  Data type | Definition |
|---|---|---|
| repo: | str object | user_name/repo_name of Git repository |
| language: | str object | TARGET: specifc language used in repository ( R or Python) |
| readme_contents: | str object | raw markdown contents of repo readme |
| stopped: | str object | prepared readme contents: prepped via process listed above |
| token_cnt: | int | a number indicating the count of tokens contained in 'stopped' |

#### Findings, Recommendations, and Takeaways:


- Term Frequency-Inverse Document Frequency metrics were generated to measure the relative impact of the appearance of each word in a README file, GIVEN the frequency of the word for the entire Corpus, and GIVEN the frequency of the word in each particular Document.  
- Modeling was optimized for Accuracy as the (lack-of) penalty for false -'s and false +'s is equivalent in this case.
- Decision Tree model performed best with the least amount of overfitting and was selected for use with test data.  This model showed improvement over the baseline accuracy (57.4%), with a TEST_data accuracy level of 67.8%.
- With a small training dataset (approx. 162 rows), the value of the work can be described as marginal.  Working with a larger Corpus with more rows would provide most models with sufficient data and lead to more refined predictions.  Additionally, the small-scale fluctuations inherent in a limited dataset will smooth-out over extended trials.
- In the future, it is recommended to increase the size of the dataset, as well as spend more time with modeling by adjusting hyperparameters and checking the performance of additional algorithms.

#### Applications:

- For the purposes of determining the nature and categories of use for each of the traditional "first-languages" in the field of Data Science, the end-user can utilize project exploration and model outputs to make an informed choice regarding which language to learn first.
- The value of the project will increase proportionally with the amount of data captured via exploration and modeling.

#### Instructions for those who wish to reproduce this work or simply follow along:
You Will Need (ALL files must be placed in THE SAME FOLDER!):
- 1. final_XXXXXXX_project.ipynb file from this git repo
- 2. wrangle_g.py file from this git repo
- 3. modeling.py file from this git repo
- 4. lang_data_prepped.csv from this git repo
- 5. Environment: nltk.download('stopwords') required copy paste the following into cmd prompt: python -c "import nltk; nltk.download('stopwords')"

Ensure:
- All files are in the SAME FOLDER
- wrangle_g.py and modeling.py each have the .py extension in the file name

Any further assistance required, please email me at myemail@somecompany.com.

