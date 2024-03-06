# Marijuana_effects_chemical_analysis

## Overview

This repository aims to analyze the effects of marijuana against the chemical composition of different strains. The project explores the relationship between various strains and their chemical components to provide insights into potential effects.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Analysis](#analysis)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

I embark on a journey through the intricate realm of marijuana effects, meticulously examining
    their classification based on chemical composition. This exploration dives deep into the 
    diverse cannabinoids and terpene present in marijuana, offering a nuanced understanding 
    of their physiological and psychological impacts on users. Through a thorough analysis of 
    these chemical constituents, I will categorize and elucidate the multifaceted experiences 
    reported by individuals. Through this analysis, I endeavor to contribute valuable insights, 
    bridging the scientific understanding of marijuana effects and facilitating informed 
    decision-making in both medical and recreational contexts. 
    
## Features

Highlight key features of the project, such as data visualization, statistical analysis, or any machine learning models utilized.

## Getting Started

Guide users on setting up the project locally.

### Prerequisites
  
#### Download necessary NLTK resources
    nltk.download('wordnet') # lemmatizer
    nltk.download('omw-1.4') # lemmatizer
    nltk.download('vader_lexicon') # Vader sentiment analyzer
    nltk.download('treebank') # treebank
    nltk.download('punkt') # tokenizer
    nltk.download('averaged_perceptron_tagger') # part of speech
    
#### Import required libraries
    import re
    import pandas as pd
    import numpy as np
    import csv
    import os
    import sys
    from rdkit import Chem
    from rdkit.Chem import Draw
    import pubchempy as pcp
    from collections import Counter
    import matplotlib.pyplot as plt #to support wordcloud
    from wordcloud import WordCloud
    import nltk
    from nltk import pos_tag, word_tokenize #parses words or sentances
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer #finds the root word
    from nltk import FreqDist
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.corpus import treebank
    from sklearn.datasets import make_classification
    from sklearn import linear_model
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.compose import ColumnTransformer 
    from sklearn.impute import SimpleImputer
    from sklearn import preprocessing
    
### Installation

Step-by-step instructions on how to install and set up the project on a local machine.

## Usage

Explain how to use the project, including any command-line instructions or configuration settings.

## Data Sources

Data from :   de la Fuente, Alethia;  Zamberlan, Federico; Ferran, Andres;  Carrillo, Facundo; 
    Tagliazucchi, Enzo;  Pallavicini , Carla (2019), “Data from: Over eight hundred cannabis 
    strains characterized by the relationship between their subjective effects, perceptual 
    profiles, and chemical compositions”, Mendeley Data, V1, doi: 10.17632/6zwcgrttkp.1
    
Additional references:
    https://www.nltk.org/index.html Bird, Steven, Edward Loper and Ewan Klein (2009), Natural 
    Language Processing with Python. O'Reilly Media Inc.
    
    Seabold, Skipper, and Josef Perktold. “statsmodels: Econometric and statistical modeling 
    with python.” Proceedings of the 9th Python in Science Conference. 2010.
    
    Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

    OpenAI. (2024). ChatGPT. Personal communication, February 20 to March 3, 2024

    Bastian M., Heymann S., Jacomy M. (2009). Gephi: an open source software for exploring and
    manipulating networks.

    
## Analysis

The data was cleaned by removing effects present in all strains and removing effects present in only one strain.  This left me with 30 effects to analyze.

    The effects were initially separated into two categories:

    Category I (effects created by the use of marijuana): Anxious, Aroused, Euphoric, Talkative, 
    Relaxed, Happy, Creative, Energetic, Hungry, Tingly, Uplifted, Focused, Paranoid, Giggly, 
    and Sleepy.

    Category II (effects relieved by use of marijuana): Depression, Eye Pressure, Epilepsy, 
    Migraines, Dizzy, Anxiety, Seizure, Headache, Dry Eyes, Pain, Stress, Spasticity, 
    Arthritis, Dry Mouth, and Fatigue.
    
Once the categories were identified, the frequency of each effect was counted in the user 
    reports for each strain of marijuana.  I then created a matrix of the chemical composition
    (as a percent of bud) and effects for each strain as a percent of total count.

    After plotting the data for each effect and assessing if there was any obvious correlation
    that would support a linear regression model, the effects were further divided.
    

## Results

Category I reported effects did not have any strong correlations that would support linear 
    regression analysis, all effects were  analyzed as classification functions.

Category II linear effects are Eye Pressure, Epilepsy, Seizure, Pain, Stress, Spasticity, 
    Arthritis, and Fatigue.

Category II classifier effects are Depression, Migraines, Dizzy, Anxiety, Headache, Dry 
    Eyes, and Dry Mouth.

For linear regression models the p-value of each compound was found utilizing statsmodel. 
    Compounds with high p-values were eliminated from the regression function.
    
The classification effects were further cleaned to remove effects that were in all strains.  
    Euphoric, Relaxed, Happy, Uplifted and Giggly were removed from the classification of 
    category I effects. No category II effects were present in all strains.

## Contributing

Guidelines for those interested in contributing to the project.

## License

Creative commons license

---

