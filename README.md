# Awesome Feature Engineering for Machine Learning

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of resources dedicated to Feature Engineering Techniques for Machine Learning 

Maintainers - [Andrei Khobnia](https://github.com/aikho)

This page is licensed under [Creative Commons Attribution-Noncommercial-ShareAlike 3.0 Unported License](https://creativecommons.org/licenses/by-nc-sa/3.0/)

Please feel free to create [pull requests](https://github.com/aikho/awesome-feature-engineering/pulls).


## Contents

 - [Numeric Data](#numeric-data)
   - [Scaling](#scaling)
   - [Ranking](#ranking)
   - [Box-Cox Transformation](#box-cox-transformation)
   - [Yeo-Johnson Transformation](#yeo-johnson-transformation)
   - [Feature Interactions](#feature-interactions)
   - [Clustering Features](#clustering-features)
   - [t-SNE Features](#t-sne-features)
 - [Textual Data](#textual-data)
   - [Bag of Words](#bag-of-words)
   - [TFIDF](#tfidf)
   - [Word Embeddings](#word-embeddings)
   - [Subword Embeddings](#subword-embeddings)
   - [Pattern Features](#pattern-features)
   - [Lexicon Features](#lexicon-features)
 - [Image Data](#image-data)
   - [Computer Vision Algorithm Features](#computer-vision-algorithm-features)
   - [Image Statistics Features](#image-statistics-features)
   - [OCR Features](#ocr-features)
   - [Deep Learning Features](#deep-learning-features)
 - [Categorical Data](#categorical-data)
   - [One Hot Encoding](#one-hot-encoding)
   - [Mean Encoding](#mean-encoding)
 - [Time Series Data](#time-series-data)
   - [Rolling Window Features](#rolling-window-features)
   - [Lag Features](#lag-features)
 - [Geospatial Data](#geospatial-data)


## Numeric Data
* [Understanding Feature Engineering (Part 1) -- Continuous Numeric Data](https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b)
### Scaling
* [sklearn.preprocessing.MinMaxScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
* [sklearn.preprocessing.StandartScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
### Ranking
* [Ranking](https://en.wikipedia.org/wiki/Ranking)
* [scipy.stats.rankdata](https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.stats.rankdata.html)
### Box-Cox Transformation
* [scipy.stats.boxcox](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html)
* `np.log (x + const)`
### Yeo-Johnson Transformation
* [Yeo-Johnson Transformation](https://gist.github.com/mesgarpour/f24769cd186e2db853957b10ff6b7a95)
### Feature Interactions
* [Featuretools](https://docs.featuretools.com/)
* [sklearn.preprocessing.PolynomialFeatures](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
* Divisions
* Other interactions
### Clustering Features
* [How to create New Features using Clustering!!](https://towardsdatascience.com/how-to-create-new-features-using-clustering-4ae772387290)
### t-SNE Features
* [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)
* [Automatic feature extraction with t-SNE](https://medium.com/jungle-book/automatic-feature-extraction-with-t-sne-62826ce09268)


## Textual Data
* [Understanding Feature Engineering (Part 3) -- Traditional Methods for Text Data](https://towardsdatascience.com/understanding-feature-engineering-part-3-traditional-methods-for-text-data-f6f7d70acd41)
### Bag of Words
* [Bag-of-words model](https://en.wikipedia.org/wiki/Bag-of-words_model)
* [A Gentle Introduction to the Bag-of-Words Model](https://machinelearningmastery.com/gentle-introduction-bag-words-model/)
* [sklearn.feature_extraction.text.CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
* [sklearn.feature_extraction.DictVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html)
* [sklearn.feature_extraction.FeatureHasher](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html)
### TFIDF
* [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
* [sklearn.feature_extraction.text.TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
### Word Embeddings
* [Word embedding](https://en.wikipedia.org/wiki/Word_embedding)
* [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
* [Gensim: models.word2vec – Word2vec embeddings](https://radimrehurek.com/gensim/models/word2vec.html)
* [fastText](https://fasttext.cc/)
* [Word2Vec and FastText Word Embedding with Gensim](https://towardsdatascience.com/word-embedding-with-word2vec-and-fasttext-a209c1d3e12c)
* [Do Pretrained Embeddings Give You The Extra Edge?](https://www.kaggle.com/sbongo/do-pretrained-embeddings-give-you-the-extra-edge)
### Subword Embeddings
* [Pre-trained subword embeddings in 275 languages, based on Byte-Pair Encoding (BPE)](https://github.com/bheinzerling/bpemb)
### Pattern Features
* [ClearTK - Feature Extraction Tutorial](https://cleartk.github.io/cleartk/docs/tutorial/feature_extraction.html)
* Regular Expressions
### Lexicon Features
* [Named Entity Recognition with Bidirectional LSTM-CNNs (arXiv:1511.08308)](https://arxiv.org/abs/1511.08308v4)


## Image Data
### Computer Vision Algorithm Features
* [Feature extraction and similar image search with OpenCV for newbies](https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774)
* [OpenCV -- Feature Detection and Description](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html)
* [SimpleCV.Features package](http://simplecv.readthedocs.io/en/latest/SimpleCV.Features.html)
* [Scikit-image feature module](http://scikit-image.org/docs/stable/api/skimage.feature.html)
### Image Statistics Features
* [ImageStat Module -- Pillow](http://pillow.readthedocs.io/en/3.1.x/reference/ImageStat.html)
### OCR Features
* [A Python wrapper for Google Tesseract](https://github.com/madmaze/pytesseract)
### Deep Learning Features
* [Keras pre-trained models feature extraction](https://keras.io/applications/)
* [Using Keras’ Pre-trained Models for Feature Extraction in Image Clustering](https://medium.com/@franky07724_57962/using-keras-pre-trained-models-for-feature-extraction-in-image-clustering-a142c6cdf5b1)


## Categorical Data
* [Understanding Feature Engineering (Part 2) -- Categorical Data](https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63)
### One Hot Encoding
* [Why One-Hot Encode Data in Machine Learning?](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)
* [How to One Hot Encode Sequence Data in Python](https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/)
* [sklearn.preprocessing.OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
* [Keras - to_categorical](https://keras.io/utils/#to_categorical)
### Mean Encoding
* [Likelihood encoding of categorical features](https://www.kaggle.com/tnarik/likelihood-encoding-of-categorical-features)
* [Python target encoding for categorical features](https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features)


## Time Series Data
* [Automatic extraction of relevant features from time series](http://tsfresh.readthedocs.io)
* [Basic Feature Engineering With Time Series Data in Python](https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/)
### Rolling Window Features
* [pandas.DataFrame.rolling](https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.DataFrame.rolling.html)
### Lag Features
* [Use pandas to lag your timeseries data in order to examine causal relationships](https://medium.com/@NatalieOlivo/use-pandas-to-lag-your-timeseries-data-in-order-to-examine-causal-relationships-f8186451b3a9)


## Geospatial Data
* [Geospatial Feature Engineering and Visualization](https://www.kaggle.com/camnugent/geospatial-feature-engineering-and-visualization)
* [Intro to Geospatial Data using Python](https://github.com/SocialDataSci/Geospatial_Data_with_Python/blob/master/Intro%20to%20Geospatial%20Data%20with%20Python.ipynb)


[Back to Top](#contents)
