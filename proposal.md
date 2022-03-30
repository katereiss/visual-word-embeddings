# Visualizing Word Embeddings on a Streamlit App

The goal of this project is to provide an interactive app that visualizes high dimensional word embeddings in vector space. These word embedding visualizations will be a tool to interpret word semantics and relationships between words. It will also be an excellent tool for those learning the concept of word embeddings. 

The MVP will be an interactive Streamlit app that visualizes the 10 closest words to a given word. For example, the following image visualizes the closest 10 words to the word “dog” taken from the 50-dimensional word embeddings from Wikipedia:

<img width="789" alt="Screen Shot 2022-03-30 at 5 51 10 PM" src="https://user-images.githubusercontent.com/84412675/160938162-19c3f35e-1bb5-4bcd-93eb-9cf20d18f90a.png">


PCA is used to reduce the dimensions from 50 to 2 or 3. While the above image is a screenshot, it can be moved with a cursor to better show the representation of these words in three dimensions.


Additional features will include interactive vectors in vector space. The user would input a word, and the word would be represented as a two or three dimensional vector in vector space. One could look at multiple words at once to show semantic similarity or difference, and analogy tests could be visualized in this way.

Another addition would be creating a pipeline that could easily substitute the 50-dimensional Wikipedia embeddings with other pre-trained Gensim embeddings, such as those from Twitter corpora and those of higher dimensions. 

