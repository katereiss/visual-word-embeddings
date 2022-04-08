# Visualizing Word Embeddings

In a previous [project](https://github.com/katereiss/nlp-word-embeddings), I compared word embeddings between models trained on a Wikipedia corpus and a Twitter corpus. The same words had different vectors associated with them, different words closest to them in vector space, and different semantic meanings. One problem I encountered during this project was the curse of dimensionality: reducing 200-dimensional vectors down to 2 or 3 meant losing important information and patterns. Many of the visualizations of these embeddings looked like random dots in space, with no visible pattern.

I had two goals in mind when I created this app. I wanted to let users interact with word embeddings and machine-learned vocabulary by being able to input their own words. I also wanted to see if certain words had specific patterns in vector space.

**Design**

In the [Streamlit](https://share.streamlit.io/katereiss/visual-word-embeddings/main/streamlit_test.py) app, users can input a word or click on the “Random Word” button. An interactive 3D graph shows that word plus the nearest words based on cosine similarity. A slider determines how many nearest words are presented, from 5 to 50. Below the graph is a table with the nearest words ranked by cosine similarity to the input word.

**Data**

The pre-trained word embeddings used in this project were 50-dimensional [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings from a 2014 Wikipedia dump. I used 50-dimensional embeddings in hopes of reducing computational time. The model was trained on 6 billion tokens, resulting in a vocabulary of 400,000 words and corresponding 50-dimensional vectors.

**Algorithms**

Caching was used in hopes of reducing the time it took for the application to load. Errors were handled with try/except clauses. PCA was used for dimensionality reduction.

**Tools**

- Streamlit for app deployment
- Plotly for 3D visualization
- Gensim for data acquisition
- Scikit-learn for PCA
- Pandas and numpy for analysis

**Communication**

The [Streamlit](https://share.streamlit.io/katereiss/visual-word-embeddings/main/streamlit_test.py) app is available for use. It takes a few minutes to load.

