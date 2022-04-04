The goal of this project is to provide an interactive app that visualizes high dimensional word embeddings in vector space. These word embedding visualizations will be a tool to interpret word semantics and relationships between words. It will also be an excellent tool for those learning the concept of word embeddings. 

The MVP is a [deployed Streamlit app](https://share.streamlit.io/katereiss/visual-word-embeddings/main/streamlit_test.py) that allows a user to input any word and visualize that word, along with its 10 closest words (in cosine similarity), in three dimensional space. The user can rotate and zoom in and out of the graph to better interact with the three dimensions.

Additions will include the following:
- Error message if a word is not in the vocabulary
- Using TensorBoard UI instead of Plotly 
- Add other models, allow user to choose model
- Improve speed



