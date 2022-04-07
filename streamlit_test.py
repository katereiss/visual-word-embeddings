from turtle import onclick
import streamlit as st
import pandas as pd
import numpy as np

import gensim.downloader as api
from gensim.models import KeyedVectors, Word2Vec

import plotly.graph_objs as go
from sklearn.decomposition import PCA

st.title('Visualizing Word Embeddings')

model_wikipedia50 = api.load("glove-wiki-gigaword-50")
# model_twitter25 = api.load("glove-twitter-25")

# st.selectbox('Select', ['GloVe Wikipedia 50 dimensions','GloVe Twitter 25 dimensions'])

# Create a text element and let the reader know the data is loading.
# data_load_state = st.text('Loading data...')
# Load pre-trained embeddings
data = model_wikipedia50
# Notify the reader that the data was successfully loaded.
# data_load_state.text('Loading data...done!')

st.header('What Are Word Embeddings?')

st.write('Word embeddings are vector representations of words. Words with similar meanings are closer together in vector space.')

st.write('Search for any word below and the graph will show the word embeddings of the most similar words!')

title = st.text_input('Word (Examples: "archaeologists", "baseball", "Google", "2006", "meningococcus", "Ushuaia")', 'dog').lower()

if st.button('Random Word'):
    title = model_wikipedia50.index_to_key[np.random.randint(0,399999)]

st.write('Selected Word: ', title)

num = st.slider('Select number of similar words:', 5, 50, 10)

@st.cache
def append_list(sim_words, words):
    
    list_of_words = []
    
    for i in range(len(sim_words)):
        
        sim_words_list = list(sim_words[i])
        sim_words_list.append(words)
        sim_words_tuple = tuple(sim_words_list)
        list_of_words.append(sim_words_tuple)
        
    return list_of_words

try: 
    word_list = [model_wikipedia50.most_similar(f'{title}', topn=num)[i][0] for i in range(num)]
    words = str(word_list).replace("'","")
    words = words.strip("[]")
    data=[model_wikipedia50.most_similar(f'{title}', topn=num)[i] for i in range(num)]
    df = pd.DataFrame(data=data, columns= ['Word','Cosine Similarity'])
    df.index +=1

    input_word = title.lower() + ',' + words
    user_input = [x.strip() for x in input_word.split(',')]
    result_word = []

    
    for words in user_input:
        
            sim_words = model_wikipedia50.most_similar(words, topn = 0)
            sim_words = append_list(sim_words, words)
                
            result_word.extend(sim_words)
        
    similar_word = [word[0] for word in result_word]
    similarity = [word[1] for word in result_word] 
    similar_word.extend(user_input)
    labels = [word[2] for word in result_word]
    label_dict = dict([(y,x+1) for x,y in enumerate(set(labels))])
    color_map = [label_dict[x] for x in labels]

except KeyError:
    st.write('Sorry! \"',title,"\" not in vocabulary." )
    
@st.cache
def display_pca_scatterplot_3D(model=model_wikipedia50, user_input=None, words=None, label=None, color_map=None, topn=5, sample=10):

    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [ word for word in model.vocab ]
    
    word_vectors = np.array([model[w] for w in words])
    
    
    three_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:3]
    # For 2D, change the three_dim variable into something like two_dim like the following:
    # two_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]

    data = []
    count = 0
    
    for i in range (len(user_input)):
        trace = go.Scatter3d(
        x = three_dim[count:count+topn,0], 
        y = three_dim[count:count+topn,1],  
        z = three_dim[count:count+topn,2],
        text = words[count:count+topn],
        name = user_input[i],
        textposition = "top center",
        textfont_size = 20,
        mode = 'markers+text',
        marker = {
            'size': 10,
            'opacity': 0.8,
            'color': 2
        }

                )
                
        # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
        # variable three_dim, use the variable that we have declared earlier (e.g two_dim)

        data.append(trace)
        count = count+topn

    trace_input = go.Scatter3d(
                    x = three_dim[count:,0], 
                    y = three_dim[count:,1],  
                    z = three_dim[count:,2],
                    text = words[count:],
                    name = 'input words',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 1,
                        'color': 'black'
                    }
                    )

    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
    # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
            
    data.append(trace_input)
    
# Configure the layout


    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=False,
        legend=dict(     
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 700,
        height = 700
        )


    

    plot_figure = go.Figure(data = data, layout = layout)
    st.plotly_chart(plot_figure)
    
try: display_pca_scatterplot_3D(model_wikipedia50, user_input, similar_word, labels, color_map)
except NameError:
    pass

try: 
    st.write(num, 'Most Similar Words:')
    st.table(df)
except NameError:
    pass

