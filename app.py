# Core Packages
import streamlit as st
import os

# NLP Packages
from textblob import TextBlob
import spacy

# Plotting Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Miscellaneous Packages
from nltk import ngrams, FreqDist
from nltk.tokenize import WhitespaceTokenizer
import re
import json
import operator
import emoji

# Web Scraping Packages
from bs4 import BeautifulSoup
from urllib.request import urlopen

def get_text(raw_url):
    page = urlopen(raw_url)
    soup = BeautifulSoup(page, features='lxml')
    fetched_text = ' '.join(map(lambda p: p.text, soup.find_all('p')))

    return fetched_text

@st.cache
def entity_analyzer(text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(text)
    entities = [(entity.text, entity.label_)for entity in docx.ents]
    allData = ['Entities":{}\n'.format(entities)]

    if entities:
        return allData

@st.cache(allow_output_mutation=True)
def pos_tagging(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    pos_dict = dict()

    for token in doc:
        if token.pos_ in pos_dict:
            pos_dict[token.pos_] += 1
        else:
            pos_dict[token.pos_] = 1

    if pos_dict:
        return pos_dict

def remap_keys(mapping):
    return [{'occurrences': v, 'succesive_tokens': k} for k, v in mapping.items()]

def instanciate_dict(message):
    tk = WhitespaceTokenizer()
    tokens = tk.tokenize(message)

    all_counts = dict()
    sorted_dict = dict()

    for size in 1, 2, 3:
        all_counts[size] = FreqDist(ngrams(tokens, size))

    for index in range(1, 4):
        all_counts[index] = {k : v for k,v in all_counts[index].items() if v >= 2}
        sorted_dict[index] = dict(sorted(all_counts[index].items(), key=operator.itemgetter(1),reverse=True))

    return sorted_dict

def main():

    # Title
    st.title("NLP Doctor v2.0 (via Streamlit)")
    st.markdown("""
    	#### Guidelines:
    	1. Simply input/extract a text
        2. Select an option
        3. Enjoy the insights
    	""")

    option = st.selectbox(
        'What would you like the text source to be?',
        ('Manual Input', 'Parse from URL'))

    if option == 'Manual Input':
        message = st.text_area("Enter Text:", "Type here...", key='sentiment')
    else:
        raw_url = st.text_input('Enter URL:','Type here...')

        regex = re.compile(
            r'^(?:http|ftp)s?://' # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
            r'localhost|' #localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
            r'(?::\d+)?' # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        if re.match(regex, raw_url) is not None:
            message = get_text(raw_url)

        if st.button('Analyse Text from URL', key='text-analysis'):
            if re.match(regex, raw_url) is not None:
                result = get_text(raw_url)

                tk = WhitespaceTokenizer()
                tokens = tk.tokenize(result)

                full_counter = len(tokens)
                st.success("There are {} words and {} characters in the entire text.".format(full_counter, len(result)))
            else:
                st.error('Invalid URL.')

        preview_length = st.slider("Length to Preview", 0, 300, 50)

        if st.button('Extract Preview', key='text-extract'):
            if re.match(regex, raw_url) is not None:
                result = get_text(raw_url)

                tk = WhitespaceTokenizer()
                tokens = tk.tokenize(result)
                full_counter = len(tokens)

                preview_text = tokens[:preview_length]
                preview_text = ' '.join(word for word in preview_text)

                st.info(preview_text)
            else:
                st.error('Invalid URL.')

    st.subheader('Options:')

    if st.checkbox('Show Repeating Ngrams'):
        if 'message' in locals():
            sorted_dict = instanciate_dict(message)
            ngram_order = st.radio(
                "Select Order:",
                ('1st Order (Unigrams)', '2nd Order (Bigrams)', '3rd Order (Trigrams)'),
                index=1)

            if ngram_order == '1st Order (Unigrams)':
                nlp_result = json.dumps(remap_keys(sorted_dict[1]))

                if sorted_dict[1]:
                    st.json(nlp_result)
                else:
                    st.warning('There are no repeating unigrams in this text.')

            if ngram_order == '2nd Order (Bigrams)':
                nlp_result = json.dumps(remap_keys(sorted_dict[2]))
                
                if sorted_dict[2]:
                    st.json(nlp_result)
                else:
                    st.warning('There are no repeating bigrams in this text.')
            
            if ngram_order == '3rd Order (Trigrams)':
                nlp_result = json.dumps(remap_keys(sorted_dict[3]))
                
                if sorted_dict[3]:
                    st.json(nlp_result)
                else:
                    st.warning('There are no repeating trigrams in this text.')
        else:
            st.warning('No valid input detected.')

    if st.button("Plot POS Tags Frequency", key='pos'):
        if 'message' in locals():
            pos_dict = pos_tagging(message)

            if pos_dict:
                labels = [pos for pos in pos_dict.keys()]
                sizes = [count for count in pos_dict.values()]

                most_common = max(pos_dict.items(), key=operator.itemgetter(1))
                st.success('The most commond POS tag is: {} with an occurrence count of {}'
                    .format(most_common[0], most_common[1]))

                st.set_option('deprecation.showPyplotGlobalUse', False)
                fig = plt.figure()
                ax = fig.add_axes([0,0,1,1])
                ax.axis('equal')
                ax.pie(sizes, labels = labels,autopct='%1.2f%%')

                st.pyplot()
            else:
                st.warning('There are no recognised POS tags in this text.')
        else:
            st.warning('No valid input detected.')

    if st.button("Extract Named Entities", key='ner'):
        if 'message' in locals():
            entity_result = entity_analyzer(message)

            if entity_result:
                st.json(entity_result)
            else:
                st.warning('There are no recognised entities in this text.')
        else:
            st.warning('No valid input detected.')

    if st.button("Perform Sentiment Analysis", key='sentiment'):
        if 'message' in locals():
            blob = TextBlob(message)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            output_text = 'This text has a polarity score of {} (ranges from -1 to 1) and it\'s {}% subjective '

            if polarity > 0.0:
                custom_emoji = ':smile:'
                output_emoji = emoji.emojize(custom_emoji, use_aliases=True)
            elif polarity < 0.0:
                custom_emoji = ':disappointed:'
                output_emoji = emoji.emojize(custom_emoji, use_aliases=True)
            else:
                output_emoji = emoji.emojize(':expressionless:', use_aliases=True)

            output_text += output_emoji
            st.success(output_text.format(round(polarity, 2), int(round(subjectivity, 2) * 100)))
        else:
            st.warning('No valid input detected.')

    st.sidebar.subheader("About App")
    st.sidebar.text("NLP Doctor v2.0")
    st.sidebar.info("Perform various fundamental NLP tasks on the go either from a manually inputted \
        or from a URL extracted text, such as: Ngram extraction, plotting POS tags frequency, named \
        entity extraction or sentiment anaylsis.")

    st.sidebar.subheader("By")
    st.sidebar.text('Teodor Tița')

    st.sidebar.markdown(
    """<a style='text-decoration: none' href="https://www.linkedin.com/in/teodor-tita/">LinkedIn</a> \
        | <a style='text-decoration: none' href="https://github.com/teodortita">GitHub</a>""", 
        unsafe_allow_html=True)

if __name__ == '__main__':
    main()
