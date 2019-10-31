import traceback
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import unidecode
'''-----------------------------------------------------------'''
'''                   Diego Cruz  @di3cruz                    '''
'''-----------------------------------------------------------'''


'''--------------------- Convert text to vectors --------'''


def fn_tdidf_vectorizer(df,name_column):
    try:
        # TF-IDF
        encoderTf = TfidfVectorizer()
        # Create vocabulary with all the text
        encoderTf.fit(df[name_column])
        # Convert each text into vector according to the vocabulary created
        data_encoded = encoderTf.transform(df[name_column])
        df_encoded = pd.DataFrame(data_encoded.toarray(), columns=encoderTf.get_feature_names())
        return df_encoded
    except Exception as e:
        print("Error in function fn_tdidf_vectorizer")
        print("type error: " + str(e))
        print(traceback.format_exc())
        exit()


'''---------- Define stop words and punctuation  --------'''


def stop_words():
    # punctuation to remove
    punctuation_ = list(punctuation)+[u'.', u'[', ']', u',', u';', u'', u')', u'),', u' ',
                                      u'(', "?", "¿", "quiero", "necesito"]
    spanish_stopwords = stopwords.words('spanish') + punctuation_
    return spanish_stopwords


'''---------- Remove stop words and punctuation in text --------'''


def fn_clean_simple_text(str, stop_word):
    try:
        # Lower case text
        sentence = str.lower()
        # Tokenize text
        tokens = [word for word in wordpunct_tokenize(sentence)]
        # Remove stop words and punctuation
        words = [token for token in tokens if token not in stop_word]
        # Join words
        detokenizer = TreebankWordDetokenizer()
        sentence = detokenizer.detokenize(words)
        # return clean text
        return sentence
    except Exception as e:
        print("Error in function fn_clean_simple_text")
        print("type error: " + str(e))
        print(traceback.format_exc())
        exit()


'''---------- Remove accents in text --------'''


def fn_remove_accents_simple_text(str):
    try:
        # Remove accents in str
        unaccented_string = unidecode.unidecode(str)
        return unaccented_string
    except Exception as e:
        print("Error in function fn_remove_accents_simple_text")
        print("type error: " + str(e))
        print(traceback.format_exc())
        exit()


'''---------- Remove stop words and punctuation in data frame  --------'''


def fn_data_frame_clean_text_df(df_clean, name_column):
    try:
        # Remove stop words and punctuation
        df_clean[name_column] = df_clean[name_column].apply(lambda x: fn_clean_simple_text(x, stop_words()))
        df_clean[name_column] = df_clean[name_column].apply(lambda x: fn_remove_accents_simple_text(x))
        return df_clean
    except Exception as e:
        print("Error in function fn_data_frame_clean_text_df")
        print("type error: " + str(e))
        print(traceback.format_exc())
        exit()



