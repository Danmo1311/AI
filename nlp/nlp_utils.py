import traceback
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
from nltk.corpus import stopwords
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


'''---------- Remove stop words and punctuation  --------'''

def fn_clean_text(df,name_column):
    try:
        # stopword list to use
        spanish_stopwords = stopwords.words('spanish') + ["quiero", "necesito"]
        # punctuation to remove
        non_words = list(punctuation)
        # we add spanish punctuation
        punctuation_ = non_words + ['¿', '¡']
        # Remove stop words and punctuation
        df[name_column]= df[name_column].apply(lambda x: ' '.join([word for word in x.split() if word not in (spanish_stopwords)]))
        df[name_column]= df[name_column].apply(lambda x: ' '.join([word for word in x.split() if word not in (punctuation_)]))
        return df
    except Exception as e:
        print("Error in function fn_clean_text")
        print("type error: " + str(e))
        print(traceback.format_exc())
        exit()


