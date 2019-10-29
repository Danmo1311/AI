import traceback
from sklearn.preprocessing import LabelEncoder

'''-----------------------------------------------------------'''
'''                   Diego Cruz  @di3cruz                    '''
'''-----------------------------------------------------------'''


'''--------------------- Encoder label --------'''

# returns the variable to target numerically


def fn_label_encoder(df, name_column):
    try:
        # Encoder label target
        le = LabelEncoder()
        y = le.fit_transform(df[name_column])
        return y
    except Exception as e:
        print("Error in function fn_label_encoder")
        print("type error: " + str(e))
        print(traceback.format_exc())
        exit()

