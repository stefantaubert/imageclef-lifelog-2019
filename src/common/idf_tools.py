from sklearn.feature_extraction.text import TfidfVectorizer

"""I don't use lambda because of speed."""
def no_tokenizer(doc):
    return doc

def get_idf(values):
    empty_values = True
    for val in values:
        if len(val) != 0:
            empty_values = False
            break

    if empty_values: return values

    tfidf = TfidfVectorizer(tokenizer=no_tokenizer, preprocessor=no_tokenizer)
    #print(values)
    tfidf.fit(values)
    feature_names = tfidf.get_feature_names()
    idf_values = {}
    
    for i in range(len(feature_names)):
        idf = tfidf.idf_[i]
        feature_name = feature_names[i]
        idf_values[feature_name] = idf
    
    return idf_values

if __name__ == "__main__":
    test = [["abc", "def"], ["abc"], ["abc", "xyz"], []]
    idf = get_idf(test)
    print(idf)