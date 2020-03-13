from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
import os
from definitions import ROOT_DIR
import numpy as np

####################################################
# Tokenizer for removing stop words, transforming stems
####################################################

def clean_text(original_text):

    stop_words = set(stopwords.words('english'))

    stemmer = PorterStemmer()

    tokenizer = RegexpTokenizer(r'\w+')

    word_tokens = tokenizer.tokenize(original_text)

    removed_stop_words = [word for word in word_tokens if word not in stop_words]

    #clean_sentence = ' '.join([stemmer.stem(word) for word in word_tokens])

    clean_list = [stemmer.stem(word) for word in removed_stop_words]

    return clean_list

####################################################
# Gets the tfidf matrix from the list
####################################################

def getTFIDF(movie_overview):


    # Create TFIDF vector
    tfidf = TfidfVectorizer(tokenizer=clean_text,
                            analyzer='word',
                            lowercase=True,
                            dtype=np.float32,
                            min_df=.001,
                            max_df=.9,
                            max_features=500
                            )

    tfidf_sparse_matrix = tfidf.fit_transform(movie_overview)

    tfidf_word_list = tfidf.get_feature_names()

    return tfidf_sparse_matrix, pd.DataFrame(tfidf_sparse_matrix.toarray(), columns=tfidf_word_list),tfidf_word_list

def getHashVector(movie_overview, features=500):

    hash = HashingVectorizer(
        tokenizer=clean_text,
        analyzer='word',
        lowercase=True,
        dtype=np.float32,
        n_features=features
    )

    vector = hash.fit_transform(movie_overview)


    return vector.toarray()

def getCountVector(movie_overview, features=500):

    count = CountVectorizer(
        tokenizer=clean_text,
        analyzer='word',
        lowercase=True,
        dtype=np.float32,
        max_features=features
    )

    vector = count.fit_transform(movie_overview)

    return vector.toarray()

def test_tfidf():
    input_text=[
        'Far far away, behind the word mountains, far from the countries Vokalia and Consonantia, there live the blind texts. Separated they live in Bookmarksgrove right at the coast of the Semantics, a large language ocean. A small river named Duden flows by their place and supplies it with the necessary regelialia. It is a paradisematic country, in which roasted parts of sentences fly into your mouth. Even the all-powerful Pointing has no control about the blind texts it is an almost unorthographic life One day however a small line of blind text by the name of Lorem Ipsum decided to leave for the far World of Grammar. The Big Oxmox advised her not to do so, because there were thousands of bad Commas, wild Question Marks and devious Semikoli, but the Little Blind Text didn’t listen. She packed her seven versalia, put her initial into the belt and made herself on the way. When she reached the first hills of the Italic Mountains, she had a last view back on the skyline of her hometown Bookmarksgrove, the headline of Alphabet Village and the subline of her own road, the Line Lane. Pityful a rethoric question ran over her cheek, then',
        '''One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed in his bed into a horrible vermin. He lay on his armour-like back, and if he lifted his head a little he could see his brown belly, slightly domed and divided by arches into stiff sections. The bedding was hardly able to cover it and seemed ready to slide off any moment. His many legs, pitifully thin compared with the size of the rest of him, waved about helplessly as he looked. "What's happened to me?" he thought. It wasn't a dream. His room, a proper human room although a little too small, lay peacefully between its four familiar walls. A collection of textile samples lay spread out on the table - Samsa was a travelling salesman - and above it there hung a picture that he had recently cut out of an illustrated magazine and housed in a nice, gilded frame. It showed a lady fitted out with a fur hat and fur boa who sat upright, raising a heavy fur muff that covered the whole of her lower arm towards the viewer. Gregor then turned to look out the window at the dull weather. Drops''',
        '''A wonderful serenity has taken possession of my entire soul, like these sweet mornings of spring which I enjoy with my whole heart. I am alone, and feel the charm of existence in this spot, which was created for the bliss of souls like mine. I am so happy, my dear friend, so absorbed in the exquisite sense of mere tranquil existence, that I neglect my talents. I should be incapable of drawing a single stroke at the present moment; and yet I feel that I never was a greater artist than now. When, while the lovely valley teems with vapour around me, and the meridian sun strikes the upper surface of the impenetrable foliage of my trees, and but a few stray gleams steal into the inner sanctuary, I throw myself down among the tall grass by the trickling stream; and, as I lie close to the earth, a thousand unknown plants are noticed by me: when I hear the buzz of the little world among the stalks, and grow familiar with the countless indescribable forms of the insects and flies, then I feel the presence of the Almighty, who formed us in his own image, and the breath '''
    ]

    tfidf_sparse_matrix, tfidf_df, tfidf_word_list = getTFIDF(input_text)

    print(tfidf_word_list)

    #non_vector_tfidf = list(tfidf_matrix.toarray())

    print(tfidf_df)

    print(type(tfidf_df))

#test_tfidf()

def clean_movie_overview(overviews):

    clean_overviews=[]

    for o in list(overviews['overview']):
        clean_overviews.append(' '.join(clean_text(o)))

    overviews['clean_overview']=clean_overviews

    return overviews

#clean_movie_overview()



