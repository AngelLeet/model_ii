from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.callbacks import ModelCheckpoint


model_lstm = Sequential()
model_lstm.add(Embedding(10000, 128, input_length=29))
model_lstm.add(SpatialDropout1D(0.5))
model_lstm.add(LSTM(40, return_sequences=True))
model_lstm.add(LSTM(40))
model_lstm.add(Dense(91, activation='sigmoid'))
model_lstm_save_path = './nlp/best_model_lstm.h5'
tokinezer_text = Tokenizer(num_words=10000)
model_lstm.load_weights(model_lstm_save_path)
comment = input()

sequence = tokinezer_text.texts_to_sequences([comment])
data = pad_sequences(sequence, maxlen=29)

rez = {'sports': 1, 'news': 2, 'life': 3, 'money': 4, 'tech': 5, 'travel': 6, 'opinion': 7, 'entertainment': 8, 'weather': 9, 'college': 10, 'gameon': 11, 'happyeverafter': 12, 'theoval': 13, 'todayinthesky': 14, 'nletter': 15, 'popcandy': 16, 'cruiselog': 17, 'onpolitics': 18, 'driveon': 19, 'experience': 20, 'dispatches': 21, 'your': 22, 'take': 23, 'nation': 24, 'idolchatter': 25, 'cybertruth': 26, 'hotelcheckin': 27, 'interactives': 28, 'ondeadline': 29, 'reality': 30, 'technologylive': 31, 'lightpost': 32, 'virtual': 33, 'community': 34, 'hub': 35, 'entertainthis': 36, 'augmented': 37, 'home': 38, 'graphics': 39, 'aerial': 40, 'journalism': 41, 'some': 42, 'bistro': 43, 'gamehunters': 44, 'grateful': 45, 'world': 46, 'ncaaw': 47, 'mlb': 48, 'ncaaf': 49, 'system': 50, 'movies': 51, 'fantasy': 52, 'cars': 53, 'cycling': 54, 'a5c9ad5f': 55, 'fc8d': 56, '49ff': 57, 'ba16': 58, '6279e1e40e63': 59, '25d6c598': 60, '048d': 61, '487b': 62, 'a992': 63, '573824906bc6': 64, '39e60bae': 65, 'e602': 66, '48ee': 67, 'b80b': 68, '5745494b8bec': 69, '13842f43': 70, '997a': 71, '4d92': 72, '9d3a': 73, '81cf4f44c9da': 74, 'section': 75, 'las': 76, 'vegas': 77, 'olympics': 78, 'ugc': 79, 'error': 80, 'appinsider': 81, 'olemisssports': 82, 'predatorsinsider': 83, 'yourtake': 84, 'nolesports': 85, 'military': 86, 'washington': 87, 'lifestyle': 88, 'advice': 89, 'results': 90}
result = model_lstm.predict(data)
n = 0
for i in result[0]:

    if i == result.max():
        print(i)
        print(list(rez.items())[n-1][0])
    n += 1