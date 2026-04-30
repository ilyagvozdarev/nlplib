import numpy as np
import re
import string
from scipy import linalg
from collections import defaultdict



def get_windows(words, C):
    """
    возвращает итератор по словам контекста и самому центральному слову заданной последовательности слов
        C - размер половины окна контекста
        words - последовательность слов
    """
    i = C
    while i < len(words) - C:
        center_word = words[i]
        context_words = words[(i - C) : i] + words[(i + 1) : (i + C+ 1)]
        yield context_words, center_word
        i += 1

def word_to_one_hot_vector(word, word2Ind, V):
    """
    преобразует слово в его one-hot вектор
        V - размер словарного запаса
        word2Ind - словарь индексов слов
    """  
    one_hot_vector = np.zeros(V)
    one_hot_vector[word2Ind[word]] = 1
    return one_hot_vector

def context_words_to_vector(context_words, word2Ind, V):
    """
    возвращает вектор контекста (мешок слов контекста - доля каждого слова словарного запаса в окне контекста)
    по переданным словам контекста.
        V - размер словарного запаса
        context_words - последовательность слов контекста
        word2Ind - словарь индексов слов
    """    
    context_words_vectors = [word_to_one_hot_vector(w, word2Ind, V) for w in context_words]
    context_words_vectors = np.mean(context_words_vectors, axis=0)
    return context_words_vectors

def get_cbow_vectors(words, word2Ind, V, C):
    """
    возвращает итератор по примерам, каждый пример - вектор контекста (мешок слов контекста - доля каждого слова 
    словарного запаса в окне контекста) (входной вектор CBOW) и one-hot вектор самого слова (выходной вектор CBOW).
        C - размер половины окна контекста
        V - размер словарного запаса
        words - последовательность слов
        word2Ind - словарь индексов слов
    """
    for context_words, center_word in get_windows(words, C):
        yield context_words_to_vector(context_words, word2Ind, V), word_to_one_hot_vector(center_word, word2Ind, V)
    
    '''
        тестирование:
        import numpy as np
        from utils_preprocessing import get_vocabs, get_cbow_vectors

        words = ['i', 'am', 'happy', 'because', 'i', 'am', 'learning']

        word2Ind, Ind2word = get_vocabs(words)
        training_examples = get_cbow_vectors(words, word2Ind, len(word2Ind), 2)
        x_array, y_array = next(training_examples)

        print(x_array)
        print(y_array)
    '''



def get_vocabs(data, pad_token = None, unk_token = None, sort = True):
    """
        По последовательности слов формирует словарный запас:
        словарь отображений слов в индексы 
        словарь отображений индексов в слова
        (индексы назначаются в лексикографическом порядке или в порядке в котором они находятся в data)
        Если pad_token задан то создает специальный токен pad_token с индексом равным len(word2Ind).
        Если unk_token задан то создает специальный токен unk_token с индексом равным len(word2Ind)+1.
        
        Input:
            data - последовательность слов
            pad_token - pad токен
            unk_token - unk токен
            sort - True - индексы назначаются в лексикографическом порядке
                   False - индексы назначаются в порядке в котором они находятся в data
        Output:
            word2Ind: словарь отображений слов в индексы
            Ind2Word: словарь отображений индексов в слова
    """

    words = list(set(data))
    if sort:
        words = sorted(words)
    n = len(words)
    idx = 0

    word2Ind = {}
    Ind2word = {}
    
    for word in words:
        word2Ind[word] = idx
        Ind2word[idx] = word
        idx += 1
    
    if pad_token:
        word2Ind[pad_token] = idx
        Ind2word[idx] = pad_token
        idx += 1
    if unk_token:
        word2Ind[unk_token] = idx
        Ind2word[idx] = unk_token
        idx += 1
    
    return word2Ind, Ind2word


def tweet_to_tokens(tweet, stemmer, stopwords):
    '''
        Делает предобработку строки (рассчитан на твит) и возвращает список слов.

        предобработка текста (рассчитан на твит):
        1. удаляет: 
            - биржевые тикеры вида $GE
            - ретвиты в старом стиле "RT"
            - гиперссылки
            - знаки #
        2. токенизация:
            - с переводом в нижний регистр
            - с удалением дескрипторов текста Twitter (пример: @katyperry)
            - с заменой последовательностей одинаковых символов длины 3 или более последовательностями 
              длиной 3 (пример: waaaaaayyyy -> waaayyy)
        3. удаление стоп-слов
        4. удаление пунктуации (string.punctuation)
        5. стемминг

        Input: 
            tweet: строка
        Output:
            tweets_clean: список слов предобработанного твита
    
    '''
    
    from nltk.tokenize import TweetTokenizer
    
    # удаляем биржевые тикеры вида $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    
    # удаляем ретвиты в старом стиле "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    
    # удаляем гиперссылки
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    
    # удаляем знаки #
    tweet = re.sub(r'#', '', tweet)
    
    # токенизация с переводом в нижний регистр, с удалением дескрипторов текста Twitter 
    # (пример: @katyperry), с заменой последовательностей одинаковых символов длины 3 или более 
    # последовательностями длиной 3 (пример: waaaaaayyyy -> waaayyy)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords and # удаляем стоп-слова
            word not in string.punctuation): # удаляем пунктуацию
            stem_word = stemmer.stem(word) # стемминг
            tweets_clean.append(stem_word)
            
    return tweets_clean

    '''
    тестирование
    import nltk
    from nltk.corpus import stopwords, twitter_samples    
    from nltk.stem import PorterStemmer
    from utils2 import tweet_to_tokens

    nltk.download('twitter_samples')
    nltk.download('stopwords')

    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    all_positive_tweets = twitter_samples.strings('positive_tweets.json')

    print(all_positive_tweets[0])

    print("Tweet at training position 0 after processing:")
    print(tweet_to_tokens(all_positive_tweets[0], stemmer, stopwords_english))
    '''


    
def sentence_to_indices(sentence, word2Ind, unk_token = '<UNK>'):
    '''
        делит предложение на токены по ' ' и заменяет токены на их индексы в соответствии 
        со словарем индексов слов word2Ind (для OOV слов индекс - индекс токена unk_token).
        Результат - список индексов токенов

        Args:
            sentence - строка
            word2Ind - словарь индексов слов
            unk_token - токен для OOV слов
        Return:
            tokens_idx - список индексов слов (того же размера что и words)
    '''
    
    return [word2Ind[token] if token in word2Ind else word2Ind['UNK'] for token in sentence.split(' ')]
    
    

def sentences_words_to_indices(words, word2Ind, unk_token = '<UNK>'):
    '''
        заменяет токены в words на их индексы в соответствии со словарем индексов слов word2Ind
        (для OOV слов индекс - индекс токена unk_token).
        Результат - список того же размера что и words.

        Args:
            words - последовательность последовательностей слов
            word2Ind - словарь индексов слов
            unk_token - токен для OOV слов
        Return:
            tokens_idx - список индексов слов (того же размера что и words)
    '''
    
    tokens_indices = []
    
    for sentence in words:
        tokens_indices.append([word2Ind[word] if word in word2Ind else word2Ind[unk_token] for word in sentence])
    
    return tokens_indices

    '''
    более простая реализация для случая когда words - последовательность последовательностей слов
    tokens_idx = []
    
    for word in words:
        token_idx = word2Ind[word] if word in word2Ind else word2Ind[unk_token]
        tokens_idx.append(token_idx)
        
    return tokens_idx
    '''



def tweet_to_indices(tweet, vocab_dict, stemmer, stopwords, unk_token = '<UNK>', verbose=False):
    '''
    Преобразует текст (рассчитан на твит) в список индексов слов текста (для OOV слов индекс - индекс токена unk_token).
    Предобработка текста (формирование списка слов) производится методом tweet_to_tokens.
    
    Input: 
        tweet - текст (строка)
        vocab_dict - словарный запас (словарь индексов слов)
        unk_token - токен для OOV слов
        stemmer - стеммер для предобработки текста
        stopwords - стоп-слова для преедобработки текста
        verbose - Print info durign runtime
    Output:
        tokens_idx - список индексов слов текста
        
    '''  

    words = tweet_to_tokens(tweet, stemmer, stopwords)
    
    if verbose:
        print(f"The unique integer ID for the unk_token is {unk_ID}")
        print("List of words from the processed tweet:")
        print(words)
        
    tokens_idx = []
 
    for word in words:
        token_idx = vocab_dict[word] if word in vocab_dict else vocab_dict[unk_token]
        tokens_idx.append(token_idx) 
    
    return tokens_idx



def batch_context_generator(data, word2Ind, V, C, batch_size):
    """
        для переданной последовательности слов возвращает батч векторов контекста (мешок слов контекста - доля 
        каждого слова словарного запаса в окне контекста) (входной вектор CBOW) и батч соответствующих one-hot 
        векторов самого центрального слова (выходной вектор CBOW).
        
        Input: 
            C - размер половины окна контекста
            V - размер словарного запаса
            batch_size - размер батча
            data - последовательность слов
            word2Ind - словарь индексов слов
        Yield: 
            (batch_x, batch_y):
            batch_x - size (2*C, batch_size)
            batch_y - size (V, batch_size)
    """
    
    batch_x = []
    batch_y = []
    
    for x, y in get_cbow_vectors(data, word2Ind, V, C):
        if len(batch_x) < batch_size:
            batch_x.append(x)
            batch_y.append(y)
        else:
            # транспонируем чтобы размерность батча была второй
            yield np.array(batch_x).T, np.array(batch_y).T
            batch_x = batch_y = []
            
            
            

def batch_seqs_generator(*data_list, batch_size = 16, need_to_indices = False, loop = True, 
                         word2Ind = None, pad_token = '<PAD>', unk_token = '<UNK>', shuffle = False, return_words = True,
                         pad_degree2 = False, same_pad_len = False):
    '''
        Принимает произвольное количество выборок данных, каждая выборка данных - список из нескольких (или одного) 
        наборов примеров (например каждый набор может соответствовать своему классу). Каждый пример - последовательность слов (текст).
        Для каждоый выборки формирует батч слов текстов или индексов слов текстов (в зависимости от need_to_indices) 
        (в каждом батче индексов слов batch_size/N примеров каждого набора (N - количество примеров в наборе)), 
        батч самих слов (возвращается в случае если нужно преобразовывать слова в индексы и нужно возвращать батчи самих слов)
        
        - с возможностью перемешивания данных перед началом выдачи батчей с начала данных
        - возможностью продолжать выдавать данные заново при достижении конца данных (признак loop)
        - с padding используя pad_token (до максимальной длины в батче или степени 2)

        Input: 
            data_list - выборки данных, каждая выборка - список из нескольких (или одного) наборов примеров. Пример выборки данных:
                data = [[[10, 20, 200], [20, 30], [30, 40], [40, 50]],
                        [[11, 22], [22, 33], [33, 44, 444, 4444], [44, 55]]]
            batch_size - размер батча
            loop - True = при достижении конца данных данные перемешиваются и батчи продолжают формироваться
                   False = при достижении конца данных батчи перестают выдаваться (последний неполный батч не выдается)
            pad_token - padding токен
            unk_token - токен для OOV слов
            word2Ind - словарный запас (словарь индексов слов)
            shuffle - нужно ли перемешивать данные
            need_to_indices - нужно ли преобразовывать слова в индексы (используя word2Ind, unk_token)
            return_words - нужно ли возвращать батчи самих слов в случае если нужно преобразовывать слова в индексы
            pad_degree2 - нужно ли увеличивать до ближайшей степени 2
            same_pad_len - если True то длина padding для одного батча для всех выборок будет одинаковой
        Yield:
            (batch_paddded_all, batch_sentences_words):
            batch_paddded_all - size (len(data_list), batch_size, max sample len in batch) батч для каждоый выборки 
                с примерами из каждого набора текстов (каждого набора batch_size/N примеров)
            batch_sentences_words - батч для каждой выборки самих слов (возвращается в случае если нужно преобразовывать 
                слова в индексы и нужно возвращать батчи самих слов)
        
    '''
    
    def pad_batches(batch_all, pad_token, need_to_indices, same_pad_len, pad_degree2):
        
        pad = word2Ind[pad_token] if need_to_indices else pad_token
        
        batch_paddded_all = []
        
        max_sample_len = 0
        if same_pad_len:
            for batch in batch_all:
                max_sample_len = max(max_sample_len, max([len(sample) for sample in batch]))
        if pad_degree2:
                # округление до ближайшей степени двойки
                max_sample_len = 2 ** int(np.ceil(np.log2(max_sample_len)))

        for batch in batch_all:
            if not same_pad_len:
                max_sample_len = max([len(sample) for sample in batch])
            if pad_degree2:
                # округление до ближайшей степени двойки
                max_sample_len = 2 ** int(np.ceil(np.log2(max_sample_len)))
            
            batch_paddded = np.full((batch_size, max_sample_len), pad)
            for i, sample in enumerate(batch):
                batch_paddded[i, :len(sample)] = sample

            batch_paddded_all.append(batch_paddded)
        
        return batch_paddded_all
    

    def get_batch(data, k_data_indices, cur_k_data_indices):
        
        batch = []
        batch_sentences_words = []
            
        for i in range(len(data)):

            i_data = data[i]
            i_data_indices = k_data_indices[i]

            for i_part_batch in range(batch_size // len(data)):

                if cur_k_data_indices[i] >= len(i_data):
                    if not loop:
                        return None, None

                    if shuffle:
                        rnd.shuffle(i_data_indices)

                    cur_k_data_indices[i] = 0

                index = i_data_indices[cur_k_data_indices[i]]
                sentence = i_data[index]
                batch_sentences_words.append(sentence)

                if need_to_indices:
                    sentence = sentences_words_to_indices(sentence, word2Ind, unk_token)

                batch.append(sentence)

                cur_k_data_indices[i] += 1
          
        return batch, batch_sentences_words        
    
    
    import random as rnd

    cur_data_indices = [[0] * len(data) for data in data_list]
    data_indices = [[list(range(len(data_i))) for data_i in data] for data in data_list]
    
    if shuffle:
        for i_data_indices in data_indices:
            for indices in i_data_indices:
                rnd.shuffle(indices)
        
        
    while True:  
        
        batch_all = []
        batch_sentences_words_all = []
        
        for k, data in enumerate(data_list):

            batch, batch_sentences_words = get_batch(data, data_indices[k], cur_data_indices[k])
            
            if not batch:
                return
            
            batch_all.append(batch)
            batch_sentences_words_all.append(batch_sentences_words)

        # padding
        batch_paddded_all = pad_batches(batch_all, pad_token, need_to_indices, same_pad_len, pad_degree2)
  
        if return_words and need_to_indices:
            yield batch_paddded_all, batch_sentences_words
        else:
            yield batch_paddded_all
                        
                    
    '''
    тестирование 1:
        data1 = [[[10, 20, 200], [20, 30], [30, 40], [40, 50]],
             [[11, 22], [22, 33], [33, 44, 444, 4444], [44, 55]]]

        data2 = [[[50, 60], [60, 70], [70, 80], [80, 90]],
                 [[55, 66], [66, 77], [77, 88], [88, 99]]]

        gen = batch_seqs_generator(data1, data2, batch_size=2, need_to_indices = False, loop=False, shuffle=True, same_pad_len = True)

        next(gen)
        
    тестирование 2:
        Named Entity Recognition (NER) - LSTM.ipynb
    
    '''


def batch_sample_generator(batch_size, data_x, data_y, shuffle=True):
    '''
      Возвращает батч примеров данных и батч меток (в виде 2-х элементного кортежа).
      
      - С возможностью перемешивания данных перед началом выдачи бачей с начала данных.
        При достижении конца данных, данные перемешиваюстя и батчи выдаются далее.
    
      Input: 
        batch_size - batch size
        data_x - данные (список примеров)
        data_y - список меток
        shuffle - нужно ли перемешивать данные перед началом выдачи бачей с начала данных
        
      Yield:
        (X, Y):
        X - (список размера batch_size) батч индексов примеров 
        Y - (список размера batch_size) батч меток 
    '''
    
    indices_data = [*range(len(data_x))] 
    
    if shuffle:
        random.shuffle(indices_data)
    
    index = 0 

    while True:
        X = [0] * batch_size 
        Y = [0] * batch_size 
        
        for i in range(batch_size):
            if index >= len(data_x):
                index = 0
                if shuffle:
                    rnd.shuffle(indices_data) 
                    
            X[i] = data_x[indices_data[index]] 
            Y[i] = data_y[indices_data[index]]          
            index += 1
        
        yield((X, Y))
    
    
    '''
    тестирование
    def test_data_generator():
    x = [1, 2, 3, 4]
    y = [xi ** 2 for xi in x]
    
    generator = data_generator(3, x, y, shuffle=False)

    assert np.allclose(next(generator), ([1, 2, 3], [1, 4, 9])),  "First batch does not match"
    assert np.allclose(next(generator), ([4, 1, 2], [16, 1, 4])), "Second batch does not match"
    assert np.allclose(next(generator), ([3, 4, 1], [9, 16, 1])), "Third batch does not match"
    assert np.allclose(next(generator), ([2, 3, 4], [4, 9, 16])), "Fourth batch does not match"

    print("\033[92mAll tests passed!")

    test_data_generator()
    '''
    