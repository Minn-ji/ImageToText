import nltk
from tqdm import tqdm
from collections import Counter
from PIL import Image
import pandas as pd
import os

def getModel(img_pil, model, mode="simple"):
    result = model(img_pil, mode=mode)
    return result


def get_dict(fns, model, img_root_path='',mode='simple'):
    result_dict_list = []
    texts = []
    images = []
    new_dict = {}
    for fn in tqdm(fns, total=len(fns)):
        temp_dict = {"text":None, "image":None, "mask":None}
        
        img = Image.open(fn).convert("RGB")
        caption = getModel(img_pil=img, model=model, mode=mode)
        temp_dict["text"] = caption
        temp_dict["image"] = os.path.join(img_root_path, os.path.basename(fn))
        result_dict_list.append(temp_dict)

    for dct in result_dict_list:
        texts.append(dct['text'])
        images.append(dct['image'])
        new_dict['image'] = images
        new_dict['text'] = texts
    new_dict = pd.DataFrame(new_dict)
    print("Complete making dictionary.")

    return new_dict


def make_pre_text(my_dict):
    pre_text = []
    tokenizer = nltk.TreebankWordTokenizer()
    tst = [my_dict['text'][i] for i in range(len(my_dict))]
    my_dict['pre_text'] = [None]* len(tst)
    data_path: str = os.path.join(os.path.dirname(__file__), 'data')

    with open(os.path.join(data_path, 'stopwords.txt'),'r') as f:
        stopwords = f.readlines()
    stopwords= [x.replace('\n','') for x in stopwords]    

    for i in tqdm(range(len((tst))), total=len(tst)):
        senten = []
        sentence = tokenizer.tokenize(tst[i])
        for word in sentence:
            if word not in stopwords and word not in senten:
                senten.append(word)
        pre_text.append(' '.join(senten))

    print("Complete preprocessing texts.")
    return pre_text

# keyword 생성
def make_keyword(my_dict, del_color=True):
    rm_list = ['black background', 'hand holding', 'black white', 'person holding', 'dark', 
               'light shining', 'images','background', 'photograph']
    
    data_path: str = os.path.join(os.path.dirname(__file__), 'data')
    
    my_dict['pre_text'] = make_pre_text(my_dict)
    keywords =[]
    with open(os.path.join(data_path, 'color_list.txt'),'r') as f:
        colorwords = f.readlines()
    colorwords= [x.replace('\n','') for x in colorwords] 

    for text in my_dict['pre_text']:
        for stop_wd in rm_list:
            if stop_wd in text:
                text = text.replace(stop_wd,'')
        
        if text.split(' ')[-1] == 'of':
            text = text.split(' ')
            text.remove('of')
            text = ' '.join(text)
      
        if '' in text.split(' '):
            text = text.split(' ')
            text.remove('')
            text = ' '.join(text)
        
        if del_color == True:
            new_word = []
            for word in text.split(' '):
                if word not in colorwords:
                    new_word.append(word)
            text = ' '.join(new_word)

        keywords.append(text)

    return keywords

def common_words(dict, col, num_common_words=50, split=True):
    if split ==True:
        split_words = [sublist.split(' ') for sublist in dict[col]]
        all_words = [item for sublist in split_words for item in sublist]
    else:
        all_words = [item for sublist in dict[col] for item in sublist]
    word_counts = Counter(all_words)
        
    common_words = [f"{word} : {count}개" for (word, count) in word_counts.most_common(num_common_words)]
    words = [word for (word, _) in word_counts.most_common(num_common_words)]

    return common_words, words

# n-gram 생성
def to_ngrams(dict, n):
    ngram_text = []
    split_words = [sublist.split(' ') for sublist in dict['pre_text']]
    for words in split_words:
        ngrams = []
        if len(words) == 2:
            ngrams.append(tuple(words))
        else:
            for b in range(0, len(words) - n + 1):
                ngrams.append(tuple(words[b:b+n]))
        ngram_text.append(ngrams)
    return ngram_text