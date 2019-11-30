import numpy as np
import re
import emoji
import mojimoji
import MeCab
from gensim.models import KeyedVectors
import pickle

mecab = MeCab.Tagger("")#Neologd辞書を使う場合はパスを記載してください
model_entity = KeyedVectors.load_word2vec_format("entity_vector.model.bin",binary = True)

with open('shoten_speaker_RF.pickle', mode='rb') as f:
    speaker_clf = pickle.load(f)
with open('shoten_zabuton_RF.pickle', mode='rb') as f:
    zabuton_clf = pickle.load(f)
    
def text_to_vector(text , w2vmodel,num_features):
    kotae = text
    kotae = kotae.replace(',','、')
    kotae = kotae.replace('/n','')
    kotae = kotae.replace('\t','')
    kotae = re.sub(r'\s','',kotae)
    kotae = re.sub(r'^@.[\w]+','',kotae)
    kotae = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+','',kotae)
    kotae = re.sub(r'[!-/:-@[-`{-~ ]+','',kotae)
    kotae = re.sub(r'[：-＠，【】★☆「」。、・]+','',kotae)
    kotae = mojimoji.zen_to_han(kotae,kana = False)
    kotae = kotae.lower()
    kotae = ''.join(['' if character in emoji.UNICODE_EMOJI else character for character in kotae])
    kotae_node = mecab.parseToNode(kotae)
    kotae_line = []
    while kotae_node:
        surface = kotae_node.surface
        meta = kotae_node.feature.split(",")
        if not meta[0] == '記号' and not meta[0] == 'BOS/EOS':
            kotae_line.append(kotae_node.surface)
        kotae_node = kotae_node.next
    feature_vec = np.zeros((num_features), dtype = "float32")
    word_count = 0
    for word in kotae_line:
        try:
            feature_vec = np.add(feature_vec,w2vmodel[word])
            word_count += 1
        except KeyError :
            pass
        if len(word) > 0:
            if word_count == 0:
                feature_vec = np.divide(feature_vec,1)
            else:
                feature_vec = np.divide(feature_vec,word_count)
        feature_vec = feature_vec.tolist()
    return feature_vec

def zabuton_challenge(insert_text):
    vector = np.array(text_to_vector(insert_text,model_entity,200)).reshape(1,-1)
    if(zabuton_clf.predict(vector)[0] == 0):
        print(str(speaker_clf.predict(vector)[0])+"さんに座布団は差し上げません")
    elif(zabuton_clf.predict(vector)[0] < 0):
        print("山田くん！"+str(speaker_clf.predict(vector)[0])+"さんに"+str(zabuton_clf.predict(vector)[0])+"枚差し上げて！")
    elif(zabuton_clf.predict(vector)[0] > 0):
        print("山田くん！"+str(speaker_clf.predict(vector)[0])+"さんの"+str(zabuton_clf.predict(vector)[0])+"枚持ってって！")
    else:
        print("山田くん！エラー出す分類器作った開発者の座布団全部持ってけ！")
        
if __name__ == "__main__":
    while True:
        text = input("答えをどうぞ:")
        zabuton_challenge(text)