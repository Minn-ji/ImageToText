from my_clip_interrogator import Config, Interrogator
from torch import nn
from clip_extract_keyword import get_dict, make_keyword
from glob import glob
from datetime import datetime
from transformers import BertTokenizer, BertModel
import torch
import os

class InterrogatorWrapper(nn.Module):
    def __init__(self, model_path, caption_model_name="blip-base", device="cpu"):
        super().__init__()
        self.clip_interrogator = Interrogator(
        Config(caption_model_name=caption_model_name, 
               clip_model_name="ViT-L-14/openai", 
               cache_path=model_path, 
               clip_model_path=model_path,
               quiet=True,
               device=device)
        )
    
    def forward(self, img_pil, mode="fast"):
        if mode=="fast":
            result = self.clip_interrogator.interrogate_fast(img_pil)
        elif mode=="classic":
            result = self.clip_interrogator.interrogate_classic(img_pil)
        elif mode=="negative":
            result = self.clip_interrogator.interrogate_negative(img_pil)
        elif mode == 'simple':
            result = self.clip_interrogator.interrogate_simple(img_pil)   
        else:
            result = self.clip_interrogator.interrogate(img_pil)
        return result
    
    ### def 함수 생성 : 빈도분석 결과  


def load_interrogator(model_path, caption_model_name="blip-base", device="cuda"):
    clip_intrerrogator = InterrogatorWrapper(
        model_path=model_path,
        caption_model_name=caption_model_name, 
        device=device
    )
    return clip_intrerrogator
def bert_score(text):
    # Pre-trained BERT model 및 tokenizer 로드
    model_name='bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True)

    # 문장 토큰화 및 토큰 인코딩
    tokens = tokenizer.encode(text, return_tensors='pt')

    # 모델 통과 및 Self-Attention 가중치 추출
    with torch.no_grad():
        outputs = model(tokens)

    # BERT 출력에서 Self-Attention 가중치 가져오기
    attention_weights = outputs.attentions

    # 마지막 레이어의 Self-Attention 가중치 가져오기
    last_layer_attention_weights = attention_weights[-1]

    # 어텐션 헤드별 평균 계산 및 토큰 차원 제거
    average_attention_weights = last_layer_attention_weights.mean(dim=1).squeeze()

    # 각 토큰 및 평균 Self-Attention 가중치 출력
    count = 0
    token_rank = []
    rm_token_list = ['[SEP]','[EOS]', '[SOS]', '[CLS]']
    for token, attention_score in zip(tokenizer.convert_ids_to_tokens(tokens.squeeze()), average_attention_weights):
        #print(token, attention_score, average_attention_weights.mean(0)[count])
        if token not in rm_token_list:
            token_rank.append((token, average_attention_weights.mean(0)[count]))
            count += 1

    token_rank = sorted(token_rank, key=lambda x:x[1], reverse=True);
    max_rank = token_rank[0]
    return max_rank


def inference(imgs_path, model, mode="simple",del_color=True):
    if imgs_path[-3:] == 'jpg' or imgs_path[-3:] == 'png':
        fns = glob(imgs_path)
    else:
        d = imgs_path+'/*'
        d = d.replace('/',"\\")
        fns = glob(d)

    new_dict = get_dict(fns, model, img_root_path='', mode=mode)
    new_dict['del_color_keword'] = make_keyword(new_dict, del_color=del_color)
    result_path: str = os.path.join(os.path.dirname(__file__), 'result')
    c_time = datetime.now().strftime('%m%d%H%M')

    new_dict.to_csv(os.path.join(result_path, f'result_{c_time}.csv'), header=['image','text', 'pre_text','keword'], index=False)
        
    print(f'"result_{c_time}.csv" saved at result.')
    print("Return {Text : [Keyword, Attention score]} Dict.")
    
    keywords = list(new_dict['del_color_keword'])
    kwd_dict = {}
    for text in list(keywords):
        keyword = bert_score(text)
        kwd_dict[text] = [keyword[0], keyword[1]]

    return kwd_dict

if __name__ == "__main__":

    model = load_interrogator("C:\\Users\\mlfav\\lib\\minn\\CLIP\\Favorfit_image_to_text\\clip", caption_model_name="blip-base", device="cuda") 
    imgs_path ='D:/FAVORFIT/black_background/conditioning_images/citrusfruit_1847_clear.jpg'
    inference(imgs_path, model)
