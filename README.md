{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import warnings\n",
    "warnings.filterwarnings(action='ignore')\n",
    "import pandas as pd\n",
    "from clip_image_to_text import load_interrogator, inference"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 5/5 [00:03<00:00,  1.47it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Complete making dictionary.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 5/5 [00:00<?, ?it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Complete preprocessing texts.\n",
      "\"result_02201340.csv\" saved at result.\n",
      "Return {Text : [Keyword, Attention score]} Dict.\n"
     ]
    }
   ],
   "source": [
    "model_path = \"C:\\\\Users\\\\mlfav\\\\lib\\\\minn\\\\ImageToText\\\\clip\"\n",
    "model = load_interrogator(model_path, caption_model_name=\"blip-base\", device=\"cuda\") \n",
    "imgs_path ='D:/FAVORFIT/black_background/conditioning_images_for_test'\n",
    "keyword_score_dict = inference(imgs_path, model)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'face cream box': ['face', tensor(0.0830)],\n",
       " 'pumpkin pine branch leaves': ['branch', tensor(0.0724)],\n",
       " 'tennis ball': ['tennis', tensor(0.1104)],\n",
       " 'coconut flower': ['flower', tensor(0.1036)],\n",
       " 'cake fruit': ['cake', tensor(0.1059)]}"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "keyword_score_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>image</th>\n",
       "      <th>text</th>\n",
       "      <th>pre_text</th>\n",
       "      <th>keword</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>cosmeticsbottle_540_atube.jpg</td>\n",
       "      <td>the face cream in a white box</td>\n",
       "      <td>face cream white box</td>\n",
       "      <td>face cream box</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>harvest_466_acoup.jpg</td>\n",
       "      <td>a pumpkin and a pine branch with leaves</td>\n",
       "      <td>pumpkin pine branch leaves</td>\n",
       "      <td>pumpkin pine branch leaves</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>tennis_247_atenn.jpg</td>\n",
       "      <td>a tennis ball on a black background</td>\n",
       "      <td>tennis ball black background</td>\n",
       "      <td>tennis ball</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>citrusfruit_1848_yello.jpg</td>\n",
       "      <td>a coconut with a flower on top</td>\n",
       "      <td>coconut flower</td>\n",
       "      <td>coconut flower</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>confectionery_1275_brown.jpg</td>\n",
       "      <td>a cake with fruit on top</td>\n",
       "      <td>cake fruit</td>\n",
       "      <td>cake fruit</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                           image                                     text   \n",
       "0  cosmeticsbottle_540_atube.jpg            the face cream in a white box  \\\n",
       "1          harvest_466_acoup.jpg  a pumpkin and a pine branch with leaves   \n",
       "2           tennis_247_atenn.jpg      a tennis ball on a black background   \n",
       "3     citrusfruit_1848_yello.jpg           a coconut with a flower on top   \n",
       "4   confectionery_1275_brown.jpg                 a cake with fruit on top   \n",
       "\n",
       "                       pre_text                      keword  \n",
       "0          face cream white box              face cream box  \n",
       "1    pumpkin pine branch leaves  pumpkin pine branch leaves  \n",
       "2  tennis ball black background                 tennis ball  \n",
       "3                coconut flower              coconut flower  \n",
       "4                    cake fruit                  cake fruit  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "result_dict = pd.read_csv('result/result_02201337.csv');result_dict"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### ."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "# commons, common_word = common_words(result_dict, col='pre_text',num_common_words=5)\n",
    "# result_dict['ngram_text'] = to_ngrams(result_dict,2)\n",
    "# commons, common_word = common_words(result_dict, col='ngram_text',num_common_words=30,split=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "# for i in range(len(result_dict)):\n",
    "#     if ('red', 'white')  in result_dict['ngram_text'][i]:\n",
    "#         print(result_dict['pre_text'][i])\n",
    "    "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "diffusion",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.15"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
