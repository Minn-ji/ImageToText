Extract Keyword from Image.

## Load model
``` Python
  root_path = ''
  model_path = os.path.join(root_path, 'clip')
```
> download safetensors on drive : 
  > https://drive.google.com/drive/folders/1Fkq6O4VeFK-6NhYQU9rAIbkgSQO0NLef?usp=drive_link


## Get keyword dictionary and attention score of keyword
``` Python
  model = load_interrogator(model_path, caption_model_name="blip-base", device="cuda") 
  imgs_path ='your image path'
  keyword_score_dict = inference(imgs_path, model)
```
- result(csv file) saved at <span style="background-color:blue">./result</span>.
- Inference.py returns <span style="background-color:blue">{Text : [Keyword, Attention score]}</span> dictionary.
