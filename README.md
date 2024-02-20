## load model
``` Python
  root_path = ''
  model_path = os.path.join(root_path, 'clip")
```
- download safetensors on drive : 
  > https://drive.google.com/drive/folders/1Fkq6O4VeFK-6NhYQU9rAIbkgSQO0NLef?usp=drive_link


### get keyword dictionary and attention score of main keyword
``` Python
  model = load_interrogator(model_path, caption_model_name="blip-base", device="cuda") 
  imgs_path ='your image path'
  keyword_score_dict = inference(imgs_path, model)
```
- result(csv file) saved at <./result>.
- Return {Text : [Keyword, Attention score]} dictionary.
