### load model
  root_path = ''
  model_path = os.path.join(root_path, 'clip")
download_model on drive

### get keword dictionary and attention score of main keyword
  model = load_interrogator(model_path, caption_model_name="blip-base", device="cuda") 
  imgs_path ='your image path'
  keyword_score_dict = inference(imgs_path, model)
  
result csv file saved at ./result.
Return {Text : [Keyword, Attention score]} Dict.
