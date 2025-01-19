import torch
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings


class DBVectirizer(Embeddings):
    """
    this class is used to make embeddings for vector database
    https://huggingface.co/ai-forever/sbert_large_nlu_ru
    """
    def __init__(self, model_name = "ai-forever/sbert_large_nlu_ru", device_map = "cuda:1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map = device_map)
        self.model = AutoModel.from_pretrained(model_name, device_map = device_map)
        self.device = self.model.device
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
        
    def embed_documents(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, max_length=24, return_tensors='pt')
        encoded_input = encoded_input.to(self.device)
        #Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        #Perform pooling. In this case, mean pooling
        sentence_embeddings = DBVectirizer.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings.to("cpu").numpy()
    
    def embed_query(self, text):
        return self(text)
    def __call__(self, text):
        res = self.embed_documents([text])
        return res[0]
            
