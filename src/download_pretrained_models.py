from transformers import AutoModel, AutoTokenizer

AutoModel.from_pretrained('DeepPavlov/rubert-base-cased-sentence')
AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')
