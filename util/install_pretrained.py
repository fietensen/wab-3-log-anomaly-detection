from transformers import BertModel, BertTokenizer

if __name__ == '__main__':
    print("[INFO] Installing pretrained BERT (Bidirectional encoder representations from transformers)")
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model.save_pretrained(".\\bert_base_uncased_hf_model")
    tokenizer.save_pretrained(".\\bert_base_uncased_hf_tokenizer")