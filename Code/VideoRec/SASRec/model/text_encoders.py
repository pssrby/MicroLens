import torch
import torch.nn as nn

class TextEncoder(torch.nn.Module):
    def __init__(self,
                 bert_model,
                 item_embedding_dim,
                 word_embedding_dim,
                 args):
        super(TextEncoder, self).__init__()
        self.bert_model = bert_model
        self.activate = nn.ReLU()
        self.pooler = nn.Linear(word_embedding_dim, item_embedding_dim)
    
    def encode_sequence(self, text):
        batch_size, num_words = text.shape
        num_words = num_words // 2
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_attmask = torch.narrow(text, 1, num_words, num_words)

        valid = (text_attmask.sum(dim=1) > 0) & (text_ids.sum(dim=1) > 0)
        seq_out = self.pooler.weight.new_zeros((batch_size, num_words, self.pooler.out_features))
        mask_out = text_attmask.new_zeros((batch_size, num_words))
        if valid.any():
            ids_v = text_ids[valid]
            mask_v = text_attmask[valid]
            hidden_states = self.bert_model(input_ids=ids_v, attention_mask=mask_v)[0]
            token_v = self.pooler(hidden_states)
            seq_out[valid] = token_v.to(dtype=seq_out.dtype)
            mask_out[valid] = mask_v
        return seq_out, mask_out
        
    def forward(self, text):
        batch_size, num_words = text.shape
        num_words = num_words // 2
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_attmask = torch.narrow(text, 1, num_words, num_words)

        valid = (text_attmask.sum(dim=1) > 0) & (text_ids.sum(dim=1) > 0)
        out = self.pooler.weight.new_zeros((batch_size, self.pooler.out_features))
        if valid.any():
            ids_v = text_ids[valid]
            mask_v = text_attmask[valid]
            hidden_states = self.bert_model(input_ids=ids_v, attention_mask=mask_v)[0]
            cls_v = self.pooler(hidden_states[:, 0])
            out[valid] = cls_v.to(dtype=out.dtype)
        return out

class TextEmbedding(torch.nn.Module):
    def __init__(self, args, bert_model):
        super(TextEmbedding, self).__init__()
        self.args = args
        # we use the title of item with a fixed length.
        self.text_length = args.num_words_title * 2 # half for mask
        self.text_encoders = TextEncoder(bert_model, args.embedding_dim, args.word_embedding_dim, args)

    def forward(self, news):
        return self.text_encoders(torch.narrow(news, 1, 0, self.text_length))

    def forward_sequence(self, news):
        return self.text_encoders.encode_sequence(torch.narrow(news, 1, 0, self.text_length))

class TextFeatureEncoder(torch.nn.Module):
    def __init__(self, args):
        super(TextFeatureEncoder, self).__init__()
        self.args = args
        self.activate = nn.ReLU()
        self.pooler = nn.Linear(args.word_embedding_dim, args.embedding_dim)

    def forward(self, text_feature):
        return self.activate(self.pooler(text_feature))
