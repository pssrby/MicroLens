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
        
    def forward(self, text):
        batch_size, num_words = text.shape
        num_words = num_words // 2
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_attmask = torch.narrow(text, 1, num_words, num_words)

        # Method 2: do NOT feed padding rows (item_id==0) into BERT.
        # Padding rows typically have all-zero input_ids/attention_mask after dataset padding.
        # We filter valid rows, run BERT only on them, then scatter back.
        valid = (text_attmask.sum(dim=1) > 0) & (text_ids.sum(dim=1) > 0)
        # Keep dtype consistent with BERT/pooler output (may be fp16 under AMP).
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
