import lmdb
import numpy as np
import os
import torch
import pickle
from torch.nn.init import xavier_normal_, constant_
import torch.nn as nn
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, BertConfig

class LMDB_VIDEO:
    def __init__(self, video):
        self.video = video.tobytes()
class SLOWFAST50Encoder(torch.nn.Module):
    def __init__(self, video_net):
        super(SLOWFAST50Encoder, self).__init__()
        self.video_net = video_net
        self.activate = nn.GELU()

    def forward(self, item_content):
        item_content = item_content.transpose(1,2)
        slow_item_content_1 = item_content[:, :, 0, :, :].unsqueeze(2)
        slow_item_content_2 = item_content[:, :, -1, :, :].unsqueeze(2)
        slow_item_content = torch.cat((slow_item_content_1, slow_item_content_2), 2)
        item_scoring = self.video_net([slow_item_content, item_content])
        return item_scoring
class TextEncoder(torch.nn.Module):
    def __init__(self,
                 bert_model,
                 ):
        super(TextEncoder, self).__init__()
        self.bert_model = bert_model
        
    def forward(self, text):
        batch_size, num_words = text.shape
        num_words = num_words // 2
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_attmask = torch.narrow(text, 1, num_words, num_words)
        hidden_states = self.bert_model(input_ids=text_ids, attention_mask=text_attmask)[0]
        return hidden_states[:, 0]

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
    def __init__(self, args_num_words_title, bert_model):
        super(TextEmbedding, self).__init__()
        # we use the title of item with a fixed length.
        self.text_length = args_num_words_title * 2 # half for mask
        self.text_encoders = TextEncoder(bert_model)

    def forward(self, news):
        return self.text_encoders(torch.narrow(news, 1, 0, self.text_length))
def load_submodule_state(module, checkpoint_path, prefix):
    print(f'Loading submodule `{prefix}` from checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    print(f'checkpoint keys: {list(checkpoint.keys())}')
    state_dict = checkpoint['model_state_dict']
    state_dict_keys = list(state_dict.keys())
    preview_keys = state_dict_keys[:20]
    print(f'model_state_dict param count: {len(state_dict_keys)}')
    print(f'model_state_dict preview keys: {preview_keys}')
    # os._exit(0)

    submodule_state = {}
    prefix_with_dot = prefix + '.'
    for key, value in state_dict.items():
        if key.startswith(prefix_with_dot):
            new_key = key[len(prefix_with_dot):]
            submodule_state[new_key] = value

    if len(submodule_state) == 0:
        raise ValueError(f"No parameters found for prefix `{prefix}` in {checkpoint_path}")

    missing_keys, unexpected_keys = module.load_state_dict(submodule_state, strict=False)
    print(f'load `{prefix}` from {checkpoint_path}')
    print(f'missing_keys: {missing_keys}')
    print(f'unexpected_keys: {unexpected_keys}')
def extract_video_feature(video_lmdb_path, video_feature_output_path):
    video_lmdb_path = os.path.expanduser(video_lmdb_path)
    video_feature_output_path = os.path.expanduser(video_feature_output_path)
    args_video_ckpt_path = '/data2/guangyi/MicroLens/Code/VideoRec/SASRec/checkpoint/checkpoint_MicroLens-100k_pairs_video/cpt_v1_sasrec_blocknum_2_tau_0.07_bs_20_fi_1_fn_5_ed_256_lr_5e-05_l2_0.1_flrVideo_0.0005_slowfast-50_freeze_270_maxLen_10/epoch-45.pt'
    video_model = torch.hub.load(
        './pytorchvideo_rs', 
        model='slowfast_r50', 
        source='local', 
        head_pool_kernel_sizes=((1, 7, 7), (4, 7, 7)), 
        pretrained=True
        )
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    video_encoder = SLOWFAST50Encoder(video_model).to(device)
    load_submodule_state(video_encoder, args_video_ckpt_path, "video_encoder")

    video_encoder.eval()
    env = lmdb.open(
        video_lmdb_path, 
        subdir=os.path.isdir(video_lmdb_path), 
        readonly=True, 
        lock=False, 
        readahead=False, 
        meminit=False
        )
    with env.begin() as txn:
        entry_size = txn.stat()['entries']
    output_env = lmdb.open(
        video_feature_output_path, 
        subdir=os.path.isdir(video_feature_output_path), 
        map_size=entry_size * np.dtype(np.float32).itemsize * 400 * 10,  # Adjust map_size based on your needs
        readonly=False, 
        lock=False, 
        readahead=False, 
        meminit=False
        )
    output_txn = output_env.begin(write=True)
    write_frequency = 100
    with env.begin() as input_txn:
        video_keys = pickle.loads(input_txn.get(b'__keys__'))
        for idx, key in enumerate(tqdm(video_keys)):
            # pos
            VIDEO = pickle.loads(input_txn.get(key))
            VIDEO = np.copy(np.frombuffer(VIDEO.video, dtype=np.float32)).reshape(5, 3, 224, 224)
            VIDEO = torch.from_numpy(VIDEO).unsqueeze(0).float().to(device)
            with torch.no_grad():
                video_feature = video_encoder(VIDEO)
                print(video_feature.shape)
                print(type(video_feature))
                os._exit(0)
                video_feature = video_feature[0]
            video_feature = video_feature.cpu().detach().numpy()
            # print(video_feature.shape)
            # print(type(video_feature[0]))
            # os._exit(0)
            output_txn.put(key, pickle.dumps(video_feature))
            if idx % write_frequency == 0 and idx != 0:
                output_txn.commit()
                output_txn = output_env.begin(write=True)
    output_txn.commit()
    with output_env.begin(write=True) as output_txn:
        output_txn.put(b'__keys__', pickle.dumps(video_keys))
        output_txn.put(b'__len__', pickle.dumps(len(video_keys)))
    print(len(video_keys))
    print("Flushing database ...")
    output_env.sync()
    output_env.close()

def read_texts(tokenizer, args_num_words_title):
    args_root_data_dir = '~/dataset'
    args_dataset = 'MicroLens-100k-Dataset'
    args_text_data = 'MicroLens-100k_title_en.csv'
    text_path = os.path.expanduser(os.path.join(args_root_data_dir, args_dataset, args_text_data))
    item_dic = {}
    item_name_to_index = {}
    item_index_to_name = {}
    index = 1

    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            splited = line.strip('\n').split(',')
            doc_name, title = splited[0], str(','.join(splited[1:]))
            # if 'scale' in args.dataset:
            #     splited = line.strip('\n').split(',')
            #     doc_name, title = splited[0], str(','.join(splited[1:]))
            # elif 'MIND' in args.dataset:
            #     splited = line.strip('\n').split('\t')
            #     doc_name, title, _ = splited
            # else:
            #     splited = line.strip('\n').split('\t')
            #     doc_name, title = splited
            item_name_to_index[doc_name] = index
            item_index_to_name[index] = doc_name
            index += 1
            # tokenizer
            tokenized_title = tokenizer(title.lower(), max_length=args_num_words_title, padding='max_length', truncation=True)
            item_dic[doc_name] = [tokenized_title]

    return item_dic, item_name_to_index, item_index_to_name

def get_doc_input_bert(text_dic, item_index, args_num_words_title):
    item_num = len(text_dic) + 1

    news_title = np.zeros((item_num, args_num_words_title), dtype='int32')
    news_title_attmask = np.zeros((item_num, args_num_words_title), dtype='int32')

    for key in text_dic:
        title = text_dic[key]
        doc_index = item_index[key]
        
        news_title[doc_index] = title[0]['input_ids']
        news_title_attmask[doc_index] = title[0]['attention_mask']

    return news_title, news_title_attmask

def extract_text_feature(text_file, text_feature_output_path):
    # Implement text feature extraction using BERT or any other text encoder
    args_root_model_dir = '~/model'
    args_text_model_load = 'bert-small'
    args_text_ckpt_path = '/data2/guangyi/MicroLens/Code/VideoRec/SASRec/checkpoint/checkpoint_MicroLens-100k_pairs_text/cpt_v3_sasrec_blocknum_2_tau_0.07_bs_512_ed_512_lr_0.0001_l2_0.1_flrText_0.0001_bert-small_freeze_0_maxLen_10/epoch-40.pt'
    text_model_load = os.path.expanduser(os.path.join(args_root_model_dir, 'pretrained_models/bert', args_text_model_load))
    text_feature_output_path = os.path.expanduser(text_feature_output_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(text_model_load)
    config = BertConfig.from_pretrained(text_model_load, output_hidden_states=True)
    text_model = BertModel.from_pretrained(text_model_load, config=config).to(device)
    load_submodule_state(text_model, args_text_ckpt_path, prefix="text_encoder.text_encoders.bert_model")
    if 'small' in args_text_model_load:
        pooler_para = [69, 70]
        args_word_embedding_dim = 512

    item_dic_titles_after_tokenizer, before_item_name_to_index, before_item_index_to_name = read_texts(tokenizer, args_num_words_title=30)
    text_title, text_title_attmask = get_doc_input_bert(item_dic_titles_after_tokenizer, before_item_name_to_index, args_num_words_title=30)
    item_content = np.concatenate([text_title, text_title_attmask], axis=1)
    max_doc_id = max(int(doc_name) for doc_name in before_item_name_to_index.keys())
    features = torch.zeros((max_doc_id + 1, args_word_embedding_dim), dtype=torch.float32)
    all_text = torch.from_numpy(item_content).long()
    text_encoder = TextEmbedding(args_num_words_title=30, bert_model=text_model).to(device)
    text_encoder.eval()

    batch_size = 256
    ordered_doc_names = [before_item_index_to_name[i] for i in range(1, len(before_item_index_to_name) + 1)]
    all_text_real = all_text[1:]
    with torch.no_grad():
        for start in tqdm(range(0, all_text_real.size(0), batch_size), desc='extract_text_feature'):
            end = start + batch_size
            batch_text = all_text_real[start:end].to(device)
            batch_feature = text_encoder(batch_text)
            batch_doc_names = ordered_doc_names[start:end]
            batch_feature = batch_feature.cpu()
            for row_idx, doc_name in enumerate(batch_doc_names):
                features[int(doc_name)] = batch_feature[row_idx]

    features[0].zero_()
    print(f'features has nan: {torch.isnan(features).any().item()}')
    torch.save({'features': features}, text_feature_output_path)
    print(f'saved text features to {text_feature_output_path}')
    print(features.shape)
    

if __name__ == "__main__":
    # video_lmdb_path = '~/dataset/MicroLens-100k-Dataset/MicroLens-100k_frames_interval_1_number_5.lmdb'
    # video_feature_output_path = '~/dataset/MicroLens-100k-Dataset/MicroLens-100k_frames_interval_1_number_5_encoder1.lmdb'
    extract_video_feature(
        video_lmdb_path='~/dataset/MicroLens-100k-Dataset/MicroLens-100k_frames_interval_1_number_5.lmdb', 
        video_feature_output_path='~/dataset/MicroLens-100k-Dataset/MicroLens-100k_frames_interval_1_number_5_encoder5.lmdb'
        )

    # extract_text_feature(text_file='~/dataset/MicroLens-100k-Dataset/MicroLens-100k_title_en.csv', text_feature_output_path='~/dataset/MicroLens-100k-Dataset/MicroLens-100k_title_en_encoder2.pt')