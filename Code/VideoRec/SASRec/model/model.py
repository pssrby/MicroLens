import os
import torch
import numpy as np
import lmdb
import pickle
from torch import nn
from torch.nn.init import xavier_normal_
from collections import Counter
import torch.nn.functional as F
from .Attention_Fusion import CoAttention, MergedAttention

from .text_encoders import TextEmbedding
from .text_encoders import TextEmbedding, TextFeatureEncoder
from .video_encoders import VideoMaeEncoder, R3D18Encoder, R3D50Encoder, C2D50Encoder
from .video_encoders import I3D50Encoder, CSN101Encoder, SLOW50Encoder, EX3DSEncoder
from .video_encoders import EX3DXSEncoder, X3DXSEncoder, X3DSEncoder, X3DMEncoder
from .video_encoders import X3DLEncoder, MVIT16Encoder, MVIT16X4Encoder, MVIT32X3Encoder
from .video_encoders import SLOWFAST50Encoder, SLOWFAST16X8101Encoder
from .video_encoders import VideoFeatureEncoder
from .image_encoders import VitEncoder, ResnetEncoder, MaeEncoder, SwinEncoder 
from .fusion_module import MoEFusion, SumFusion, ConcatFusion, FiLM, GatedFusion, CoAttnFusion, CoAttentionSingle, CrossAttentionSingle, CrossAttentionSeq, CoAttentionSeq 
from .user_encoders import User_Encoder_GRU4Rec, User_Encoder_SASRec, User_Encoder_NextItNet

class Model(torch.nn.Module):
    def __init__(self, args, pop_prob_list, item_num, bert_model, image_net, video_net, tokenizer=None, item_id_to_keys=None):
        super(Model, self).__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.item_num = item_num
        self.pop_prob_list = torch.FloatTensor(pop_prob_list)
        self.item_id_to_keys = item_id_to_keys
        self.tokenizer = tokenizer
        self._image_env = None
        self._video_env = None

        if args.model == 'sasrec':
            self.user_encoder = User_Encoder_SASRec(args)
        elif args.model == 'gru4rec':
            self.user_encoder = User_Encoder_GRU4Rec(args)
        elif args.model == 'nextitnet':
            self.user_encoder = User_Encoder_NextItNet(args)

        if args.item_tower in ('image', 'modal', 'text_image'):
            if 'resnet' in args.image_model_load:
                self.image_encoder = ResnetEncoder(image_net=image_net, args=args)
            elif 'vit-b-32-clip' in args.image_model_load:
                self.image_encoder = VitEncoder(image_net=image_net, args=args)
            elif 'vit-base-mae' in args.image_model_load:
                self.image_encoder = MaeEncoder(image_net=image_net, args=args)
            elif 'swin_tiny' in args.image_model_load or 'swin_base' in args.image_model_load:
                self.image_encoder = SwinEncoder(image_net=image_net, args=args)
            elif 'clip-vit-base-patch32' in args.image_model_load:
                self.image_encoder = VitEncoder(image_net=image_net, args=args)

        if args.item_tower in ('text', 'modal', 'text_image', 'text_video'):
            if args.text_feature_path in [None, 'None', 'none', '']:
                self.text_path = os.path.join(args.root_data_dir, args.dataset, args.text_data)
                self.text_feature_path = None
                self.text_content = None
                self.text_encoder = TextEmbedding(args=args, bert_model=bert_model)
            else:
                self.text_feature_path = os.path.expanduser(args.text_feature_path)
                self.text_path = None
                self.text_content = None
                self.text_encoder = TextFeatureEncoder(args=args)
        
        if args.item_tower in ('video', 'modal', 'text_video') and args.video_feature_path in [None, 'None', 'none', '']:
            if 'mae' in args.video_model_load:
                self.video_encoder = VideoMaeEncoder(video_net=video_net, args=args)
            elif 'r3d18' in args.video_model_load:
                self.video_encoder = R3D18Encoder(video_net=video_net, args=args)
            elif 'r3d50' in args.video_model_load:
                self.video_encoder = R3D50Encoder(video_net=video_net, args=args)
            elif 'c2d50' in args.video_model_load:
                self.video_encoder = C2D50Encoder(video_net=video_net, args=args)
            elif 'i3d50' in args.video_model_load:
                self.video_encoder = I3D50Encoder(video_net=video_net, args=args)
            elif 'csn101' in args.video_model_load:
                self.video_encoder = CSN101Encoder(video_net=video_net, args=args)
            elif 'slow50' in args.video_model_load:
                self.video_encoder = SLOW50Encoder(video_net=video_net, args=args)
            elif 'efficient-x3d-s' in args.video_model_load:
                self.video_encoder = EX3DSEncoder(video_net=video_net, args=args)
            elif 'efficient-x3d-xs' in args.video_model_load:
                self.video_encoder = EX3DXSEncoder(video_net=video_net, args=args)
            elif 'x3d-xs' in args.video_model_load:
                self.video_encoder = X3DXSEncoder(video_net=video_net, args=args)
            elif 'x3d-s' in args.video_model_load:
                self.video_encoder = X3DSEncoder(video_net=video_net, args=args)
            elif 'x3d-m' in args.video_model_load:
                self.video_encoder = X3DMEncoder(video_net=video_net, args=args)
            elif 'x3d-l' in args.video_model_load:
                self.video_encoder = X3DLEncoder(video_net=video_net, args=args)
            elif 'mvit-base-16' in args.video_model_load:
                self.video_encoder = MVIT16Encoder(video_net=video_net, args=args)
            elif 'mvit-base-16x4' in args.video_model_load:
                self.video_encoder = MVIT16X4Encoder(video_net=video_net, args=args)
            elif 'mvit-base-32x3' in args.video_model_load:
                self.video_encoder = MVIT32X3Encoder(video_net=video_net, args=args)
            elif 'slowfast-50' in args.video_model_load:
                self.video_encoder = SLOWFAST50Encoder(video_net=video_net, args=args)
            elif 'slowfast16x8-101' in args.video_model_load:
                self.video_encoder = SLOWFAST16X8101Encoder(video_net=video_net, args=args)
        if args.item_tower in ('video', 'modal', 'text_video') and args.video_feature_path not in [None, 'None', 'none', '']:
            if 'slowfast-50' in args.video_model_load:
                self.video_encoder = VideoFeatureEncoder(args=args)

        self.id_encoder = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        xavier_normal_(self.id_encoder.weight.data)

        self.criterion = nn.CrossEntropyLoss()
        self.contrastive_criterion = nn.CrossEntropyLoss()
        if args.use_text_video_seq_contrastive or getattr(args, 'text_video_seq_lambda', 0) > 0:
            self.text_seq_proj = nn.Linear(args.embedding_dim, args.embedding_dim)
            self.video_seq_proj = nn.Linear(args.embedding_dim, args.embedding_dim)
            xavier_normal_(self.text_seq_proj.weight.data)
            xavier_normal_(self.video_seq_proj.weight.data)
            if self.text_seq_proj.bias is not None:
                nn.init.zeros_(self.text_seq_proj.bias)
            if self.video_seq_proj.bias is not None:
                nn.init.zeros_(self.video_seq_proj.bias)

        fusion = args.fusion_method.lower()
        if fusion == 'concat' and args.item_tower in ('modal', 'text_image', 'text_video', 'text'):
            self.fusion_module = ConcatFusion(args=args)
        elif fusion == 'moe' and args.item_tower in ('modal', 'text_image', 'text_video', 'text'):
            num_modalities = 3 if args.item_tower == 'modal' else 2
            self.fusion_module = MoEFusion(args=args, num_modalities=num_modalities)
        elif fusion == 'sum' and args.item_tower in ('modal', 'text_image', 'text_video', 'text'):
            self.fusion_module = SumFusion(args=args)
        elif fusion == 'film' and args.item_tower in ('modal', 'text_image', 'text_video', 'text'):
            self.fusion_module = FiLM(args=args, x_film=False)
        elif fusion == 'gated' and args.item_tower in ('modal', 'text_image', 'text_video', 'text'):
            self.fusion_module = GatedFusion(args=args)
        elif fusion == 'co_att' :
            self.fusion_module = CoAttention.from_pretrained("~/model/pretrained_models/bert/bert-base-uncased", args=args)
        elif fusion == 'merge_attn':
            self.fusion_module = MergedAttention.from_pretrained("~/model/pretrained_models/bert/bert-base-uncased",args=args)
        elif fusion == 'coattnfusion':
            self.fusion_module = CoAttnFusion(args=args)
        elif fusion == 'coattentionsingle':
            self.fusion_module = CoAttentionSingle(args=args)
        elif fusion == 'crossattentionsingle':
            self.fusion_module = CrossAttentionSingle(args=args)
        elif fusion == 'crossattentionseq':
            self.fusion_module = CrossAttentionSeq(args=args)
        elif fusion == 'coattentionseq':
            self.fusion_module = CoAttentionSeq(args=args)
        else:
            # fallback to concat for modal as default, or no fusion for single modality.
            if args.item_tower in ('modal', 'text_image', 'text_video', 'text'):
                self.fusion_module = ConcatFusion(args=args)

    def _load_text_tokens_from_ids(self, args, tokenizer, item_id_to_keys, item_num):
        if tokenizer is None:
            raise ValueError('tokenizer is required for online text towers.')
        if item_id_to_keys is None:
            raise ValueError('item_id_to_keys is required to load text from item ids.')

        text_content = np.zeros((item_num + 1, args.num_words_title * 2), dtype=np.int64)
        text_path = os.path.join(args.root_data_dir, args.dataset, args.text_data)
        key_to_id = {str(raw_key): item_id for item_id, raw_key in item_id_to_keys.items()}

        with open(text_path, 'r', encoding='utf-8') as f:
            for line in f:
                splited = line.strip('\n').split(',')
                if len(splited) < 2:
                    continue
                doc_name, title = splited[0], str(','.join(splited[1:]))
                if doc_name not in key_to_id:
                    continue
                item_id = key_to_id[doc_name]
                tokenized_title = tokenizer(title.lower(), max_length=args.num_words_title, padding='max_length', truncation=True)
                text_content[item_id, :args.num_words_title] = tokenized_title['input_ids']
                text_content[item_id, args.num_words_title:] = tokenized_title['attention_mask']

        return torch.LongTensor(text_content)

    def _load_text_features_from_ids(self, args, item_id_to_keys, item_num):
        if item_id_to_keys is None:
            raise ValueError('item_id_to_keys is required to load text features from item ids.')

        text_feature_data = torch.load(args.text_feature_path, map_location='cpu')
        if isinstance(text_feature_data, dict) and 'features' in text_feature_data:
            item_content = text_feature_data['features']
        else:
            item_content = text_feature_data
        if isinstance(item_content, torch.Tensor):
            item_content = item_content.cpu().numpy()

        args.word_embedding_dim = int(item_content.shape[1])
        reordered_item_content = np.zeros((item_num + 1, item_content.shape[1]), dtype=item_content.dtype)
        for item_id, raw_doc_name in item_id_to_keys.items():
            raw_index = int(raw_doc_name)
            if 0 <= raw_index < item_content.shape[0]:
                reordered_item_content[item_id] = item_content[raw_index]
        if np.isnan(reordered_item_content).any():
            raise ValueError('loaded text features contain nan values.')
        return torch.FloatTensor(reordered_item_content)

    def _ensure_text_content_loaded(self):
        if self.text_content is not None:
            return

        if self.item_id_to_keys is None:
            raise ValueError('item_id_to_keys is required to load text from item ids.')

        if self.text_feature_path in [None, 'None', 'none', '']:
            if self.tokenizer is None:
                raise ValueError('tokenizer is required for online text towers.')
            self.text_content = self._load_text_tokens_from_ids(self.args, self.tokenizer, self.item_id_to_keys, self.item_num)
        else:
            args = self.args
            args.text_feature_path = self.text_feature_path
            self.text_content = self._load_text_features_from_ids(args, self.item_id_to_keys, self.item_num)

    def get_text_inputs_from_ids(self, item_ids, local_rank):
        self._ensure_text_content_loaded()
        return self.text_content.to(local_rank)[item_ids]

    def _open_lmdb(self, db_path, cache_name):
        env = getattr(self, cache_name)
        if env is None:
            env = lmdb.open(db_path, subdir=os.path.isdir(db_path), readonly=True, lock=False, readahead=False, meminit=False)
            setattr(self, cache_name, env)
        return env

    def _load_lmdb_items(self, item_ids, db_path, cache_name, mode, local_rank):
        if self.item_id_to_keys is None:
            raise ValueError('item_id_to_keys is required to load modality data from item ids.')

        item_ids_cpu = item_ids.detach().cpu().view(-1).tolist()
        env = self._open_lmdb(db_path, cache_name)

        if mode == 'image':
            shape = (len(item_ids_cpu), 3, self.args.image_resize, self.args.image_resize)
        elif mode == 'video':
            shape = (len(item_ids_cpu), self.args.frame_no, 3, 224, 224)
        elif mode == 'video_feature':
            shape = (len(item_ids_cpu), 400)
        else:
            raise ValueError(f'unsupported lmdb mode: {mode}')

        output = np.zeros(shape, dtype=np.float32)
        with env.begin() as txn:
            for idx, item_id in enumerate(item_ids_cpu):
                if item_id == 0:
                    continue
                byteflow = txn.get(self.item_id_to_keys[item_id].encode())
                if byteflow is None:
                    continue
                data = pickle.loads(byteflow)
                if mode == 'image':
                    output[idx] = np.copy(np.frombuffer(data.image, dtype=np.float32)).reshape(3, 224, 224)
                elif mode == 'video':
                    output[idx] = np.copy(np.frombuffer(data.video, dtype=np.float32)).reshape(self.args.frame_no, 3, 224, 224)
                else:
                    output[idx] = data
        return torch.FloatTensor(output).to(local_rank)

    def build_modality_inputs_from_ids(self, sample_items_id, local_rank):
        sample_items_text = None
        sample_items_image = None
        sample_items_video = None

        if self.args.item_tower in ('modal', 'text', 'text_image', 'text_video'):
            sample_items_text = self.get_text_inputs_from_ids(sample_items_id, local_rank)

        if self.args.item_tower in ('modal', 'image', 'text_image'):
            image_db_path = os.path.join(self.args.root_data_dir, self.args.dataset, self.args.image_data)
            sample_items_image = self._load_lmdb_items(sample_items_id, image_db_path, '_image_env', 'image', local_rank)

        if self.args.item_tower in ('modal', 'video', 'text_video'):
            if self.args.video_feature_path in [None, 'None', 'none', '']:
                video_db_path = os.path.join(self.args.root_data_dir, self.args.dataset, self.args.video_data)
                sample_items_video = self._load_lmdb_items(sample_items_id, video_db_path, '_video_env', 'video', local_rank)
            else:
                video_db_path = os.path.expanduser(self.args.video_feature_path)
                sample_items_video = self._load_lmdb_items(sample_items_id, video_db_path, '_video_env', 'video_feature', local_rank)

        return sample_items_text, sample_items_image, sample_items_video

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def text_video_contrastive_loss(self, text_embs, video_embs):
        if text_embs is None or video_embs is None:
            return None
        if text_embs.size(0) == 0 or video_embs.size(0) == 0:
            return None

        text_embs = F.normalize(text_embs, dim=-1)
        video_embs = F.normalize(video_embs, dim=-1)
        logits = torch.matmul(text_embs, video_embs.t()) / self.args.text_video_tau
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_t2v = self.contrastive_criterion(logits, labels)
        loss_v2t = self.contrastive_criterion(logits.t(), labels)
        return 0.5 * (loss_t2v + loss_v2t)

    def text_video_item_contrastive_loss(self, item_ids, text_embs, video_embs):
        if item_ids is None or text_embs is None or video_embs is None:
            return None
        if item_ids.numel() == 0:
            return None
        if item_ids.dim() != 1:
            item_ids = item_ids.view(-1)

        valid = item_ids.ne(0)
        if valid.sum().item() < 2:
            return text_embs.new_zeros(())

        item_ids = item_ids[valid]
        text_embs = text_embs[valid]
        video_embs = video_embs[valid]

        # Aggregate multiple occurrences of the same item within the batch
        unique_ids, inverse = torch.unique(item_ids, sorted=True, return_inverse=True)
        n_unique = unique_ids.size(0)
        if n_unique < 2:
            return text_embs.new_zeros(())

        text_sum = text_embs.new_zeros((n_unique, text_embs.size(-1)))
        video_sum = video_embs.new_zeros((n_unique, video_embs.size(-1)))
        text_sum.index_add_(0, inverse, text_embs)
        video_sum.index_add_(0, inverse, video_embs)
        counts = torch.bincount(inverse, minlength=n_unique).to(text_embs.dtype).unsqueeze(1).clamp(min=1)
        text_item = text_sum / counts
        video_item = video_sum / counts

        # Use item-specific temperature
        text_item = F.normalize(text_item, dim=-1)
        video_item = F.normalize(video_item, dim=-1)
        logits = torch.matmul(text_item, video_item.t()) / self.args.text_video_item_tau
        labels = torch.arange(n_unique, device=logits.device)
        loss_t2v = self.contrastive_criterion(logits, labels)
        loss_v2t = self.contrastive_criterion(logits.t(), labels)
        return 0.5 * (loss_t2v + loss_v2t)

    def text_video_seq_contrastive_loss(self, text_user_vecs, video_user_vecs):
        if text_user_vecs is None or video_user_vecs is None:
            return None
        if text_user_vecs.size(0) < 2 or video_user_vecs.size(0) < 2:
            return text_user_vecs.new_zeros(())

        text_user_vecs = F.normalize(text_user_vecs, dim=-1)
        video_user_vecs = F.normalize(video_user_vecs, dim=-1)
        logits = torch.matmul(text_user_vecs, video_user_vecs.t()) / self.args.text_video_seq_tau
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_t2v = self.contrastive_criterion(logits, labels)
        loss_v2t = self.contrastive_criterion(logits.t(), labels)
        return 0.5 * (loss_t2v + loss_v2t)

    def pool_sequence_embeddings(self, seq_embs, log_mask):
        if seq_embs.dim() != 3:
            raise ValueError(f"seq_embs must be [B, L, D], got {seq_embs.shape}")

        seq_mask = log_mask.unsqueeze(-1).to(seq_embs.dtype)
        seq_lengths = seq_mask.sum(dim=1).clamp(min=1.0)
        return (seq_embs * seq_mask).sum(dim=1) / seq_lengths

    def encode_item_sequence(self, sample_items_id, sample_items_text, sample_items_image, sample_items_video, log_mask, local_rank, args):
        self.pop_prob_list = self.pop_prob_list.to(local_rank)
        if sample_items_text is None and sample_items_image is None and sample_items_video is None and args.item_tower != 'id':
            sample_items_text, sample_items_image, sample_items_video = self.build_modality_inputs_from_ids(sample_items_id, local_rank)
        debias_logits = torch.log(self.pop_prob_list[sample_items_id.view(-1)])

        sequence_fused_embs = None
        text_video_loss = None
        text_video_item_loss = None
        text_video_seq_loss = None

        if 'modal' == args.item_tower:
            if args.text_feature_path in [None, 'None', 'none', '']:
                input_all_text = self.text_encoder(sample_items_text.long())
            else:
                input_all_text = self.text_encoder(sample_items_text)
            input_all_image = self.image_encoder(sample_items_image)
            input_all_video = self.video_encoder(sample_items_video)
            score_embs = self.fusion_module(input_all_text, input_all_image, input_all_video)
        elif 'text_image' == args.item_tower:
            if args.text_feature_path in [None, 'None', 'none', '']:
                input_all_text = self.text_encoder(sample_items_text.long())
            else:
                input_all_text = self.text_encoder(sample_items_text)
            input_all_image = self.image_encoder(sample_items_image)
            if args.fusion_method.lower() in ('co_att', 'merge_attn', 'coattnfusion'):
                text_feats = input_all_text.unsqueeze(1) if input_all_text.dim() == 2 else input_all_text
                image_feats = input_all_image.unsqueeze(1) if input_all_image.dim() == 2 else input_all_image
                item_mask = (sample_items_id.view(-1) != 0).long().unsqueeze(-1)
                score_embs = self.fusion_module(text_feats, item_mask, image_feats, item_mask)
            elif args.fusion_method.lower() in ('coattentionsingle', 'crossattentionsingle'):
                score_embs = self.fusion_module(input_all_text, input_all_image)
            else:
                score_embs = self.fusion_module(input_all_text, input_all_image)
        elif 'text_video' == args.item_tower:
            text_seq = None
            text_seq_mask = None
            input_all_text = None
            if args.text_feature_path in [None, 'None', 'none', '']:
                if args.fusion_method.lower() in ('crossattentionseq', 'coattentionseq'):
                    input_all_text, text_seq_mask = self.text_encoder.forward_sequence(sample_items_text.long())
                    text_seq = input_all_text
                    text_lengths = text_seq_mask.sum(dim=1, keepdim=True).clamp(min=1)
                    text_pooled_for_cl = (text_seq * text_seq_mask.unsqueeze(-1)).sum(dim=1) / text_lengths
                else:
                    input_all_text = self.text_encoder(sample_items_text.long())
                    text_pooled_for_cl = input_all_text
            else:
                input_all_text = self.text_encoder(sample_items_text)
                text_pooled_for_cl = input_all_text
            input_all_video = self.video_encoder(sample_items_video)
            video_pooled_for_cl = input_all_video

            batch_size = log_mask.size(0)
            text_item_seq_embs = text_pooled_for_cl.view(batch_size, self.max_seq_len + 1, self.args.embedding_dim)
            video_item_seq_embs = video_pooled_for_cl.view(batch_size, self.max_seq_len + 1, self.args.embedding_dim)
            if args.fusion_method.lower() in ('co_att', 'merge_attn'):
                text_feats = input_all_text.unsqueeze(1) if input_all_text.dim() == 2 else input_all_text
                video_feats = input_all_video.unsqueeze(1) if input_all_video.dim() == 2 else input_all_video
                item_mask = (sample_items_id.view(-1) != 0).long().unsqueeze(-1)
                score_embs = self.fusion_module(text_feats, item_mask, video_feats, item_mask)
            elif args.fusion_method.lower() == 'coattnfusion':
                text_seq = input_all_text.view(batch_size, self.max_seq_len + 1, self.args.embedding_dim)
                video_seq = input_all_video.view(batch_size, self.max_seq_len + 1, self.args.embedding_dim)
                item_mask = sample_items_id.view(batch_size, self.max_seq_len + 1).ne(0).long()
                sequence_fused_embs = self.fusion_module(text_seq, item_mask, video_seq, item_mask)
                score_embs = sequence_fused_embs.view(-1, self.args.embedding_dim)
            elif args.fusion_method.lower() in ('crossattentionseq', 'coattentionseq'):
                if text_seq is None or text_seq_mask is None:
                    raise ValueError('sequence attention fusion requires online text encoder features, not precomputed text features.')
                score_embs = self.fusion_module(text_seq, text_seq_mask, input_all_video)
            elif args.fusion_method.lower() in ('coattentionsingle', 'crossattentionsingle'):
                score_embs = self.fusion_module(input_all_text, input_all_video)
            else:
                score_embs = self.fusion_module(input_all_text, input_all_video)

            if args.use_text_video_contrastive and args.text_video_lambda > 0:
                valid_mask = sample_items_id.view(-1).ne(0)
                text_video_loss = self.text_video_contrastive_loss(
                    text_pooled_for_cl[valid_mask],
                    video_pooled_for_cl[valid_mask]
                )

            if args.use_text_video_item_contrastive and args.text_video_item_lambda > 0:
                text_video_item_loss = self.text_video_item_contrastive_loss(
                    sample_items_id.view(-1),
                    text_pooled_for_cl,
                    video_pooled_for_cl,
                )

            if args.use_text_video_seq_contrastive and args.text_video_seq_lambda > 0:
                text_user_vecs = self.text_seq_proj(self.pool_sequence_embeddings(text_item_seq_embs[:, :-1, :], log_mask))
                video_user_vecs = self.video_seq_proj(self.pool_sequence_embeddings(video_item_seq_embs[:, :-1, :], log_mask))
                text_video_seq_loss = self.text_video_seq_contrastive_loss(text_user_vecs, video_user_vecs)
        elif 'text' == args.item_tower:
            ## TO MAKE text+id
            # input_all_text = self.text_encoder(sample_items_text.long())
            # input_all_id = self.id_encoder(sample_items_id)
            # input_all_text = F.normalize(input_all_text, dim=-1)
            # input_all_id = F.normalize(input_all_id, dim=-1)
            # # print(input_all_text.shape, input_all_id.shape)
            # # os._exit(0)
            # score_embs = self.fusion_module(input_all_id, input_all_text)
            ## TO MAKE text+id
            if args.text_feature_path in [None, 'None', 'none', '']:
                score_embs = self.text_encoder(sample_items_text.long())
            else:
                if torch.isnan(sample_items_text).any():
                    print('sample_items_text has nan')
                    print(sample_items_text)
                    os._exit(0)
                score_embs = self.text_encoder(sample_items_text)
            if torch.isnan(score_embs).any():
                print('score_embs has nan')
                os._exit(0)
        elif 'image' == args.item_tower:
            score_embs = self.image_encoder(sample_items_image)
        elif 'video' == args.item_tower:
            score_embs = self.video_encoder(sample_items_video)
        elif 'id' == args.item_tower:
            score_embs = self.id_encoder(sample_items_id)

        input_embs = score_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim) if sequence_fused_embs is None else sequence_fused_embs
        return input_embs, score_embs, debias_logits, text_video_loss, text_video_item_loss, text_video_seq_loss

    def encode_user_sequence(self, input_embs, log_mask, local_rank):
        if self.args.model == 'sasrec':
            prec_vec = self.user_encoder(input_embs[:, :-1, :], log_mask, local_rank)
        else:
            prec_vec = self.user_encoder(input_embs[:, :-1, :])
        return prec_vec.reshape(-1, self.args.embedding_dim)

    def compute_rec_loss(self, prec_vec, score_embs, debias_logits, sample_items_id, log_mask, local_rank):
        logits = torch.matmul(prec_vec, score_embs.t())
        logits = logits - debias_logits

        bs, seq_len = log_mask.size()
        label = torch.arange(bs * (seq_len + 1)).reshape(bs, seq_len + 1)
        label = label[:, 1:].to(local_rank).view(-1)

        flatten_item_seq = sample_items_id
        user_history = torch.zeros(bs, seq_len + 2).type_as(sample_items_id)
        user_history[:, :-1] = sample_items_id.view(bs, -1)
        user_history = user_history.unsqueeze(-1).expand(-1, -1, len(flatten_item_seq))
        history_item_mask = (user_history == flatten_item_seq).any(dim=1)
        history_item_mask = history_item_mask.repeat_interleave(seq_len, dim=0)
        unused_item_mask = torch.scatter(history_item_mask, 1, label.view(-1, 1), False)

        logits[unused_item_mask] = -1e4
        indices = torch.where(log_mask.view(-1) != 0)
        logits = logits.view(bs * seq_len, -1)
        rec_loss = self.criterion(logits[indices], label[indices])

        loss = rec_loss
        user = prec_vec.view(-1, self.max_seq_len, self.args.embedding_dim)[:, -1, :]
        item = score_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim)[:, -1, :]
        align = self.alignment(user, item)
        uniform = (self.uniformity(user) + self.uniformity(item)) / 2
        return loss, rec_loss, align, uniform

    def forward(self, sample_items_id, sample_items_text, sample_items_image, sample_items_video, log_mask, local_rank, args):
        input_embs, score_embs, debias_logits, text_video_loss, text_video_item_loss, text_video_seq_loss = \
            self.encode_item_sequence(sample_items_id, sample_items_text, sample_items_image, sample_items_video, log_mask, local_rank, args)

        prec_vec = self.encode_user_sequence(input_embs, log_mask, local_rank)

        loss, rec_loss, align, uniform = self.compute_rec_loss(prec_vec, score_embs, debias_logits, sample_items_id, log_mask, local_rank)

        if text_video_loss is not None:
            loss = loss + self.args.text_video_lambda * text_video_loss
        else:
            text_video_loss = loss.new_zeros(())

        if text_video_item_loss is not None:
            loss = loss + self.args.text_video_item_lambda * text_video_item_loss
        else:
            text_video_item_loss = loss.new_zeros(())

        if text_video_seq_loss is not None:
            loss = loss + self.args.text_video_seq_lambda * text_video_seq_loss
        else:
            text_video_seq_loss = loss.new_zeros(())

        return (
            loss,
            align,
            uniform,
            rec_loss.detach(),
            text_video_loss.detach(),
            text_video_item_loss.detach(),
            text_video_seq_loss.detach(),
        )
