import torch
import numpy as np
from torch import nn
from torch.nn.init import xavier_normal_
from collections import Counter
import torch.nn.functional as F

from .text_encoders import TextEmbedding
from .video_encoders import VideoMaeEncoder, R3D18Encoder, R3D50Encoder, C2D50Encoder
from .video_encoders import I3D50Encoder, CSN101Encoder, SLOW50Encoder, EX3DSEncoder
from .video_encoders import EX3DXSEncoder, X3DXSEncoder, X3DSEncoder, X3DMEncoder
from .video_encoders import X3DLEncoder, MVIT16Encoder, MVIT16X4Encoder, MVIT32X3Encoder
from .video_encoders import SLOWFAST50Encoder, SLOWFAST16X8101Encoder
from .image_encoders import VitEncoder, ResnetEncoder, MaeEncoder, SwinEncoder 
from .fusion_module import MoEFusion, SumFusion, ConcatFusion, FiLM, GatedFusion 
from .user_encoders import User_Encoder_GRU4Rec, User_Encoder_SASRec, User_Encoder_NextItNet

class Model(torch.nn.Module):
    def __init__(self, args, pop_prob_list, item_num, bert_model, image_net, video_net, text_content=None):
        super(Model, self).__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.item_num = item_num
        self.pop_prob_list = torch.FloatTensor(pop_prob_list)

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
            self.text_content = torch.LongTensor(text_content)
            self.text_encoder = TextEmbedding(args=args, bert_model=bert_model)
        
        if args.item_tower in ('video', 'modal', 'text_video'):
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

        self.id_encoder = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        xavier_normal_(self.id_encoder.weight.data)

        self.criterion = nn.CrossEntropyLoss()

        fusion = args.fusion_method.lower()
        if fusion == 'concat' and args.item_tower in ('modal', 'text_image', 'text_video', 'text'):
            self.fusion_module = ConcatFusion(args=args)
        elif fusion == 'moe' and args.item_tower in ('modal', 'text_image', 'text_video', 'text'):
            num_modalities = 3 if args.item_tower == 'modal' else 2
            self.fusion_module = MoEFusion(args=args, num_modalities=num_modalities)
        elif fusion == 'sum' and args.item_tower in ('modal', 'text_image', 'text_video', 'text'):
            self.fusion_module = SumFusion(args=args)
        elif fusion == 'film' and args.item_tower in ('modal', 'text_image', 'text_video', 'text'):
            self.fusion_module = FiLM(args=args)
        elif fusion == 'gated' and args.item_tower in ('modal', 'text_image', 'text_video', 'text'):
            self.fusion_module = GatedFusion(args=args)
        else:
            # fallback to concat for modal as default, or no fusion for single modality.
            if args.item_tower in ('modal', 'text_image', 'text_video', 'text'):
                self.fusion_module = ConcatFusion(args=args)

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def forward(self, sample_items_id, sample_items_text, sample_items_image, sample_items_video, log_mask, local_rank, args):
        self.pop_prob_list = self.pop_prob_list.to(local_rank)
        debias_logits = torch.log(self.pop_prob_list[sample_items_id.view(-1)])

        if 'modal' == args.item_tower:
            input_all_text = self.text_encoder(sample_items_text.long())
            input_all_image = self.image_encoder(sample_items_image)
            input_all_video = self.video_encoder(sample_items_video)
            score_embs = self.fusion_module(input_all_text, input_all_image, input_all_video)
        elif 'text_image' == args.item_tower:
            input_all_text = self.text_encoder(sample_items_text.long())
            input_all_image = self.image_encoder(sample_items_image)
            score_embs = self.fusion_module(input_all_text, input_all_image)
        elif 'text_video' == args.item_tower:
            input_all_text = self.text_encoder(sample_items_text.long())
            input_all_video = self.video_encoder(sample_items_video)
            score_embs = self.fusion_module(input_all_text, input_all_video)
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
            score_embs = self.text_encoder(sample_items_text.long())
        elif 'image' == args.item_tower:
            score_embs = self.image_encoder(sample_items_image)
        elif 'video' == args.item_tower:
            score_embs = self.video_encoder(sample_items_video)
        elif 'id' == args.item_tower:
            score_embs = self.id_encoder(sample_items_id)

        input_embs = score_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim)        
        if self.args.model == 'sasrec':
            prec_vec = self.user_encoder(input_embs[:, :-1, :], log_mask, local_rank)
        else:
            prec_vec = self.user_encoder(input_embs[:, :-1, :])
        prec_vec = prec_vec.reshape(-1, self.args.embedding_dim)

        ######################################  IN-BATCH CROSS-ENTROPY LOSS  ######################################
        # logits = torch.matmul(F.normalize(prec_vec, dim=-1), F.normalize(score_embs, dim=-1).t()) # (bs * max_seq_len, bs * (max_seq_len + 1))
        # logits = logits / self.args.tau - debias_logits
        logits = torch.matmul(prec_vec, score_embs.t())
        logits = logits - debias_logits

        ###################################### MASK USELESS ITEM ######################################
        bs, seq_len = log_mask.size(0), log_mask.size(1)
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
        loss = self.criterion(logits[indices], label[indices])

        ###################################### CALCULATE ALIGNMENT AND UNIFORMITY ######################################
        user = prec_vec.view(-1, self.max_seq_len, self.args.embedding_dim)[:, -1, :]
        item = score_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim)[:, -1, :]
        align = self.alignment(user, item)
        uniform = (self.uniformity(user) + self.uniformity(item)) / 2
        
        return loss, align, uniform
