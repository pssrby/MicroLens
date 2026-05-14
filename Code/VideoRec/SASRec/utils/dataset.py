import os
import math
import lmdb
import torch
import pickle
import random

import numpy as np
import torchvision as tv
import torch.distributed as dist

import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

class LMDB_VIDEO:
    def __init__(self, video):
        self.video = video.tobytes()
        
class LMDB_Image:
    def __init__(self, image, id):
        self.image = image.tobytes()


class ItemsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]

class ModalDataset(Dataset):
    def __init__(self, u2seq, item_content, max_seq_len, item_num, text_size, image_db_path, video_db_path, item_id_to_keys, resize, args):
        self.u2seq = u2seq
        self.item_content = item_content
        self.max_seq_len =  max_seq_len + 1
        self.item_num = item_num
        self.text_size = text_size
        self.image_db_path = image_db_path
        self.video_db_path = video_db_path
        self.item_id_to_keys = item_id_to_keys
        self.resize = resize
        self.args = args

        self.transform = transforms.Compose([
            tv.transforms.Resize((self.resize, self.resize)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.u2seq)

    def worker_init_fn(self, worker_id):
        initial_seed = torch.initial_seed() % 2 ** 31
        worker_seed = initial_seed + worker_id + self.args.local_rank + 8 * self.args.node_rank 
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    def __getitem__(self, index):
        seq = self.u2seq[index]
        seq_Len = len(seq)
        tokens = seq[:-1]
        tokens_Len = len(tokens)
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len
        use_text = self.args.item_tower in ['text', 'text_image', 'text_video']
        use_image = self.args.item_tower in ['image', 'text_image', 'modal']
        use_video = self.args.item_tower in ['video', 'text_video', 'modal']
        sample_items_id = None
        sample_items_image = None
        sample_items_video = None
        sample_items_text = None

        if use_text:
            if self.args.text_feature_path in [None, 'None', 'none', '']:
                sample_items_text = np.zeros((self.max_seq_len, self.text_size * 2), dtype=np.int64)
            else:
                sample_items_text = np.zeros((self.max_seq_len, self.item_content.shape[1]), dtype=np.float32)

        if use_image:
            sample_items_image = np.zeros((self.max_seq_len, 3, self.resize, self.resize), dtype=np.float32)

        if use_video:
            if self.args.video_feature_path in [None, 'None', 'none', '']:
                sample_items_video = np.zeros((self.max_seq_len, self.args.frame_no, 3, 224, 224), dtype=np.float32)
            else:
                sample_items_video = np.zeros((self.max_seq_len, 400), dtype=np.float32)
        sample_items_id = [0] * mask_len_head + seq

        ##################################### Text #####################################
        # For left-padding positions, avoid all-zero attention_mask rows.
        # sample_items_text layout: [input_ids(L) | attention_mask(L)]
        # Set attention_mask[0]=1 for padding rows.
        # if mask_len_head > 0:
        #     sample_items_text[:mask_len_head, self.text_size] = 1
        if use_text:
            for i in range(tokens_Len):
                sample_items_text[mask_len_head + i] = self.item_content[seq[i]]
            sample_items_text[mask_len_head + tokens_Len] = self.item_content[seq[-1]]
            if self.args.text_feature_path in [None, 'None', 'none', '']:
                sample_items_text = torch.LongTensor(sample_items_text)
            else:
                sample_items_text = torch.FloatTensor(sample_items_text)

        ##################################### Image #####################################
        if use_image:
            env = lmdb.open(self.image_db_path, subdir=os.path.isdir(self.image_db_path),
                            readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin() as txn:
                for i in range(tokens_Len):
                    # pos
                    IMAGE = pickle.loads(txn.get(self.item_id_to_keys[seq[i]].encode()))
                    image_trans = np.copy(np.frombuffer(IMAGE.image, dtype=np.float32)).reshape(3, 224, 224) 
                    sample_items_image[mask_len_head + i] = image_trans
                # target
                IMAGE = pickle.loads(txn.get(self.item_id_to_keys[seq[-1]].encode()))
                image_trans = np.copy(np.frombuffer(IMAGE.image, dtype=np.float32)).reshape(3, 224, 224) 
                sample_items_image[mask_len_head + tokens_Len] = image_trans
            sample_items_image = torch.FloatTensor(sample_items_image)

        ##################################### video #####################################
        if use_video:
            video_db_path = self.video_db_path
            if self.args.video_feature_path not in [None, 'None', 'none', '']:
                video_db_path = os.path.expanduser(self.args.video_feature_path)
            env = lmdb.open(video_db_path, subdir=os.path.isdir(video_db_path),
                            readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin() as txn:
                for i in range(tokens_Len):
                    # pos
                    VIDEO = pickle.loads(txn.get(self.item_id_to_keys[seq[i]].encode()))
                    if self.args.video_feature_path in [None, 'None', 'none', '']:
                        VIDEO = np.copy(np.frombuffer(VIDEO.video, dtype=np.float32)).reshape(self.args.frame_no, 3, 224, 224)
                    sample_items_video[mask_len_head + i] = VIDEO

                # target
                VIDEO = pickle.loads(txn.get(self.item_id_to_keys[seq[-1]].encode()))
                if self.args.video_feature_path in [None, 'None', 'none', '']:
                    VIDEO = np.copy(np.frombuffer(VIDEO.video, dtype=np.float32)).reshape(self.args.frame_no, 3, 224, 224)
                sample_items_video[mask_len_head + tokens_Len] = VIDEO
            sample_items_video = torch.FloatTensor(sample_items_video)
        if sample_items_text is None:
            sample_items_text = torch.empty(0)
        if sample_items_image is None:
            sample_items_image = torch.empty(0)
        if sample_items_video is None:
            sample_items_video = torch.empty(0)

        sample_items_id = torch.LongTensor(sample_items_id)
        return sample_items_id, sample_items_text, sample_items_image, sample_items_video, \
            torch.FloatTensor(log_mask)

class ImageDataset(Dataset):
    def __init__(self, u2seq, item_num, max_seq_len, db_path, item_id_to_keys, resize):
        self.u2seq = u2seq
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.db_path = db_path
        self.item_id_to_keys = item_id_to_keys
        self.resize = resize

        self.transform = transforms.Compose([
            tv.transforms.Resize((self.resize, self.resize)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.u2seq) 

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = len(seq) - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = np.zeros((self.max_seq_len, 3, self.resize, self.resize))
        sample_id_items = [0] * mask_len_head + seq

        env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
             readonly=True, lock=False,
             readahead=False, meminit=False)

        with env.begin() as txn:
            for i in range(tokens_Len):
                # pos
                IMAGE = pickle.loads(txn.get(self.item_id_to_keys[seq[i]].encode()))
                image_trans = np.copy(np.frombuffer(IMAGE.image, dtype=np.float32)).reshape(3, 224, 224) 
                sample_items[mask_len_head + i] = image_trans
            # target
            IMAGE = pickle.loads(txn.get(self.item_id_to_keys[seq[-1]].encode()))
            image_trans = np.copy(np.frombuffer(IMAGE.image, dtype=np.float32)).reshape(3, 224, 224) 
            sample_items[mask_len_head + tokens_Len] = image_trans

        sample_id_items = torch.LongTensor(sample_id_items)
        sample_items = torch.FloatTensor(sample_items)
        return sample_id_items, sample_items, torch.FloatTensor(log_mask)

class TextFeatureDataset(Dataset):
    def __init__(self, userseq, text_features, max_seq_len, item_num, item_id_to_keys):
        self.userseq = userseq
        self.text_features = text_features
        self.max_seq_len = max_seq_len + 1
        self.item_num = item_num
        self.item_id_to_keys = item_id_to_keys
        self.text_feature_dim = text_features.shape[1]

    def __len__(self):
        return len(self.userseq)

    def __getitem__(self, index):
        seq = self.userseq[index]
        seq_len = len(seq)
        tokens_len = seq_len - 1
        mask_len_head = self.max_seq_len - seq_len
        log_mask = [0] * mask_len_head + [1] * tokens_len

        sample_id_items = [0] * mask_len_head + seq
        sample_items = np.zeros((self.max_seq_len, self.text_feature_dim), dtype=np.float32)
        for i in range(tokens_len):
            sample_items[mask_len_head + i] = self.text_features[seq[i]]
        sample_items[mask_len_head + tokens_len] = self.text_features[seq[-1]]

        sample_items = torch.FloatTensor(sample_items)
        sample_id_items = torch.LongTensor(sample_id_items)
        return sample_id_items, sample_items, torch.FloatTensor(log_mask)

class TextDataset(Dataset):
    def __init__(self, userseq, item_content, max_seq_len, item_num, text_size):
        self.userseq = userseq
        self.item_content = item_content
        self.max_seq_len =  max_seq_len + 1
        self.item_num = item_num
        self.text_size = text_size

    def __len__(self):
        return len(self.userseq)

    def __getitem__(self, index):
        seq = self.userseq[index]
        seq_Len = len(seq)
        tokens = seq[:-1]
        tokens_Len = len(tokens)
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len
        
        sample_id_items = [0] * mask_len_head + seq
        sample_items = np.zeros((self.max_seq_len, self.text_size * 2))
        # For left-padding positions, avoid all-zero attention_mask rows.
        # sample_items layout: [input_ids(L) | attention_mask(L)]
        # Set attention_mask[0]=1 for padding rows.
        # if mask_len_head > 0:
        #     sample_items[:mask_len_head, self.text_size] = 1
        for i in range(tokens_Len):
            # pos
            sample_items[mask_len_head + i] = self.item_content[seq[i]]
        # target
        sample_items[mask_len_head + tokens_Len] = self.item_content[seq[-1]]
        sample_items = torch.FloatTensor(sample_items)
        sample_id_items = torch.LongTensor(sample_id_items)
        return sample_id_items, sample_items, torch.FloatTensor(log_mask)

class VideoFeatureDataset(Dataset):
    def __init__(self, u2seq, item_num, max_seq_len, item_id_to_keys, feature_db_path):
        self.u2seq = u2seq
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.item_id_to_keys = item_id_to_keys
        self.db_path = feature_db_path

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = len(seq) - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = np.zeros((self.max_seq_len, 400)) 
        sample_id_items = [0] * mask_len_head + seq

        env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                        readonly=True, lock=False, readahead=False, meminit=False)

        with env.begin() as txn:
            for i in range(tokens_Len):
                # pos
                VIDEO = pickle.loads(txn.get(self.item_id_to_keys[seq[i]].encode()))
                # VIDEO = np.copy(np.frombuffer(VIDEO.video, dtype=np.float32)).reshape(self.frame_no, 3, 224, 224) 
                sample_items[mask_len_head + i] = VIDEO

            # target
            VIDEO = pickle.loads(txn.get(self.item_id_to_keys[seq[-1]].encode()))
            # VIDEO = np.copy(np.frombuffer(VIDEO.video, dtype=np.float32)).reshape(self.frame_no, 3, 224, 224) 
            sample_items[mask_len_head + tokens_Len] = VIDEO

        sample_id_items = torch.LongTensor(sample_id_items)
        sample_items = torch.FloatTensor(sample_items)
        return sample_id_items, sample_items, torch.FloatTensor(log_mask)
class VideoDataset(Dataset):
    def __init__(self, u2seq, item_num, max_seq_len, item_id_to_keys, db_path, frame_no):
        self.u2seq = u2seq
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.item_id_to_keys = item_id_to_keys
        self.video_lmdb_path = db_path
        self.frame_no = frame_no

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = len(seq) - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = np.zeros((self.max_seq_len, self.frame_no, 3, 224, 224)) 
        sample_id_items = [0] * mask_len_head + seq

        env = lmdb.open(self.video_lmdb_path, subdir=os.path.isdir(self.video_lmdb_path),
                        readonly=True, lock=False, readahead=False, meminit=False)

        with env.begin() as txn:
            for i in range(tokens_Len):
                # pos
                vdo = pickle.loads(txn.get(self.item_id_to_keys[seq[i]].encode()))
                vdo = np.copy(np.frombuffer(vdo.video, dtype=np.float32)).reshape(self.frame_no, 3, 224, 224) 
                sample_items[mask_len_head + i] = vdo

            # target
            vdo = pickle.loads(txn.get(self.item_id_to_keys[seq[-1]].encode()))
            vdo = np.copy(np.frombuffer(vdo.video, dtype=np.float32)).reshape(self.frame_no, 3, 224, 224) 
            sample_items[mask_len_head + tokens_Len] = vdo

        sample_id_items = torch.LongTensor(sample_id_items)
        sample_items = torch.FloatTensor(sample_items)
        return sample_id_items, sample_items, torch.FloatTensor(log_mask)

class IdDataset(Dataset):
    def __init__(self, u2seq, item_num, max_seq_len, args):
        self.u2seq = u2seq
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.args = args

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id][-self.max_seq_len:]
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = [0] * mask_len_head + seq
        sample_items = torch.LongTensor(np.array(sample_items))

        return sample_items, torch.FloatTensor(log_mask)

class IdClDataset(Dataset):
    def __init__(self, u2seq, item_num, max_seq_len, args):
        self.u2seq = u2seq
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.args = args
        self.aug_min_seq_len = max(1, getattr(args, 'cl_aug_min_seq_len', 1))
        self.crop_ratio = float(getattr(args, 'cl_crop_ratio', 0.8))
        self.mask_ratio = float(getattr(args, 'cl_mask_ratio', 0.2))
        self.reorder_ratio = float(getattr(args, 'cl_reorder_ratio', 0.2))

    def __len__(self):
        return len(self.u2seq)

    def _pad_and_mask(self, seq):
        seq = seq[-self.max_seq_len:]
        seq_len = len(seq)
        tokens_len = max(seq_len - 1, 0)
        mask_len_head = self.max_seq_len - seq_len
        log_mask = [0] * mask_len_head + [1] * tokens_len
        sample_items = [0] * mask_len_head + seq
        return torch.LongTensor(np.array(sample_items)), torch.FloatTensor(log_mask)

    def _item_crop(self, seq):
        if len(seq) <= 2:
            return seq
        target_len = max(self.aug_min_seq_len, int(round((len(seq) - 1) * self.crop_ratio)))
        target_len = min(target_len, len(seq) - 1)
        if target_len <= 0:
            return seq
        start = random.randint(0, len(seq) - 1 - target_len)
        cropped_tokens = seq[start:start + target_len]
        return cropped_tokens + [seq[-1]]

    def _item_mask(self, seq):
        if len(seq) <= 2:
            return seq
        augmented = list(seq)
        candidate_indices = list(range(len(seq) - 1))
        n_mask = max(1, int(round(len(candidate_indices) * self.mask_ratio)))
        n_mask = min(n_mask, len(candidate_indices))
        for idx in random.sample(candidate_indices, k=n_mask):
            augmented[idx] = 0
        return augmented

    def _item_reorder(self, seq):
        if len(seq) <= 3:
            return seq
        augmented = list(seq)
        reorder_len = max(2, int(round((len(seq) - 1) * self.reorder_ratio)))
        reorder_len = min(reorder_len, len(seq) - 1)
        start = random.randint(0, len(seq) - 1 - reorder_len)
        sub_seq = augmented[start:start + reorder_len]
        random.shuffle(sub_seq)
        augmented[start:start + reorder_len] = sub_seq
        return augmented

    def _augment(self, seq):
        aug_ops = [self._item_crop, self._item_mask, self._item_reorder]
        aug_op = random.choice(aug_ops)
        augmented = aug_op(list(seq))
        if len(augmented) < 2:
            return list(seq)
        return augmented

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id][-self.max_seq_len:]
        original_items, original_mask = self._pad_and_mask(seq)
        aug_seq_1 = self._augment(seq)
        aug_seq_2 = self._augment(seq)
        aug_items_1, aug_mask_1 = self._pad_and_mask(aug_seq_1)
        aug_items_2, aug_mask_2 = self._pad_and_mask(aug_seq_2)
        return original_items, aug_items_1, aug_items_2, original_mask, aug_mask_1, aug_mask_2

class IdEvalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]

class EvalDataset(Dataset):
    def __init__(self, u2seq, item_content, max_seq_len, item_num):
        self.u2seq = u2seq
        self.item_content = item_content
        self.max_seq_len = max_seq_len
        self.item_num = item_num

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id][-self.max_seq_len:]
        tokens = seq[:-1]
        target = seq[-1]
        mask_len = self.max_seq_len - len(seq)
        pad_tokens = [0] * mask_len + tokens
        log_mask = [0] * mask_len + [1] * len(tokens)
        input_embs = self.item_content[pad_tokens]
        labels = np.zeros(self.item_num)
        labels[target - 1] = 1.0
        return torch.LongTensor([user_id]), input_embs, torch.FloatTensor(log_mask), labels

class LmdbEvalDataset(Dataset):
    def __init__(self, data, item_id_to_keys, db_path, resize, mode, frame_no=-1):
        self.data = data
        self.item_id_to_keys = item_id_to_keys
        self.db_path = db_path
        self.resize = resize
        self.mode = mode
        self.frame_no = frame_no
        if mode == 'image':
            self.padding_emb = torch.zeros((3, 224, 224)) 
        elif mode == 'video_feature':
            self.padding_emb = torch.zeros((400,))
        else:
            self.padding_emb = torch.zeros((self.frame_no, 3, 224, 224)) 

        # self.transform = transforms.Compose([
        #         tv.transforms.Resize((self.resize, self.resize)),
        #         tv.transforms.ToTensor(),
        #         tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #     ])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        item_id = self.data[index]
        if index == 0:
            if self.mode == 'image':
                return torch.zeros((3, 224, 224)) 
            elif self.mode == 'video':
                return torch.zeros((self.frame_no, 3, 224, 224)) 
            elif self.mode == 'video_feature':
                return torch.zeros((400,)) 

        env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path), \
            readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            byteflow = txn.get(self.item_id_to_keys[item_id].encode())
        if self.mode == 'image':
            IMAGE = pickle.loads(byteflow)
            output = np.frombuffer(IMAGE.image, dtype=np.float32).reshape(3, 224, 224) 
        elif self.mode == 'video':
            VIDEO = pickle.loads(byteflow)
            output = np.frombuffer(VIDEO.video, dtype=np.float32).reshape(self.frame_no, 3, 224, 224) 
        elif self.mode == 'video_feature':
            VIDEO_FEATURE = pickle.loads(byteflow)
            output = VIDEO_FEATURE
        return torch.FloatTensor(output)

class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples
