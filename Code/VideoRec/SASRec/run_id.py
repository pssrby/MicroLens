import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_data_dir = '~/dataset/'
root_model_dir = '~/model/'

# dataset = 'bili_food'
# tag = 'bilibili_food_'
# behaviors = tag + 'trans_users.tsv'
# text_data = tag + 'trans_items_texts.tsv'
# image_data = tag + 'trans_items_images.lmdb'
dataset = 'MicroLens-100k-Dataset'
tag = 'MicroLens-100k_'
behaviors = tag + 'pairs.tsv'
text_data = tag + 'title_en.csv'
image_data = tag + 'covers_default.lmdb'

frame_interval = 1
frame_no = 5
video_data = tag + 'frames_interval_'+str(frame_interval)+'_number_'+str(frame_no)+'.lmdb'
max_seq_len_list = [10]

logging_num = 10
testing_num = 1
save_step = 1

image_resize = 224
max_video_no = 34321 # 34321 for 10wu

text_model_load = 'bert-small' #'bert-base-uncased' # 'bert-base-cn' 
# text_model_load = 'xlm-roberta-base'
image_model_load = 'clip-vit-base-patch32' # 'vit-b-32-clip' 'resnet50'
video_model_load = 'slowfast-50' # video-mae

# last 2 layer of trms
text_freeze_paras_before = 40 #165 198
image_freeze_paras_before = 9999 #164
video_freeze_paras_before = 9999 #270

mode = 'test' # train test
item_tower = 'video' # modal, text, image, video, id, text_image, text_video

epoch = 50
load_ckpt_name = 'None'
# load_ckpt_name = 'epoch-200.pt'
load_ckpt_name = 'epoch-15.pt'

weight_decay = 0.1
drop_rate = 0.1
batch_size_list = [512]

embedding_dim_list = [512]
lr_list = [1e-4]
text_fine_tune_lr_list = [1e-4]
image_fine_tune_lr_list = [1e-4]
video_fine_tune_lr_list = [1e-4]
index_list = [0]

scheduler = 'step_schedule_with_warmup'
scheduler_gap = 1
scheduler_alpha = 1
version = 'v4' # for recording different fusion methods, to be added in the futurek
#1:pos 2:item 3:pos+item 4:seq
num_workers = 4
fusion_method = 'coattentionsingle' # none, sum, concat, film, gated, moe, co_att, merge_attn, coattnfusion, coattentionsingle, crossattentionsingle, crossattentionseq, coattentionseq
text_ckpt_path = None #'./pretrain/epoch-44.pt'
image_ckpt_path = None
video_ckpt_path = None #'./pretrain/epoch-45.pt'
video_feature_path = '~/dataset/MicroLens-100k-Dataset/MicroLens-100k_frames_interval_1_number_5_encoder2.lmdb' # to input, if using pre-extracted video features
text_feature_path = None #'~/dataset/MicroLens-100k-Dataset/MicroLens-100k_title_en_encoder2.pt' # to input, if using pre-extracted text features
use_text_video_contrastive = 0
text_video_lambda = 0.2
text_video_tau = 0.07
use_text_video_item_contrastive = 1
text_video_item_lambda = 0.1
text_video_item_tau = 0.07
use_text_video_seq_contrastive = 0
text_video_seq_lambda = 0.1
text_video_seq_tau = 0.07

for batch_size in batch_size_list:
    for embedding_dim in embedding_dim_list:
        for max_seq_len in max_seq_len_list:
            for index in index_list:
                text_fine_tune_lr = text_fine_tune_lr_list[index]
                image_fine_tune_lr = image_fine_tune_lr_list[index]
                video_fine_tune_lr = video_fine_tune_lr_list[index]
                lr = lr_list[index]   

                label_screen = '{}_bs{}_ed{}_lr{}_dp{}_L2{}_len{}'.format(
                        item_tower, batch_size, embedding_dim, lr,
                        drop_rate, weight_decay, max_seq_len)

                run_py = "CUDA_VISIBLE_DEVICES='0,1,2,3' \
                        torchrun \
                        --nproc_per_node 4 --master_port 29500 main.py \
                        --root_data_dir {} --root_model_dir {} --dataset {} --behaviors {} --text_data {}  --image_data {} --video_data {}\
                        --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --save_step {}\
                        --testing_num {} --weight_decay {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {}\
                        --image_resize {} --image_model_load {} --text_model_load {} --video_model_load {} --epoch {} \
                        --text_freeze_paras_before {} --image_freeze_paras_before {} --video_freeze_paras_before {} --max_seq_len {} --frame_interval {} --frame_no {}\
                        --text_fine_tune_lr {} --image_fine_tune_lr {} --video_fine_tune_lr {}\
                        --scheduler {} --scheduler_gap {} --scheduler_alpha {} --max_video_no {}\
                        --version {} \
                        --num_workers {} --fusion_method {} \
                        --text_ckpt_path {} --image_ckpt_path {} --video_ckpt_path {} --video_feature_path {} --text_feature_path {} \
                        --use_text_video_contrastive {} --text_video_lambda {} --text_video_tau {} \
                        --use_text_video_item_contrastive {} --text_video_item_lambda {} --text_video_item_tau {} \
                        --use_text_video_seq_contrastive {} --text_video_seq_lambda {} --text_video_seq_tau {}".format(
                        root_data_dir, root_model_dir, dataset, behaviors, text_data, image_data, video_data,
                        mode, item_tower, load_ckpt_name, label_screen, logging_num, save_step,
                        testing_num,weight_decay, drop_rate, batch_size, lr, embedding_dim,
                        image_resize, image_model_load, text_model_load, video_model_load, epoch,
                        text_freeze_paras_before, image_freeze_paras_before, video_freeze_paras_before, max_seq_len, frame_interval, frame_no,
                        text_fine_tune_lr, image_fine_tune_lr, video_fine_tune_lr, 
                        scheduler, scheduler_gap, scheduler_alpha, max_video_no,
                        version, num_workers, fusion_method, 
                        text_ckpt_path, image_ckpt_path, video_ckpt_path, video_feature_path, text_feature_path,
                        use_text_video_contrastive, text_video_lambda, text_video_tau,
                        use_text_video_item_contrastive, text_video_item_lambda, text_video_item_tau,
                        use_text_video_seq_contrastive, text_video_seq_lambda, text_video_seq_tau)
            
                os.system(run_py)
