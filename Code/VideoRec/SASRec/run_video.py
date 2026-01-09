import os
import subprocess
import runpy, sys
import shlex
work_path = "/data4/guangyi/MicroLens/Code/VideoRec/SASRec"
os.chdir(work_path)

# ==================== 调试配置 ====================
DEBUG_MODE = True          # True: 单进程调试模式(单卡), False: 多GPU训练模式
DEBUG_WAIT = True          # True: 注入 debugpy 并等待你附加调试器
DEBUG_HOST = "127.0.0.1"
DEBUG_PORT = 5678

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_data_dir = '/data4/guangyi/MicroLens-Dataset/100k/'
root_model_dir = '/data4/guangyi/MicroLens/model/'

dataset = 'MicroLens'
tag = 'MicroLens-100k_'
# behaviors = tag + '_ks_pairs.tsv'
behaviors = tag + 'pairs.tsv'
# behaviors = 'MicroLens-100k_pairs.tsv'
text_data = tag + 'title.csv'
image_data = tag + 'cover.lmdb'
frame_interval = 1
frame_no = 5
video_data = tag + 'frames_interval_'+str(frame_interval)+'_number_'+str(frame_no)+'.lmdb'
max_seq_len_list = [10]

logging_num = 10
testing_num = 1
save_step = 1

image_resize = 224
max_video_no = 34321 # 34321 for 10wu, 91717 for 100wu

text_model_load = 'bert-base-uncased' # 'bert-base-cn' 
image_model_load = 'vit-base-mae' # 'vit-b-32-clip'

# last 2 layer of trms
text_freeze_paras_before = 165
image_freeze_paras_before = 164


'''
model name: freeze layer/batch size
video-mae: 9999/45 152/45 92/18 0/10 (frozen, topT, halfT, FT, TFS)
r3d18: 9999/32 30/32 xx/xx 0/20
r3d50: 9999/70 168/70 xx/xx 0/10
c2d50: 9999/45 129/45 xx/xx 0/10
i3d50: 9999/45 72/45 xx/xx 0/15
csn101: 9999/45 282/45 xx/xx 0/10
slow50: 9999/45 129/45 xx/xx 0/10
efficient-x3d-s, efficient-x3d-xs: 9999/100 228/100 xx/xx 0/30
x3d-l: 9999/90 455/90 xx/xx 0/12
x3d-m, x3d-s, x3d-xs: 9999/110 226/110 xx/xx 0/20
mvit-base-16(0/5), mvit-base-16x4(0/5), mvit-base-32x3: 9999/30 326/30 192/30 0/10 0/10-scratch
slowfast-50: 9999/120 270/120 153/60 0/20 0/20-scratch
slowfast16x8-101: 9999/120 576/120 153/25 0/15 0/15-scratch
'''
video_model_load = 'video-mae' # mvit-base-32x3 slowfast-50 slowfast16x8-101
video_freeze_paras_before = 576 # 326 270 576
batch_size_list = [20] # 30 120 120

mode = 'train' # train test
item_tower = 'id' # modal, text, image, video, id

epoch = 50
load_ckpt_name = 'None'
# load_ckpt_name = 'epoch-47.pt'

weight_decay = 0.1
drop_rate = 0.1

embedding_dim_list = [512]
lr_list = [1e-5]
text_fine_tune_lr_list = [1e-4]
image_fine_tune_lr_list = [1e-4]
video_fine_tune_lr_list = [1e-4]
index_list = [0]

scheduler = 'step_schedule_with_warmup'
scheduler_gap = 1
scheduler_alpha = 1
version = 'v1'

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

                if DEBUG_MODE:
                    # ==================== 调试模式：直接运行main.py ====================
                    # 设置命令行参数
                    sys.argv = [
                        'main.py',
                        '--root_data_dir', root_data_dir,
                        '--root_model_dir', root_model_dir,
                        '--dataset', dataset,
                        '--behaviors', behaviors,
                        '--text_data', text_data,
                        '--image_data', image_data,
                        '--video_data', video_data,
                        '--mode', mode,
                        '--item_tower', item_tower,
                        '--load_ckpt_name', load_ckpt_name,
                        '--label_screen', label_screen,
                        '--logging_num', str(logging_num),
                        '--save_step', str(save_step),
                        '--testing_num', str(testing_num),
                        '--weight_decay', str(weight_decay),
                        '--drop_rate', str(drop_rate),
                        '--batch_size', str(batch_size),
                        '--lr', str(lr),
                        '--embedding_dim', str(embedding_dim),
                        '--image_resize', str(image_resize),
                        '--image_model_load', image_model_load,
                        '--text_model_load', text_model_load,
                        '--video_model_load', video_model_load,
                        '--epoch', str(epoch),
                        '--text_freeze_paras_before', str(text_freeze_paras_before),
                        '--image_freeze_paras_before', str(image_freeze_paras_before),
                        '--video_freeze_paras_before', str(video_freeze_paras_before),
                        '--max_seq_len', str(max_seq_len),
                        '--frame_interval', str(frame_interval),
                        '--frame_no', str(frame_no),
                        '--text_fine_tune_lr', str(text_fine_tune_lr),
                        '--image_fine_tune_lr', str(image_fine_tune_lr),
                        '--video_fine_tune_lr', str(video_fine_tune_lr),
                        '--scheduler', scheduler,
                        '--scheduler_gap', str(scheduler_gap),
                        '--scheduler_alpha', str(scheduler_alpha),
                        '--max_video_no', str(max_video_no),
                        '--version', version
                    ]
                    
                    # 设置单GPU环境变量（调试时使用单卡）
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                    
                    # 设置分布式训练环境变量（单进程模式）
                    os.environ['RANK'] = '0'              # 全局进程排名
                    os.environ['LOCAL_RANK'] = '0'        # 本地进程排名
                    os.environ['WORLD_SIZE'] = '1'        # 总进程数
                    os.environ['MASTER_ADDR'] = 'localhost'  # 主节点地址
                    os.environ['MASTER_PORT'] = '12355'   # 主节点端口（避免与1024冲突）
                    
                    print(f"[DEBUG MODE] 使用单卡调试模式运行: {label_screen}")
                    print(f"[DEBUG MODE] GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
                    print(f"[DEBUG MODE] 分布式配置: RANK=0, WORLD_SIZE=1 (单进程模式)")
                    
                    # 直接运行main.py，VSCode调试器可以跟踪进入
                    runpy.run_path('main.py', run_name='__main__')
                    
                else:
                    # ==================== 训练模式：多GPU分布式训练 ====================
                    run_py = "CUDA_VISIBLE_DEVICES='0,1,2,3' \
                            python -m torch.distributed.launch \
                            --nproc_per_node 4 --master_port 1024 main.py \
                            --root_data_dir {} --root_model_dir {} --dataset {} --behaviors {} --text_data {}  --image_data {} --video_data {}\
                            --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --save_step {}\
                            --testing_num {} --weight_decay {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {}\
                            --image_resize {} --image_model_load {} --text_model_load {} --video_model_load {} --epoch {} \
                            --text_freeze_paras_before {} --image_freeze_paras_before {} --video_freeze_paras_before {} --max_seq_len {} --frame_interval {} --frame_no {}\
                            --text_fine_tune_lr {} --image_fine_tune_lr {} --video_fine_tune_lr {}\
                            --scheduler {} --scheduler_gap {} --scheduler_alpha {} --max_video_no {}\
                            --version {}".format(
                            root_data_dir, root_model_dir, dataset, behaviors, text_data, image_data, video_data,
                            mode, item_tower, load_ckpt_name, label_screen, logging_num, save_step,
                            testing_num,weight_decay, drop_rate, batch_size, lr, embedding_dim,
                            image_resize, image_model_load, text_model_load, video_model_load, epoch,
                            text_freeze_paras_before, image_freeze_paras_before, video_freeze_paras_before, max_seq_len, frame_interval, frame_no,
                            text_fine_tune_lr, image_fine_tune_lr, video_fine_tune_lr, 
                            scheduler, scheduler_gap, scheduler_alpha, max_video_no,
                            version)
                    
                    print(f"[TRAIN MODE] 使用多卡训练模式运行: {label_screen}")
                    print(f"[TRAIN MODE] 使用 4 个GPU进行分布式训练")
                    os.system(run_py)
