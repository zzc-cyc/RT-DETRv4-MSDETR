from detrex.config import get_config
from ..models.dino_swin_large_384 import model
import copy

# get default config
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "projects_dev/swin_large_patch4_window12_384_22kto1k.pth"
train.output_dir = "./output/dino_swin_large_384_4scale_12ep"

# max training iterations
train.max_iter = 90000
train.eval_period = 5000
train.log_period = 1000
train.checkpointer.period = 10000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 16



# more dn queries, set 300 here
model.dn_number = 100


model.criterion.weight_dict = {
            "loss_class": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_class_dn": 1,
            "loss_bbox_dn": 5.0,
            "loss_giou_dn": 2.0,
            "loss_class_o2m": 1.0,
            "loss_bbox_o2m": 5.0,
            "loss_giou_o2m": 2.0,
            "loss_class_sep": 1.0,
            "loss_bbox_sep": 5.0,
            "loss_giou_sep": 2.0,
            "loss_class_o2m_enc": 1.0,
            "loss_bbox_o2m_enc": 5.0,
            "loss_giou_o2m_enc": 2.0,
            "loss_vfl": 1.0,
            "loss_vfl_dn": 1.0,
            "loss_vfl_enc": 1.0,
        }

# set aux loss weight dict
base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    aux_weight_dict.update({k + "_enc": v for k, v in base_weight_dict.items()})
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict

# output dir
train.output_dir = "./output/mrdetrpp_swinl_4scale_12ep"

train.amp.enabled = False 
dataloader.train.total_batch_size = 16
dataloader.train.num_workers = 16 
train.checkpointer.period = 10000 

model.select_box_nums_for_evaluation = 300 
model.num_queries = 900 
model.transformer.encoder.use_checkpoint = True
model.backbone.use_checkpoint=True
model.box_noise_scale = 0.4
model.dn_number = 300