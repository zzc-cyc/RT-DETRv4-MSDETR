# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from detrex.modeling.criterion import SetCriterion
from detrex.utils import get_world_size, is_dist_avail_and_initialized
from .one2manyMatcherDAC import Stage2Assigner
import torch.nn.functional as F


from detrex.layers import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
def _get_src_permutation_idx(indices):
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def get_vfl_loss(outputs, targets, indices, num_boxes, num_classes=80, **kwargs):
    assert "pred_ious" in outputs
    src_logits = outputs["pred_ious"]

    idx = _get_src_permutation_idx(indices)
    target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    target_classes = torch.full(
        src_logits.shape[:2],
        num_classes,
        dtype=torch.int64,
        device=src_logits.device,
    )
    target_classes[idx] = target_classes_o

    src_boxes = outputs['pred_boxes'][idx]
    target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
    ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
    ious = torch.diag(ious).detach()
    src_logits = outputs['pred_ious']
    target = F.one_hot(target_classes, num_classes=num_classes + 1)[..., :-1]

    target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
    target_score_o[idx] = ious.to(target_score_o.dtype)
    target_score = target_score_o.unsqueeze(-1) * target
    # target_score = torch.sqrt(target_score) 
    target_score = target_score **0.9

    pred_score = torch.sigmoid(src_logits).detach()
    
    alpha, gamma = 0.25, 2
    weight = alpha * pred_score.pow(gamma) * (1 - target) + target_score
    
    loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
    loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
    return {"loss_vfl": loss}

class TwoStageCriterion(SetCriterion):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses=["class", "boxes"],
        eos_coef=None,
        loss_class_type="focal_loss",
        alpha: float = 0.25,
        gamma: float = 2,
        two_stage_binary_cls=False,
    ):
        super().__init__(
            num_classes, matcher, weight_dict, losses, eos_coef, loss_class_type, alpha, gamma
        )
        self.two_stage_binary_cls = two_stage_binary_cls
        self.topk_matcher = Stage2Assigner(num_queries=900, max_k=6)
        self.rank_weight = None

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        losses.update(get_vfl_loss(outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "o2m" in outputs:
            for i, aux_outputs in enumerate(outputs["o2m"]):
                indices = self.topk_matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes * 3, userank=True)
                    l_dict = {k + f"_o2m_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        
        if "sep" in outputs:
            for i, aux_outputs in enumerate(outputs["sep"]):
                indices = self.topk_matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes * 3, userank=True)
                    l_dict = {k + f"_sep_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

                    
        
        
        # for two stage
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            if self.two_stage_binary_cls:
                for bt in targets:
                    bt["labels"] = torch.zeros_like(bt["labels"])
            indices = self.matcher(enc_outputs, targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)
            l_dict = get_vfl_loss(enc_outputs, targets, indices, num_boxes)
            l_dict = {k + "_enc": v for k, v in l_dict.items()}
            losses.update(l_dict)
            
            enc_outputs = outputs["o2m_enc"]
            indices = self.matcher(outputs["o2m_enc"], targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, outputs["o2m_enc"], targets, indices, num_boxes)
                l_dict = {k + "_o2m_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        

        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, **kwargs):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        src_boxes = outputs['pred_boxes'][idx].float()
        target_boxes = torch.cat([t['boxes'][i].float() for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        alpha, gamma = 0.25, 2
        
        # Ensure pred_score is in FP32 for operations that require higher precision
        pred_score = torch.sigmoid(src_logits)
        
        pos_weights = torch.zeros_like(src_logits)
        neg_weights = pred_score ** gamma
        pos_ind = [id for id in idx]
        pos_ind.append(target_classes_o)
        
        # Ensure t is in FP32 for operations that require higher precision
        t = pred_score[pos_ind].pow(alpha) * ious.pow(1 - alpha)
        t = torch.clamp(t, 0.01).detach()
        tau = 1.5
        # if 'userank' in kwargs and kwargs['userank']:
        #     rank = self.get_local_rank_o2m(t, indices)
        #     rank_weight = torch.exp(-rank/tau).cuda()
        # else:
        #     rank_weight = 1
        
        # t = t * rank_weight
        
        pos_weights[pos_ind] = t
        neg_weights[pos_ind] = 1 - t 
        
        # Ensure loss_ce is calculated in FP32
        loss_ce = - pos_weights * pred_score.log() - neg_weights * (1 - pred_score).log()
        loss_ce = loss_ce.sum().float() / num_boxes
        losses = {"loss_class": loss_ce}
        # self.rank_weight = rank_weight
        return losses
    
    def get_local_rank_o2m(self, quality, indices):
        #quality: one-dimension tensor 
        #indices: matching result
        bs = len(indices)
        device = quality.device
        tgt_size = [len(tgt_ind) for _,tgt_ind in indices]
        ind_start = 0
        rank_list = []
        for i in range(bs):
            if  tgt_size[i] == 0:
                rank_list.append(torch.zeros(0,dtype=torch.long,device=device))
                continue     
            num_tgt = max(indices[i][1]) + 1
            # split quality of one item
            quality_per_img = quality[ind_start:ind_start+tgt_size[i]]
            ind_start += tgt_size[i]

            rank_per_img = torch.zeros(len(indices[i][1]),dtype=torch.long,).cuda()

            int_unique, int_unique_count = indices[i][1].unique(return_counts=True)
            for j, id in enumerate(int_unique):
                if int_unique_count[j] == 1:
                    k_indices = torch.where(indices[i][1] == id)[0]
                    rank_per_img[k_indices] = 0
                if int_unique_count[j] > 1:
                    # ForkedPdb().set_trace()
                    k_indices = torch.where(indices[i][1] == id)[0]
                    k_indices = k_indices[torch.sort(quality_per_img[k_indices], descending=True)[1]]
                    keep_k = torch.arange(int_unique_count[j])
                    rank_per_img[k_indices] = keep_k.cuda()
            rank_list.append(rank_per_img.flatten())
        return torch.cat(rank_list, 0)

    