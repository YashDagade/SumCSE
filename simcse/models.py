import pdb
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

# =========================
# MLP Layer for [CLS]-based pooling
# =========================
class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x


# =========================
# Cosine Similarity Module
# =========================
class Similarity(nn.Module):
    """
    Dot product or cosine similarity, scaled by a temperature parameter.
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        # Return pairwise cosine similarity / temp
        return self.cos(x, y) / self.temp


# =========================
# Pooler for different pooling strategies
# =========================
class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding:
    - 'cls': [CLS] representation
    - 'cls_before_pooler': [CLS] representation (no MLP)
    - 'avg': average of last hidden states
    - 'avg_top2': average of the last two layers
    - 'avg_first_last': average of the first and the last layers
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            "cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"
        ], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ["cls_before_pooler", "cls"]:
            # Use the [CLS] token (position 0)
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            # Average over valid tokens
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) /
                    attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


# =========================
# Contrastive learning init
# =========================
def cl_init(cls, config):
    """
    Contrastive learning class init function.
    Sets up pooler, MLP (if needed), similarity function, etc.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()


# =========================
# Orthonormality Loss: skip same-instance pairs
# =========================
def compute_orthogonality_loss(embeddings: torch.Tensor,
                               margin: float,
                               num_sent: int) -> torch.Tensor:
    """
    Compute a batch-wise orthogonality penalty while skipping pairs that 
    belong to the same instance.

    L_ortho = average( ReLU(|cos(e_i, e_j)| - margin) ), for i != j 
              AND (i,j) not in the same instance.

    Args:
      embeddings: (N, D)  [N = b * num_sent]
      margin: threshold
      num_sent: number of sentences per instance (2 or 3)

    Returns:
      A scalar orthogonality loss
    """
    device = embeddings.device
    N = embeddings.size(0)

    # Cosine similarity matrix: shape (N, N)
    cos_sims = F.cosine_similarity(
        embeddings.unsqueeze(1),
        embeddings.unsqueeze(0),
        dim=2
    )
    abs_cos_sims = cos_sims.abs()

    # Mask out diagonal (i == j)
    diag_mask = torch.eye(N, device=device).bool()
    abs_cos_sims.masked_fill_(diag_mask, 0.0)

    # Mask out pairs from the same instance: i//num_sent == j//num_sent
    row_idx = torch.arange(N, device=device)
    col_idx = torch.arange(N, device=device)
    same_instance_mask = (row_idx.unsqueeze(1) // num_sent) == (
        col_idx.unsqueeze(0) // num_sent
    )
    

    abs_cos_sims.masked_fill_(same_instance_mask, 0.0)

    # ReLU penalty
    penalty = F.relu(abs_cos_sims - margin)

    # Count valid pairs
    total_pairs = N * (N - 1)  # exclude diagonal
    # number of pairs in each instance: num_sent * (num_sent - 1)
    # for b instances => b * num_sent*(num_sent-1)
    # but we don't know b directly, so b = N//num_sent
    b = N // num_sent
    same_inst_pairs = b * num_sent * (num_sent - 1)
    valid_pairs = total_pairs - same_inst_pairs

    L_ortho = penalty.sum() / valid_pairs
    
    # ipdb.set_trace()
    return L_ortho


# =========================
# Forward pass for contrastive learning
# =========================
def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    batch_size = input_ids.size(0)
    num_sent = input_ids.size(1)

    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1)))
    attention_mask = attention_mask.view((-1, attention_mask.size(-1)))
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))

    # Encode
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ["avg_top2", "avg_first_last"] else False,
        return_dict=True,
    )

    # Optional MLM encoding
    mlm_outputs = None
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,  # same attention
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ["avg_top2", "avg_first_last"] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    # Reshape back to (batch_size, num_sent, hidden_size)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))

    # Apply MLP if pooler_type == "cls"
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # Separate embeddings
    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather embeddings in multi-GPU training
    if dist.is_initialized() and cls.training:
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    # Contrastive loss
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # shape (B, B)
    if num_sent == 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], dim=1)  # shape (B, 2B)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # Hard Negative weighting
    if num_sent == 3:
        z3_weight = cls.model_args.hard_negative_weight
        weights = torch.tensor(
            [
                [0.0]*(cos_sim.size(-1) - z1_z3_cos.size(-1)) +
                [0.0]*i + [z3_weight] + [0.0]*(z1_z3_cos.size(-1) - i - 1)
                for i in range(z1_z3_cos.size(-1))
            ]
        ).to(cls.device)
        cos_sim = cos_sim + weights

    ce_loss = loss_fct(cos_sim, labels)  # cross-entropy loss

    # ---------------------------
    #  Orthonormality Penalty
    # ---------------------------
    loss = ce_loss
    if cls.model_args.ortho_loss_percent > 0.0:
        # Flatten (B, num_sent, hidden) => (B*num_sent, hidden)
        # for multi-GPU, z1 and z2 are already concatenated, so we rely on the full pooler_output
        # but let's keep it consistent by re-building them
        # If we are training with distribution, z1, z2, z3 might have changed shape.
        # Easiest is to do the same flattening approach we did for cos_sim
        if num_sent == 2:
            all_z = torch.cat([z1, z2], dim=0)
        elif num_sent == 3:
            all_z = torch.cat([z1, z2, z3], dim=0)
        else:
            all_z = pooler_output.view(-1, pooler_output.size(-1))

        # Compute orthogonality
        ortho_penalty = compute_orthogonality_loss(
            all_z, 
            margin=cls.model_args.ortho_margin,
            num_sent=num_sent
        )
        # Scale orthogonality by a fraction of the CE loss
        ce_val = ce_loss.detach()  # detach the scalar
        loss = ce_loss + (cls.model_args.ortho_loss_percent * ce_val * ortho_penalty)

    # MLM auxiliary
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, cls.config.vocab_size),
            mlm_labels.view(-1)
        )
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output)

    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


# =========================
# Sentence embedding forward
# =========================
def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    # Standard inference forward pass (no contrastive / orthogonality loss)
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ["avg_top2", "avg_first_last"] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


# =========================
# BERT for CL
# =========================
class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )


# =========================
# RoBERTa for CL
# =========================
class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )
