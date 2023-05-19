# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

__all__ = [k for k in globals().keys() if not k.startswith("_")]


def _assert_valid_inputs(input_embeddings, query_embeddings):
    assert (
        input_embeddings.dim() == 2
        and query_embeddings.dim() == 2
        and input_embeddings.size(1) == query_embeddings.size(1)
    ), (input_embeddings.shape, query_embeddings.shape)


def _product_attr(vision, text, alter):
    """
    Args:
        vision: N x D
        text: M x D
        alter: N x M, to replace results in some cases, see details in Returns
    Returns: N x M.
        For (n, m) element, set J_m = {j : text[m, j] == 1}.
        - if |J_m| > 0, it equals to (prod_{j in J_m} vision[n, j])**(1/|J_m|)
        - if |J_m| == 0, it equals to alter[n, m]
    """
    vision = vision.unsqueeze(1)  # N x 1 x D
    text = text.unsqueeze(0)  # 1 x M x D

    # (m) element -> |{j : text[m, j] == 1}|, i.e. |J_m|
    num_attr = text.sum(-1)  # 1 x M

    # (n, m, j) element -> vision[n, j] if text[m, j] == 1 else 1
    queried_attr = vision * text  # N x M x D
    queried_attr = queried_attr.masked_fill(text == 0, 1)
    # (n, m) element -> (prod_{j in J_m} vision[n, j])**(1/|J_m|) if |J_m|>0 else 1
    queried_attr = torch.float_power(
        queried_attr.prod(dim=2),  # N x M
        1 / torch.max(num_attr, torch.ones_like(num_attr)),  # 1 x M
    ).float()

    no_attr_queries = num_attr.squeeze(0) == 0  # M
    queried_attr[:, no_attr_queries] = alter[:, no_attr_queries]
    return queried_attr


def obj_with_attributes(input_embeddings, query_embeddings, n_obj, n_part, n_attr):
    """
    For a query involves an object and Q object attributes:
        - if Q > 0, return sqrt( obj_sc * (prod of queried_attr)**(1/Q) )
        - if Q == 0, return obj_sc
    Args:
        input_embeddings (tensor): N x D, where N is number of boxes
        query_embeddings (tensor): M x D, where M is number of queries
        n_obj, n_part, n_attr (int)
    Returns:
        similarity (tensor): N x M
    """
    obj_score = input_embeddings[:, :n_obj] @ query_embeddings[:, :n_obj].T  # N x M

    # -> obj_sc * [ (prod of queried_attr)**(1/Q) if Q > 0 else obj_sc ]
    obj_score *= _product_attr(
        input_embeddings[:, n_obj + n_part : n_obj + n_part + n_attr],
        query_embeddings[:, n_obj + n_part : n_obj + n_part + n_attr],
        obj_score,
    )
    return torch.sqrt(obj_score)


def parts_with_attributes(
    input_embeddings, query_embeddings, fused, n_obj, n_part, n_attr
):
    """
    For a query involves K parts and Q_k part attributes:
        - if K > 0,
            - Each part
                - if Q_k > 0, sqrt( part_k_sc * (prod of queried_attr) ** (1/Q_k) )
                - if Q_k == 0, part_k_sc
            - Sum over all parts
        - if K == 0,
            - returns 1
    Args:
        input_embeddings (tensor): N x D, where N is number of boxes
        query_embeddings (tensor): M x D, where M is number of queries
        fused (bool): This flag will be true if part_attr is a product of
            partness scores and attribute classification scores.
        n_obj, n_part, n_attr (int)
    Returns:
        similarity (tensor): N x M
    """
    vision_partness = input_embeddings[:, n_obj : n_obj + n_part]  # N x #part
    text_partness = query_embeddings[:, n_obj : n_obj + n_part]  # M x #part
    M = text_partness.size(0)

    # Each part
    part_score_list = [
        torch.sqrt(
            _product_attr(
                # - if Q_k > 0,  (prod of part_k_sc*queried_attr)**(1/Q_k)
                #               = part_k_sc * (prod of queried_attr)**(1/Q_k)
                # - if Q_k == 0, square(part_k_sc)
                vision_part_attr,  # N x #attr
                text_part_attr,  # M x #attr
                torch.square(partness.unsqueeze(1).repeat(1, M)),  # N x M
            )
        )
        if fused
        else torch.sqrt(
            partness.unsqueeze(1)  # N x 1
            * _product_attr(
                # - if Q_k > 0,  (prod of queried_attr)**(1/Q_k)
                # - if Q_k == 0, part_k_sc
                vision_part_attr,  # N x #attr
                text_part_attr,  # M x #attr
                partness.unsqueeze(1).repeat(1, M),  # N x M
            )
        )
        for vision_part_attr, text_part_attr, partness in zip(
            input_embeddings[:, n_obj + n_part + n_attr :].split(n_attr, dim=1),
            query_embeddings[:, n_obj + n_part + n_attr :].split(n_attr, dim=1),
            vision_partness.T,
        )
    ]  # [N x M] * #part
    # Average over all parts
    part_score = torch.stack(part_score_list, dim=2)  # N x M x #part
    part_score *= text_partness.unsqueeze(0)
    part_score = part_score.sum(2)  # N x M
    n_text_parts = text_partness.sum(1)  # M
    part_score[:, n_text_parts > 0] /= n_text_parts[n_text_parts > 0]

    # if no parts, return 1
    part_score[:, n_text_parts == 0] = 1

    return part_score


def compute_similarity_matrix(
    input_embeddings: torch.tensor,
    query_embeddings: torch.tensor,
    n_obj: int,
    n_part: int,
    n_attr: int,
    *,
    part_attr_fused: bool = False,
):
    """
    Args:
        input_embeddings (tensor): N x D, where N is number of boxes
        query_embeddings (tensor): M x D, where M is number of queries
        n_obj, n_part, n_attr (int)
        part_attr_fused (bool): This flag will be true if part_attr is a product of
            partness scores and attribute classification scores.
    Returns:
        similarity (tensor): N x M
    """
    _assert_valid_inputs(input_embeddings, query_embeddings)
    assert input_embeddings.size(1) == n_obj + n_part + n_attr + n_part * n_attr, (
        input_embeddings.shape,
        n_obj,
        n_part,
        n_attr,
    )
    assert input_embeddings.min() >= 0 and input_embeddings.max() <= 1

    obj_score = obj_with_attributes(
        input_embeddings, query_embeddings, n_obj, n_part, n_attr
    )
    part_score = parts_with_attributes(
        input_embeddings, query_embeddings, part_attr_fused, n_obj, n_part, n_attr
    )

    return obj_score * part_score
