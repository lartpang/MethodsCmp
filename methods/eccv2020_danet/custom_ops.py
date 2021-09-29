import torch

from methods.eccv2020_danet.base_modules import PAFEM


def count_attention(ops: PAFEM, in_tensor, out_tensor):
    # 由于内部的卷积会被单独计算，所以这里不统计信息
    total_ops = 0

    if len(in_tensor) != 0:
        in_tensor = in_tensor[0]
        _, C, H, W = in_tensor.shape  # N = 1

        emb_dim = ops.down_dim
        num_tokens = H * W

        attn_ops = 0
        # torch.bmm(proj_query2, proj_key2)  B,HW,C and B,C,HW
        attn_ops += num_tokens * emb_dim * num_tokens
        # torch.bmm(proj_value2, attention2.permute(0, 2, 1)) B,C,HW and B,HW,HW
        attn_ops += emb_dim * num_tokens * num_tokens
        total_ops = attn_ops * 3
    else:
        print(ops)

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


custom_ops = {
    PAFEM: count_attention,
}
