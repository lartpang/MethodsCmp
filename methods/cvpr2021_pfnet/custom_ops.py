import torch

from .pfnet import CA_Block, SA_Block


def count_ca(ops: CA_Block, in_tensor, out_tensor):
    """
    m_batchsize, C, height, width = x.size()
    proj_query = x.view(m_batchsize, C, -1)
    proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
    energy = torch.bmm(proj_query, proj_key)
    attention = self.softmax(energy)
    proj_value = x.view(m_batchsize, C, -1)

    out = torch.bmm(attention, proj_value)
    out = out.view(m_batchsize, C, height, width)

    out = self.gamma * out + x
    """
    total_ops = 0

    if len(in_tensor) != 0:
        x = in_tensor[0]

        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        # energy = torch.bmm(proj_query, proj_key)
        total_ops += proj_query.shape[1] * proj_query.shape[2] * proj_key.shape[2]

        # attention = self.softmax(energy)

        proj_value = x.view(m_batchsize, C, -1)
        # out = torch.bmm(attention, proj_value)
        total_ops += proj_query.shape[1] * proj_key.shape[2] * proj_value.shape[2]
        # out = self.gamma * out + x
        total_ops += x[0].numel()
    else:
        print(ops)

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


def count_sa(ops: SA_Block, in_tensor, out_tensor):
    """
    m_batchsize, C, height, width = x.size()
    proj_query = (
    self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
    )
    proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
    energy = torch.bmm(proj_query, proj_key)
    attention = self.softmax(energy)
    proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

    out = torch.bmm(proj_value, attention.permute(0, 2, 1))
    out = out.view(m_batchsize, C, height, width)

    out = self.gamma * out + x
    """
    total_ops = 0

    if len(in_tensor) != 0:
        x = in_tensor[0]

        m_batchsize, C, height, width = x.size()
        num_tokens = height * width
        q_dim = ops.query_conv.weight.shape[1]
        k_dim = ops.key_conv.weight.shape[1]
        v_dim = ops.value_conv.weight.shape[1]

        # proj_query = (
        #     self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # )
        # proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        # proj_query B,ops.query_conv.weight.shape[0],H,W  -> B,HW,C
        # proj_key B,ops.key_conv.weight.shape[0],H,W  -> B,C,HW
        # energy = torch.bmm(proj_query, proj_key)
        assert q_dim == k_dim
        total_ops += num_tokens * q_dim * num_tokens

        # attention = self.softmax(energy)

        # proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        # proj_value B,ops.value_conv.weight.shape[0],H,W  -> B,C,HW
        # out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        total_ops += num_tokens * num_tokens * v_dim
        # out = out.view(m_batchsize, C, height, width)

        # out = self.gamma * out + x
        total_ops += x[0].numel()
    else:
        print(ops)

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


custom_ops = {
    CA_Block: count_ca,
    SA_Block: count_sa,
}
