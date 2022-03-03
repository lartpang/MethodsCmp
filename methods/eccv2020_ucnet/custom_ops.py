import torch

from .ucnet import CAM_Module


def count_cam_module(ops: CAM_Module, in_tensor, out_tensor):
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


custom_ops = {
    CAM_Module: count_cam_module,
}
