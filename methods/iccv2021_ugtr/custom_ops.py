import torch

from .pmm import PMMs
from .ugtr import GetPE


def count_get_pe(ops: GetPE, in_tensor, out_tensor):
    total_ops = 0

    if len(in_tensor) != 0:
        position_encoding, z_ = in_tensor
        b, c, n = position_encoding.flatten(2).shape

        # position_encoding = torch.bmm(position_encoding.flatten(2), z_).unsqueeze(2)
        assert n == z_.shape[1]
        total_ops += c * n * z_.shape[2]
    else:
        print(ops)

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


def count_ops_of_pmms(ops: PMMs, in_tensor, out_tensor):
    """ """

    total_ops = 0

    if len(in_tensor) != 0:
        support_feature = in_tensor[0]
        b, c, h, w = support_feature.shape
        n = h * w
        _, _, mu_n = ops.mu.shape

        # self.generate_prototype(support_feature)
        #   -> self.get_prototype(z)
        #   -> self.EM(x) x(b,c,hw)
        # with torch.no_grad():
        #     for i in range(self.stage_num):
        num_stages = ops.stage_num
        #         # E STEP:
        #         z = self.Kernel(x, mu)
        # def Kernel(self, x, mu):
        #     x_t = x.permute(0, 2, 1)  # b * n * c
        #     z = self.kappa * torch.bmm(x_t, mu)  # b * n * k
        num_ops_in_em = n * c * mu_n
        #         # M STEP:
        #         z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
        #         mu = torch.bmm(x, z_)  # b * c * k
        #         mu = self._l2norm(mu, dim=1)
        num_ops_in_em += c * n * mu_n
        num_ops_in_em *= num_stages
        num_ops_in_generate_prototype = 2 * num_ops_in_em

        total_ops += num_ops_in_generate_prototype
    else:
        print(ops)

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


custom_ops = {
    GetPE: count_get_pe,
    PMMs: count_ops_of_pmms,
}
