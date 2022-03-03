import torch
from methods.cvpr2021_mgl.basicnet import (
    CascadeGCNet,
    GraphConvNet,
    MutualModule0,
    MutualModule1,
)


def count_graphconvnet(ops: GraphConvNet, in_tensor, out_tensor):
    """
    x_t = x.permute(0, 2, 1).contiguous()  # b x k x c
    support = torch.matmul(x_t, self.weight)  # b x k x c

    adj = torch.softmax(adj, dim=2)
    output = (torch.matmul(adj, support)).permute(0, 2, 1).contiguous()  # b x c x k
    """

    total_ops = 0

    if len(in_tensor) != 0:
        x = in_tensor[0]
        adj = in_tensor[1]

        # torch.matmul(x_t, self.weight)  # b x k x c
        total_ops += x.shape[1] * x.shape[2] * ops.out_features
        # (torch.matmul(adj, support)).permute(0, 2, 1).contiguous()  # b x c x k
        total_ops += adj.shape[1] * adj.shape[2] * ops.out_features

        if ops.bias is not None:
            # Cout in_tensor 1
            total_ops += ops.bias.nelement()
    else:
        print(ops)

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


def count_cascadegcnet(ops: CascadeGCNet, in_tensor, out_tensor):
    """
    for gcn in self.gcns:
        x_t = x.permute(0, 2, 1).contiguous()  # b x k x c
        x = gcn(x, adj=torch.matmul(x_t, x))  # b x c x k
    """

    total_ops = 0

    if len(in_tensor) != 0:
        length_of_loops = len(ops.gcns)
        x = in_tensor[0]

        # torch.matmul(x_t, x)
        total_ops += (x.shape[2] * x.shape[1] * x.shape[2]) * length_of_loops
    else:
        print(ops)

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


def count_mutualmodule0(ops: MutualModule0, in_tensor, out_tensor):
    """
    # graph0: edge, graph1/2: region, assign:edge
    def forward(self, edge_graph, region_graph1, region_graph2, assign):
        m = self.corr_matrix(edge_graph, region_graph1, region_graph2)
        edge_graph = edge_graph + m

        edge_graph = self.gcn(edge_graph)
        edge_x = edge_graph.bmm(assign)  # reprojection
        edge_x = self.conv(edge_x.unsqueeze(3)).squeeze(3)
        return edge_x

    def corr_matrix(self, edge, region1, region2):
        assign = edge.permute(0, 2, 1).contiguous().bmm(region1)
        assign = F.softmax(assign, dim=-1)  # normalize region-node
        m = assign.bmm(region2.permute(0, 2, 1).contiguous())
        m = m.permute(0, 2, 1).contiguous()
        return m
    """

    total_ops = 0

    if len(in_tensor) != 0:
        edge_graph, region_graph1, region_graph2, assign = in_tensor

        # m = self.corr_matrix(edge_graph, region_graph1, region_graph2)
        # ==>>
        # assign = edge.permute(0, 2, 1).contiguous().bmm(region1)
        # assign = F.softmax(assign, dim=-1)  # normalize region-node
        # m = assign.bmm(region2.permute(0, 2, 1).contiguous())
        assert edge_graph.shape[1] == region_graph1.shape[1]
        total_ops += edge_graph.shape[2] * edge_graph.shape[1] * region_graph1.shape[2]
        assert region_graph1.shape[2] == region_graph2.shape[2]
        total_ops += (
            region_graph1.shape[1] * region_graph2.shape[2] * region_graph2.shape[1]
        )

        # edge_graph = self.gcn(edge_graph)
        # edge_x = edge_graph.bmm(assign)  # reprojection
        assert edge_graph.shape[2] == assign.shape[1]
        total_ops += edge_graph.shape[1] * assign.shape[1] * assign.shape[2]
    else:
        print(ops)

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


def count_mutualmodule1(ops: MutualModule1, in_tensor, out_tensor):
    """
    def forward(self, region_x, region_graph, assign, edge_x):
        b, c, h, w = edge_x.shape
        edge = self.pred0(edge_x)

        region_graph = self.gcn(region_graph)
        n_region_x = region_graph.bmm(assign)
        n_region_x = self.conv0(n_region_x.view(region_x.size()))
        region_x = region_x + n_region_x  # raw-feature with residual
        region_x = region_x + edge_x
        region_x = self.conv1(region_x)

        # enhance
        region_x = self.ecg(region_x, edge)
        region = self.pred1_(region_x)
        return region_x, edge, region
    """

    total_ops = 0

    if len(in_tensor) != 0:
        region_x, region_graph, assign, edge_x = in_tensor

        # n_region_x = region_graph.bmm(assign)
        assert region_graph.shape[2] == assign.shape[1]
        total_ops += region_graph.shape[1] * assign.shape[1] * assign.shape[2]
    else:
        print(ops)

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


custom_ops = {
    GraphConvNet: count_graphconvnet,
    CascadeGCNet: count_cascadegcnet,
    MutualModule0: count_mutualmodule0,
    MutualModule1: count_mutualmodule1,
}
