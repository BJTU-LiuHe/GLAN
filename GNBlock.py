import torch
import time
from torch.nn import Sequential as Seq, Linear as Lin,LayerNorm, ReLU,Sigmoid,Tanh,LeakyReLU,BatchNorm1d
from torch_scatter import scatter_mean,scatter_max,scatter_min
from myMetaLayer import MetaLayer

class GlobalNorm(torch.nn.Module):
    def __init__(self):
        super(GlobalNorm, self).__init__()
    def forward(self, data):
        [h,l]=data.shape
        mean_=torch.mean(data,dim=0).detach()
        var_=torch.var(data,dim=0).detach()

        return (data-mean_)/torch.sqrt(var_+0.00001)

class EdgeModel(torch.nn.Module):
    def __init__(self,dim_edge,dim_node):
        super(EdgeModel, self).__init__()
        self.edge_mlp1 = Seq(Lin(dim_edge+dim_node*2, dim_edge*2),ReLU(),GlobalNorm(),Lin(dim_edge*2, dim_edge),ReLU(),GlobalNorm())
        self.edge_mlp2 = Seq(Lin(dim_edge * 2, dim_edge * 2), ReLU(), GlobalNorm(), Lin(dim_edge * 2, dim_edge),ReLU(), GlobalNorm())

    def forward(self, src, dest, edge_attr, u, batch,wEdgeC=None,wNodeC=None):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        # src,dest=src*wNodeC,dest*wNodeC
        # edge_attr_=edge_attr*wEdgeC
        if u != None:
            out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        else:
            out = torch.cat([src, dest, edge_attr], 1)
        out=self.edge_mlp1(out)
        return self.edge_mlp2(torch.cat((out,edge_attr),dim=1))

class NodeModel(torch.nn.Module):
    def __init__(self,dim_edge,dim_node):
        super(NodeModel, self).__init__()

        self.wAggregation=Seq(Lin(dim_node*2,dim_node),ReLU(),GlobalNorm(), Lin(dim_node, 1),Sigmoid())
        self.node_mlp_1 = Seq(Lin(dim_edge+dim_node,dim_edge+dim_node),ReLU(),GlobalNorm(), Lin(dim_edge+dim_node, dim_node),ReLU(),GlobalNorm())
        self.node_mlp_2 = Seq(Lin(dim_node*2, dim_node*2),ReLU(),GlobalNorm(), Lin(dim_node*2,dim_node),ReLU(),GlobalNorm())

    def forward(self, x, edge_index, edge_attr, u, batch,wEdgeC=None,wNodeC=None):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        wAgg=self.wAggregation(torch.cat((x[col],x[row]),dim=1))
        out = torch.cat([x[row]*wAgg, edge_attr], dim=1)
        # out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out=scatter_mean(out, col, dim=0, dim_size=x.size(0))

        if u != None:
            out = torch.cat([x, out, u[batch]], dim=1)
        else:
            out = torch.cat([x, out], dim=1)

        return self.node_mlp_2(out)

class GlobalModel(torch.nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.global_mlp = Seq(Lin(4, 4), ReLU(), Lin(4, 4))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        return self.global_mlp(out)

class _Model(torch.nn.Module):
    def __init__(self,layer_num,edge_dim,node_dim):
        super(_Model,self).__init__()
        self.layer_num=layer_num
        self.dim_node=node_dim
        self.encoder_node=Seq(
            Lin(3,8),ReLU(),GlobalNorm(),Lin(8,8),ReLU(),GlobalNorm()
        )
        self.encoder_edge = Seq(
            Lin(1, int(edge_dim/2)),ReLU(),GlobalNorm(), Lin(int(edge_dim/2), edge_dim),ReLU(),GlobalNorm()
        )
        self.core_layers=torch.nn.Sequential()
        self.wEdgeC_MLP= Seq(Lin(edge_dim*3,edge_dim*2),LeakyReLU(),LayerNorm(edge_dim*2),
                             Lin(edge_dim*2,edge_dim),LeakyReLU(),LayerNorm(edge_dim))
        self.wNodeC_MLP=Seq(Lin(node_dim*3,node_dim*2),LeakyReLU(),LayerNorm(node_dim*2),
                             Lin(node_dim*2,node_dim),LeakyReLU(),LayerNorm(node_dim))

        for idx in range(self.layer_num):
            coreName="core_"+str(idx)
            self.core_layers.add_module(coreName,MetaLayer(EdgeModel(edge_dim,node_dim), NodeModel(edge_dim,node_dim)))

        self.decoder_node = Seq(
            Lin(8, 4),ReLU(),GlobalNorm(), Lin(4, 1), Sigmoid()
        )

        self.decoder_edge= Seq(
            Lin(edge_dim, int(edge_dim/2)), ReLU(),GlobalNorm(), Lin(int(edge_dim/2), 1), Sigmoid()
        )

        self.softmax0=torch.nn.Softmax(dim=0)
        self.softmax1=torch.nn.Softmax(dim=1)

    def cal_edge_batch(self, data):

        num_graphs = data.num_graphs
        batch_edges = []
        for i in range(num_graphs):
            num_edges = data.kwargs[i][-1]
            item = torch.full((num_edges,), i, dtype=torch.long)
            batch_edges.append(item)

        result = torch.cat(batch_edges, dim=0)
        if torch.cuda.is_available():
            result = result.cuda()

        return result

    def forward(self, data):

        x_, edge_index, edge_attr=data.x,data.edge_index,data.edge_attr
        batch_node=data.batch.cuda()
        batch_edge= self.cal_edge_batch(data)
        num_node=x_.shape[0]
        x=torch.zeros((num_node,self.dim_node)).cuda()
        time_start = time.time()
        edge_attr=self.encoder_edge(edge_attr)
        for idx in range(self.layer_num):
            core=self.core_layers[idx]

            node_mean = scatter_mean(x, batch_node, dim=0)
            node_max = scatter_max(x, batch_node, dim=0)[0]
            node_min = scatter_min(x, batch_node, dim=0)[0]

            edge_mean = scatter_mean(edge_attr, batch_edge, dim=0)
            edge_max = scatter_max(edge_attr, batch_edge, dim=0)[0]
            edge_min = scatter_min(edge_attr, batch_edge, dim=0)[0]

            edge_info = torch.cat((edge_mean, edge_max, edge_min), dim=1)
            node_info = torch.cat((node_mean, node_max, node_min), dim=1)

            wEdgeC = self.wEdgeC_MLP(edge_info)
            wNodeC = self.wNodeC_MLP(node_info)

            x = x * wNodeC[batch_node]
            edge_attr = edge_attr * wEdgeC[batch_edge]
            
            x, edge_attr, _=core(x,edge_index,edge_attr=edge_attr,wEdgeC=None,wNodeC=None)


        edge_attr=self.decoder_edge(edge_attr)
        res = torch.mean(edge_attr.view(-1, 2), dim=1).unsqueeze(1)

        return res

    def data_normlize(self,data):
        return -1+2*data