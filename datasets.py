import math
import torch
import random
from torch_geometric.data import Data,Dataset
import os
import numpy as np


class Adap_TopK_Graph(torch.nn.Module):
    def __init__(self,step):
        super(Adap_TopK_Graph, self).__init__()
        self.step = step
    def knn_idx(self,distance_matrix,k):
        _, index = distance_matrix.sort(dim=1)

        return index[:, :k]
    def build_graph(self, distance_matrix,target_matrix):
        row_size,col_size=distance_matrix.shape
        k = min(row_size, 10 + self.step * int(row_size / 10))

        idx_knn=self.knn_idx(distance_matrix,k).reshape(-1,1)
        idx_row=torch.range(0,row_size-1).view(-1,1).repeat(1,k).reshape(-1,1)
        idx_row=idx_row.type(torch.int64)

        edge_index=torch.cat((idx_row,idx_knn+row_size),dim=1).type(torch.long)
        edge_attr=distance_matrix[idx_row,idx_knn]
        ground_truth=target_matrix[idx_row,idx_knn]

        edge_attr=torch.cat((edge_attr,edge_attr),dim=1).view(-1,1)
        edge_index=torch.cat((edge_index,edge_index[:,1].unsqueeze(1),edge_index[:,0].unsqueeze(1)),dim=1)
        edge_index=edge_index.view(-1,2).permute(1,0)

        return edge_index,edge_attr,idx_row,idx_knn,k

    def forward(self,distance_matrix,target):
        gt_cost = torch.sum(distance_matrix * target)
        edge_index,edge_attr,idx_row,idx_knn,k=self.build_graph(distance_matrix,target)

        if torch.cuda.is_available():
            gt_cost = gt_cost.cuda()

        if torch.cuda.is_available():
            data = Data(x=torch.zeros((sum(distance_matrix.shape), 8)).cuda(), edge_index=edge_index.cuda(),
                        edge_attr=edge_attr.cuda(), y=target.view(-1, 1).cuda(),
                        kwargs=[distance_matrix.shape[0], k, idx_row, idx_knn, gt_cost.cuda(),edge_attr.shape[0]],
                        cost_vec = distance_matrix.view(-1,1).cuda())
        else:
            data=Data(x=torch.zeros((sum(distance_matrix.shape),8)),edge_index=edge_index,
                  edge_attr=edge_attr,y=target.view(-1,1),
                      kwargs=[distance_matrix.shape[0],k,idx_row,idx_knn, gt_cost,edge_attr.shape[0]],
                      cost_vec = distance_matrix.view(-1,1))

        return data

def check_data(file):

    data = np.load(file)
    h, l = data.shape
    if np.sum(data) != min(h, l):
        return False
    else:
        return True

def prepare_Data(data_pth, train=True):
    """
    :param data_pth: string that gives the data path
    :return: data list
    """
    data=list()
    if train:
        d_pth = os.path.join(data_pth, 'train/')
    else:
        d_pth = os.path.join(data_pth, 'eval/')
    files = os.listdir(d_pth)

    for file in files:
        data.append(os.path.join(d_pth, file))

    return data

class GraphData_AdpK(Dataset):

    def __init__(self, data_path, train=True, step =0):
        super(GraphData_AdpK, self).__init__(data_path)
        '''Initialization'''
        self.data_pth = data_path
        self.data_list = prepare_Data(data_path, train)
        self.transor=Adap_TopK_Graph(step = step)
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_list)

    def __getitem__(self, index):
        'Generates one sample of data'
        data_file=self.data_list[index]
        data_=torch.load(data_file)

        # Load data and get label
        matrix = data_["X"]
        matrix = torch.from_numpy(matrix.astype(np.float32))
        target = torch.from_numpy(data_["Y"].astype(np.int32))

        matrix_target=self.transor(matrix,target)

        return matrix_target