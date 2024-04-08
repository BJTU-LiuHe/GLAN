import argparse
from os.path import realpath, dirname
import torch
import os
import numpy as np
from torch_geometric.data import Data,Dataset
from torch_geometric.data import DataLoader

from GNBlock import _Model

torch.set_grad_enabled(False)

def sinkhorn_v1(mat):

    for _ in range(5):
        row_sum = torch.sum(mat, dim=2).unsqueeze(2)
        mat = mat/row_sum

        col_sum = torch.sum(mat, dim=1).unsqueeze(1)
        mat = mat/col_sum

    return mat

def sinkhorn_v1_np(mat):

    for _ in range(5):
        row_sum = np.expand_dims(np.sum(mat, axis=1), axis=1)
        mat = mat/ row_sum

        col_sum = np.expand_dims(np.sum(mat, axis = 0), axis = 0)
        mat = mat/col_sum

    return mat

def eval_acc(score, target, weight, args):
    """
    :param score: torch tensor, predicted score of shape [batch, H, W]
    :param target: torch tensor, ground truth value {0,1} of shape [batch, H, W]
    :param weight: torch tensor, weight for each batch for negative and positive examples of shape [batch, 2, 1, 1]
    :return: accuracy
    """
    acc = []
    predicted = torch.zeros_like(score)
    for b in range(score.size(0)):
        for h in range(score.size(1)):
            value, indice = score[b, h].max(0)
            if float(value) > args.threshold:
                predicted[b, h, int(indice)] = 1.0
        num_positive = float(target[b, :, :].sum())
        num_negative = float(target.size(1)*target.size(2) - num_positive)
        num_tp = float(((predicted[b, :, :] == target[b, :, :]) + (target[b, :, :] == 1.0)).eq(2).sum())
        num_tn = float(((predicted[b, :, :] == target[b, :, :]) + (target[b, :, :] == 0.0)).eq(2).sum())

        acc.append((num_tp * float(weight[b, 1, 0, 0]) + num_tn * float(weight[b, 0, 0, 0]))/
                   (num_positive * float(weight[b, 1, 0, 0]) + num_negative * float(weight[b, 0, 0, 0])))

    return np.mean(np.array(acc))

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

def prepare_Data(data_pth, train=True,dest_size=60):
    """
    :param data_pth: string that gives the data path
    :return: data list
    """
    data=list()
    d_pth=data_pth
    files = os.listdir(d_pth)

    for file in files:
        size=int(float(file.split("_")[0]))
        if size==dest_size:
            data.append(os.path.join(d_pth,file))
    print("dest_size:", dest_size,"\t",data_pth," has ",len(data)," data!")
    return data


class MatrixRealData(Dataset):
  'Characterizes a dataset for PyTorch'

  def __init__(self, data_path,dest_size, n_step = 2):
      super(MatrixRealData, self).__init__(data_path)#Initialization
      self.data_pth = data_path
      self.data = prepare_Data(data_path,False,dest_size)
      self.transor = Adap_TopK_Graph(step=n_step)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        data_file = self.data[index]
        data_ = torch.load(data_file)

        # Load data and get label
        matrix = data_["X"]
        matrix = torch.from_numpy(matrix.astype(np.float32))
        target = torch.from_numpy(data_["Y"].astype(np.int32))

        matrix_target = self.transor(matrix, target)

        return matrix,target,matrix_target,data_file

def greedyMap(pred_matrix):
    h,l=pred_matrix.shape
    result=np.zeros_like(pred_matrix)
    for hh in range(h):
        row,col=np.unravel_index(np.argmax(pred_matrix),pred_matrix.shape)
        result[row,col]=1
        pred_matrix[row,:]=0
        pred_matrix[:,col]=0

    return result

def statistic(model, tst_dataloader, record, size):

    for batch_idx ,cur_data in enumerate(tst_dataloader):

        matrix, target,Dt_target,file_name=cur_data

        shapes_info=Dt_target.kwargs[0]

        shape, k, idx_row, idx_knn,_ , num_edges = shapes_info
        target=Dt_target['y'].view(-1,shape,shape)

        pred = model(Dt_target)
        pred=pred.view(-1,1)
        tag_scores = torch.zeros((1, shape, shape)).cuda()
        tag_scores[:, idx_row, idx_knn] = pred

        matrix, target, tag_scores = matrix.squeeze(0), target.squeeze(0), tag_scores.squeeze(0)
        cost, target, pred_matrix = matrix.data.cpu().numpy(), target.data.cpu().numpy(), tag_scores.data.cpu().numpy()
        pred_matrix = sinkhorn_v1_np(pred_matrix + 1e-9)
        soft_cost = np.sum(cost * pred_matrix)
        ## greedy mapping ##
        h, l = pred_matrix.shape
        prediction = np.zeros_like(pred_matrix)
        for hh in range(h):
            row, col = np.unravel_index(np.argmax(pred_matrix), pred_matrix.shape)
            prediction[row, col] = 1
            pred_matrix[row, :] = 0
            pred_matrix[:, col] = 0
            if np.sum(pred_matrix) == 0:
                break

        rows = np.where(np.sum(prediction, axis=1) == 0)[0]
        cols = np.where(np.sum(prediction, axis=0) == 0)[0]

        if rows.shape[0] != 0:
            sub_cost = (cost[rows, :])[:, cols]
            sub_results = greedyMap(np.max(sub_cost) - sub_cost)

            idx_row, idx_col = np.where(sub_results == 1.0)

            prediction[rows[idx_row], cols[idx_col]] = 1

        if size not in record.keys():
            record[size] = dict()

            record[size]["acc"] = [100 * (np.sum((prediction == target)[target == 1]).astype(np.float) / size)]
            record[size]["score"] = [100 * (np.sum(target * cost) / np.sum(prediction * cost))]
            record[size]["soft_score"] = [100 * (np.sum(target * cost) / soft_cost)]
        else:
            record[size]["acc"].append(
                100 * (np.sum((prediction == target)[target == 1]).astype(np.float) / size))
            record[size]["score"].append(100 * (np.sum(target * cost) / np.sum(prediction * cost)))
            record[size]["soft_score"].append(100 * (np.sum(target * cost) / soft_cost))

if __name__ == '__main__':
    # parameters #

    print("Loading parameters...")
    curr_path = realpath(dirname(__file__))
    parser = argparse.ArgumentParser(description='PyTorch DeepMOT train')

    # data configs
    parser.add_argument('--data_root', dest='data_root',
                        default='dataset_10_150/test/',
                        help='dataset root path')
    parser.add_argument('--dest_size',dest='dest_size',default='60',type=int,help="size tested")
    parser.add_argument('--is_cuda', dest='is_cuda', default=True, type=bool, help='use GPU?')
    parser.add_argument('--model_path', dest='model_path',
                        default="",
                        help='pretrained model path')


    args = parser.parse_args()

    model_file = "best model file"
    model = _Model(layer_num = 5, edge_dim = 16, node_dim = 8)
    record = dict()
    model.load_state_dict(torch.load(model_file))

    model.cuda()
    model.eval()
    for size in range(10,151,10):
        tst_dataset = MatrixRealData(args.data_root, size, n_step = 2)
        tst_dataloader = DataLoader(tst_dataset, batch_size=args.batch_size, shuffle=True)
        statistic(model, tst_dataloader, record, size)

    total_acc, total_score, total_soft_score = [], [], []
    for k, v in record.items():
        size = k
        acc_list = v["acc"]
        scores = v["score"]
        soft_scores = v["soft_score"]

        acc_avg, acc_std = np.mean(acc_list), np.std(acc_list)
        score_avg, score_std = np.mean(scores), np.std(scores)
        soft_score_avg, soft_score_std = np.mean(soft_scores), np.std(soft_scores)

        total_acc.append(acc_avg)
        total_score.append(score_avg)
        total_soft_score.append(soft_score_avg)

        print(
            "size: {:<5d}, acc: {:.4f}, acc_std: {:.4f}, score: {:.4f}, score_std: {:.4f}, soft_score: {:.4f}, "
            "soft_std: {:.4f}".format(
                size, acc_avg, acc_std, score_avg, score_std, soft_score_avg, soft_score_std
            ))

    total_acc_avg, total_acc_std = np.mean(total_acc), np.std(total_acc)
    total_score_avg, total_score_std = np.mean(total_score), np.std(total_score)
    total_soft_score_avg, total_soft_score_std = np.mean(total_soft_score), np.std(total_soft_score)

    print(
        "Total acc: {:.4f}, acc_std: {:.4f}, score: {:.4f}, score_std: {:.4f}, soft_score: {:.4f}, soft_std: {:.4f}".format(
            total_acc_avg, total_acc_std, total_score_avg, total_score_std, total_soft_score_avg,
            total_soft_score_std
        ))