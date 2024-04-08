import torch.optim as optim
from datasets import GraphData_AdpK
import torch
import torch.nn as nn
import numpy as np
from utils import getLayers
from losses import binaryloss,sum_loss,sparse_loss
import os
import argparse
from os.path import realpath, dirname
from torch_geometric.data import DataLoader
from GNBlock import _Model

def greddy_map(tag_scores, cost):
    pred_matrix = tag_scores.squeeze(0).data.cpu().numpy()
    cost = cost.squeeze(0).data.cpu().numpy()
    h, l = pred_matrix.shape
    result = np.zeros_like(pred_matrix)
    for hh in range(h):
        row, col = np.unravel_index(np.argmax(pred_matrix), pred_matrix.shape)
        result[row, col] = 1
        pred_matrix[row, :] = 0
        pred_matrix[:, col] = 0

        if np.sum(pred_matrix) == 0:
            break

    rows = np.where(np.sum(result, axis=1) == 0)[0]
    cols = np.where(np.sum(result, axis=0) == 0)[0]

    if len(rows) != 0:
        sub_cost = (cost[rows, :])[:, cols]
        sub_results = np.zeros_like(sub_cost)
        sub_pred = 1 - sub_cost

        for hh in range(h):
            row, col = np.unravel_index(np.argmax(sub_pred), sub_pred.shape)
            sub_results[row, col] = 1
            sub_pred[row, :] = 0
            sub_pred[:, col] = 0

        idx_row, idx_col = np.where(sub_results == 1.0)

        result[rows[idx_row], cols[idx_col]] = 1

    predicted = torch.from_numpy(result).unsqueeze(0).cuda()
    return predicted

def sinkhorn_v1(mat):

    for _ in range(5):
        row_sum = torch.sum(mat, dim=2).unsqueeze(2)
        mat = mat/row_sum

        col_sum = torch.sum(mat, dim=1).unsqueeze(1)
        mat = mat/col_sum

    return mat

def main(args,dim_node,dim_edge,layer_num, step):
    torch.backends.cudnn.deterministic = True
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    old_lr = args.learning_rate

    model=_Model(layer_num,dim_edge,dim_node)
    model_layers = getLayers(model)
    for m in model_layers:  # init parameters
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data, gain=1)

    model = model.train()
    if args.is_cuda:
        model = model.cuda()

    # optimizer #
    assert args.optimizer in ['SGD', 'RMSprop']
    if args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=old_lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=old_lr, momentum=0.9)

    # data loaders #
    train_dataset=GraphData_AdpK(args.data_path,train=True, step = step)
    val_dataset = GraphData_AdpK(args.data_path, train=False, step = step)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    print('val length in #batches: ', len(val_dataloader))
    print('train length in #batches: ', len(train_dataloader))

    iteration = 0
    rate = 0.0
    train_loss, train_diffRate = list(), list()
    best_acc = 0.0
    for epoch in range(args.epochs):
        for batch_idx, Dt_target in enumerate(train_dataloader):
            model = model.train()
            # input to model
            tag_scores = model(Dt_target)
            targets = Dt_target["y"]
            shapes_info = Dt_target["kwargs"]
            cost_vec = Dt_target["cost_vec"]

            loss, last_idx_pred, last_idx_target = 0, 0,0

            for idx_ in range(len(shapes_info)):
                shape,k,idx_row,idx_knn, gt_cost, num_edges = shapes_info[idx_]
                data_size = shape*k
                target_size=shape*shape

                target_square = targets[last_idx_target:target_size + last_idx_target].view(-1, shape, shape)
                target_choosen = target_square[:, idx_row, idx_knn].view(-1, 1)
                cost_mat = cost_vec[last_idx_target:target_size + last_idx_target].view(-1, shape, shape)
                pred_matrix=tag_scores[last_idx_pred:data_size+last_idx_pred].view(-1,1)
                tag_score = torch.zeros((1, shape, shape)).cuda() + 1e-9
                tag_score [:,idx_row,idx_knn]=pred_matrix
                tag_score = sinkhorn_v1(tag_score)
                pred_matrix = tag_score[:, idx_row, idx_knn].view(-1, 1)

                weight[:, 0, 0, 0] = 1/k
                weight[:, 1, 0, 0] = (k-1)/k
                if args.is_cuda:
                    weight = weight.cuda()

                if args.type_loss == "binary":
                    loss += (binaryloss(pred_matrix, target_choosen.float(), weights=weight) +
                         rate * sum_loss(tag_score.squeeze(0), target_square.float().squeeze(0)) +
                         rate * sparse_loss(tag_score.squeeze(0), target_square.float().squeeze(0)))
                else:
                    loss += (torch.pow(torch.sum(cost_mat*tag_score)-gt_cost, 2) +
                         rate * sum_loss(tag_score.squeeze(0), target_square.float().squeeze(0)) +
                         rate * sparse_loss(tag_score.squeeze(0), target_square.float().squeeze(0)))

                loss_show = (10.0 * binaryloss(pred_matrix, target_choosen.float(), weights=weight))

                train_loss.append(float(loss_show.item()))
                train_diffRate.append(float((100 * gt_cost/torch.sum(cost_mat * tag_score) ).item()))

                last_idx_pred += data_size
                last_idx_target+=target_size


            loss = loss / len(shapes_info)
            # clean gradients & back propagation
            model.zero_grad()
            loss.backward()
            optimizer.step()


            if (iteration+1) % args.print_train == 0:
                print('Epoch: [{}][{}/{}]\tLoss {:.4f}\tDiffRate {:.4f}\t'.format(epoch, iteration % len(train_dataloader),
                                                                 len(train_dataloader), np.mean(np.array(train_loss)),
                                                                                  np.mean(train_diffRate)),
                      end="")
                train_loss.clear()
                train_diffRate.clear()


            if (iteration+1) % args.print_test == 0:

                model = model.eval()
                test_loss = []
                acc = []
                diff_list, diff_discrete_list = [], []

                for test_num, Dt_target in enumerate(val_dataloader):
                    # input to model
                    tag_scores = model(Dt_target)

                    targets = Dt_target["y"]
                    shapes_info = Dt_target["kwargs"]
                    cost_vec = Dt_target["cost_vec"]

                    loss, last_idx_pred, last_idx_target = 0, 0, 0
                    for idx_ in range(len(shapes_info)):
                        shape,k,idx_row,idx_knn, gt_cost, num_edges = shapes_info[idx_]
                        data_size = shape * k
                        target_size = shape * shape

                        target_square = targets[last_idx_target:target_size + last_idx_target].view(-1, shape, shape)
                        target_choosen = target_square[:, idx_row, idx_knn].view(-1, 1)
                        cost_mat = cost_vec[last_idx_target:target_size + last_idx_target].view(-1, shape, shape)
                        pred_matrix = tag_scores[last_idx_pred:data_size + last_idx_pred].view(-1, 1)
                        tag_score = torch.zeros((1, shape, shape)).cuda() + 1e-9
                        tag_score[:, idx_row, idx_knn] = pred_matrix 
                        tag_score = sinkhorn_v1(tag_score)

                        weight[:, 0, 0, 0] = 1/k
                        weight[:, 1, 0, 0] = (k-1)/k
                        if args.is_cuda:
                            weight = weight.cuda()

                        if args.type_loss =="binary":
                            loss = binaryloss(pred_matrix, target_choosen.float(), weights=weight)
                        else:
                            loss = torch.pow(torch.sum(cost_mat*tag_score)-gt_cost, 2)

                        test_loss.append(float(loss.item()))

                        tag_score_gm = greddy_map(tag_score, cost_mat)
                        cost_discrete_cur = torch.sum(cost_mat * tag_score_gm).data.cpu().numpy()
                        target_square, tag_score_gm = target_square.squeeze(0), tag_score_gm.squeeze(0)
                        target_square, tag_score_gm = target_square.data.cpu().numpy(), tag_score_gm.data.cpu().numpy()
                        gt_cost = gt_cost.data.cpu().numpy()
                        acc_cur = 100 * (np.sum((tag_score_gm == target_square)[target_square == 1]).astype(np.float) / shape)
                        cost_cur = torch.sum(cost_mat*tag_score).data.cpu().numpy()

                        acc.append(acc_cur)
                        diff_list.append(100.0 * gt_cost / cost_cur)
                        diff_discrete_list.append(100 * gt_cost / cost_discrete_cur)

                        last_idx_pred += data_size
                        last_idx_target += target_size


                print('Epoch: [{}][{}/{}]\tLoss {:.4f}\t Accuracy {:.2f} %, \t diff_rate {:.2f} %, \t diff_discrete_rate '
                      '{:.2f} %'.format(epoch, iteration % len(train_dataloader),len(train_dataloader),np.mean(test_loss),
                                        np.mean(acc),np.mean(diff_list), np.mean(diff_discrete_list)))

                if best_acc < np.mean(acc):
                    best_acc = np.mean(acc) + 0.0
                    torch.save(model.state_dict(), os.path.join(args.save_path, "best_model.pth"))

                acc.clear()
                diff_list.clear()
                test_loss.clear()
                diff_discrete_list.clear()
            iteration += 1

        if (epoch + 1) % 5 ==0:
            for param_group in optimizer.param_groups:
                new_lr = param_group['lr'] * 0.95
                param_group['lr'] = max(new_lr, 0.00001)
                print('lr:', param_group['lr'])

        rate += 0.01


if __name__ == '__main__':
    # parameters #

    print("Loading parameters...")
    curr_path = realpath(dirname(__file__))
    parser = argparse.ArgumentParser(description='PyTorch DeepMOT train')

    # data configs
    parser.add_argument('--data_path', dest='data_path', default='G:/LAP_dataset/dataset_10_150/',
                        help='dataset root path')
    parser.add_argument('--is_cuda', default=True,action='store_true', help="use GPU if set.")

    # train configs
    parser.add_argument('-b', dest='batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', dest='epochs', default=10, type=int, help='number of training epochs')
    parser.add_argument('--print_test', dest='print_test', default=200, type=int, help='test frequency')
    parser.add_argument('--print_train', dest='print_train', default=100, type=int, help='training print frequency')
    parser.add_argument('--logs_path', dest='logs_path', default=os.path.join(curr_path, 'log/'), help='log files path')
    parser.add_argument('--save_path', dest='save_path', default=os.path.join(curr_path, 'output/'), help='save path')
    parser.add_argument('--optimizer',dest='optimizer',default='RMSprop',type=str,help='the optimizer used')
    parser.add_argument('--lr',dest='learning_rate',default=0.003,type=float,help='the initial learning rate')
    parser.add_argument('--type_loss', dest='binary', default="", type=str, help='the type of loss function')
    args = parser.parse_args()

    step = 2
    main(args, dim_node=8, dim_edge=16, layer_num=5, step = 2)
        

