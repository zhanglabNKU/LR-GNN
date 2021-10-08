import torch
import torch.nn as nn
import time
import argparse
import LR_GNN
import data_loader
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc


def get_auc(test_label, test_prob):
    auc = roc_auc_score(test_label, test_prob)
    return auc

def get_aupr(test_label, test_prob):
    precision, recall, _ = precision_recall_curve(test_label, test_prob)
    return auc(recall, precision)

def metric(pred, true):
    pred = pred.reshape(-1)
    true = true.reshape(-1)
    pred = pred.ge(0.5).int()
    pred = pred.detach().numpy()
    recall = recall_score(true, pred, pos_label=1)
    f1 = f1_score(true, pred, pos_label=1)
    return recall,f1

def Com_acc(pred,true):
    pred = pred.reshape(-1)
    true = true.reshape(-1)
    result = pred.ge(0.5).float() == true
    acc = result.float().mean()
    return acc

def cvMetric(accCV, recallCV, f1CV, aucCV, auprCV):
    fold_acc = []
    fold_recall = []
    fold_f1 = []
    fold_auc = []
    fold_aupr = []
    for k in range(5):
        fold_acc.append(torch.tensor(accCV[k]).reshape(1,-1))
        fold_recall.append(torch.tensor(recallCV[k]).reshape(1,-1))
        fold_f1.append(torch.tensor(f1CV[k]).reshape(1,-1))
        fold_auc.append(torch.tensor(aucCV[k]).reshape(1,-1))
        fold_aupr.append(torch.tensor(auprCV[k]).reshape(1,-1))
    acc_mat = torch.cat(fold_acc,dim=0)
    recall_mat = torch.cat(fold_recall, dim=0)
    f1_mat = torch.cat(fold_f1, dim=0)
    auc_mat = torch.cat(fold_auc, dim=0)
    aupr_mat = torch.cat(fold_aupr, dim=0)

    accMean = torch.mean(acc_mat).item()
    recallMean = torch.mean(recall_mat).item()
    f1Mean = torch.mean(f1_mat).item()
    aucMean = torch.mean(auc_mat).item()
    auprMean = torch.mean(aupr_mat).item()

    accStd = torch.std(torch.mean(acc_mat, dim=1)).item()
    recallStd = torch.std(torch.mean(recall_mat, dim=1)).item()
    f1Std = torch.std(torch.mean(f1_mat, dim=1)).item()
    aucStd = torch.std(torch.mean(auc_mat, dim=1)).item()
    auprStd = torch.std(torch.mean(aupr_mat, dim=1)).item()
    print('mean acc %.4f' % accMean,
          '\nmean recall %.4f' % recallMean,
          '\nmean f1 %.4f' % f1Mean,
          '\nmean auc %.4f' % aucMean,
          '\nmean aupr %.4f' % auprMean)

    print('std acc %.4f' % accStd,
          '\nstd recall %.4f' % recallStd,
          '\nstd f1 %.4f' % f1Std,
          '\nstd auc %.4f' % aucStd,
          '\nstd aupr %.4f' % auprStd)

def train_forHetero():
    cv_sample, cv_label, cv_norm_mat, \
    gene_node_embedding, dis_node_embedding = data_loader.data_loader_heter(args['data_name'])
    optm_test_acc_list = {}
    optm_test_recall_list = {}
    optm_test_f1_list = {}
    optm_test_auc_list = {}
    optm_test_aupr_list = {}
    optm_epoch_list = {}
    BCE = nn.BCELoss()
    for k in range(5):
        print(k, '--iteration')
        optm_test_acc_list[k] = []
        optm_test_recall_list[k] = []
        optm_test_f1_list[k] = []
        optm_test_auc_list[k] = []
        optm_test_aupr_list[k] = []
        optm_epoch_list[k] = []
        for f in range(5):
            print('fold %d:' % f)
            train_idx = cv_sample[f]['train sample']
            valid_idx = cv_sample[f]['valid sample']
            test_idx = cv_sample[f]['test sample']
            tensor_train_label = cv_label[f]['train label']
            tensor_valid_label = cv_label[f]['valid label']
            tensor_test_label = cv_label[f]['test label']
            gda_adj = cv_norm_mat[f]

            model = LR_GNN.LR_GNN(in_feat=256,
                                  hidd_list=args['hidd_list'],
                                  num_layer=args['num_layer'],
                                  num_rela_type=args['num_rela_type'])
            optm = torch.optim.Adam(model.parameters(), lr=0.01)
            start_time = time.time()
            max_valid_acc = 0
            optm_test_acc = 0
            optm_test_recall = 0
            optm_test_f1 = 0
            optm_test_auc = 0
            optm_test_aupr = 0
            optm_epoch = 0
            for epoch in range(50):
                model.train()
                model.to(args['device'])
                train_prob = model(node_embed1=gene_node_embedding.to(args['device']),
                                   node_embed2=dis_node_embedding.to(args['device']),
                                   adj_wei=gda_adj.to(args['device']),
                                   sample=train_idx,
                                   output_thred=args['output_thred'])
                loss = BCE(train_prob.reshape(-1), tensor_train_label) + \
                        LR_GNN.regu(model, args['lam1'], args['lam2'], args['lam3']).to(args['device'])
                if epoch % 10 == 0:
                    print('Epoch: %d'%epoch,'loss: %.4f'%loss.item())
                optm.zero_grad()
                loss.backward()
                optm.step()

                model.cpu()
                model.eval()
                vali_prob = model(node_embed1=gene_node_embedding,
                                  node_embed2=dis_node_embedding,
                                  adj_wei=gda_adj,
                                  sample=valid_idx,
                                  output_thred=args['output_thred'])
                valid_acc = Com_acc(vali_prob, tensor_valid_label)
                if valid_acc > max_valid_acc:
                    max_valid_acc = valid_acc
                    print('Epoch :', epoch,
                          'valid acc %.4f' % valid_acc.item())
                    optm_epoch = epoch
                    optm_state = {'LR-GNN': model.state_dict(),
                                 'optimizer': optm.state_dict(),
                                 'epoch': epoch}
                    test_prob = model(node_embed1=gene_node_embedding,
                                      node_embed2=dis_node_embedding,
                                      adj_wei=gda_adj,
                                      sample=test_idx,
                                      output_thred=args['output_thred'])
                    optm_test_acc = Com_acc(test_prob, tensor_test_label).item()
                    optm_test_recall, optm_test_f1 = metric(test_prob, tensor_test_label.numpy())
                    optm_test_auc = get_auc(tensor_test_label.numpy(),
                                            test_prob.detach().numpy())
                    optm_test_aupr = get_aupr(tensor_test_label.numpy(),
                                              test_prob.detach().numpy())
                    optm_test_prob = test_prob
                    print('Epoch: %d' % epoch,
                          'test acc: %.4f' % optm_test_acc,
                          'test recall: %.4f' % optm_test_recall,
                          'test f1: %.4f' % optm_test_f1,
                          '\ntest auc: %.4f' % optm_test_auc,
                          'test aupr: %.4f' % optm_test_aupr)

            # torch.save(optm_state, str(k)+' '+str(optm_epoch)+'model parameter.pth')
            # torch.save(optm_test_prob, str(k)+' '+str(optm_epoch)+'test prob.pth')
            end_time = time.time()
            optm_test_acc_list[k].append(optm_test_acc)
            optm_test_recall_list[k].append(optm_test_recall)
            optm_test_auc_list[k].append(optm_test_auc)
            optm_test_aupr_list[k].append(optm_test_aupr)
            optm_test_f1_list[k].append(optm_test_f1)
            optm_epoch_list[k].append(optm_epoch)
            print('total time: %.2f' % (end_time-start_time),
                  'Epoch: %d' % optm_epoch,
                  'test acc: %.4f' % optm_test_acc,
                  'test recall: %.4f' % optm_test_recall,
                  'test f1 score: %.4f' % optm_test_f1,
                  'test auc: %.4f' % optm_test_auc,
                  'test aupr: %.4f' % optm_test_aupr)
    return optm_test_acc_list, optm_test_recall_list, \
           optm_test_f1_list, optm_test_auc_list, optm_test_aupr_list, optm_epoch_list

def train_forHomogeCV(cv_sample, cv_label, cv_norm_mat, subg_node_embedding):
    optm_test_acc_list = {}
    optm_test_recall_list = {}
    optm_test_f1_list = {}
    optm_test_auc_list = {}
    optm_test_aupr_list = {}
    optm_epoch_list = {}
    BCE = nn.BCELoss()
    for k in range(5):
        print(k, '--iteration')
        optm_test_acc_list[k] = []
        optm_test_recall_list[k] = []
        optm_test_f1_list[k] = []
        optm_test_auc_list[k] = []
        optm_test_aupr_list[k] = []
        optm_epoch_list[k] = []
        for f in range(5):
            print('fold %d:' % f)
            train_idx = cv_sample[f]['train sample']
            valid_idx = cv_sample[f]['valid sample']
            test_idx = cv_sample[f]['test sample']
            tensor_train_label = cv_label[f]['train label']
            tensor_valid_label = cv_label[f]['valid label']
            tensor_test_label = cv_label[f]['test label']
            norm_matrix = cv_norm_mat[f]

            model = LR_GNN.LR_GNN(in_feat=256,
                                  hidd_list=args['hidd_list'],
                                  num_layer=args['num_layer'],
                                  num_rela_type=args['num_rela_type'])
            optm = torch.optim.Adam(model.parameters(), lr=0.01)
            start_time = time.time()
            max_valid_acc = 0
            optm_test_acc = 0
            optm_test_recall = 0
            optm_test_f1 = 0
            optm_test_auc = 0
            optm_test_aupr = 0
            optm_epoch = 0
            for epoch in range(50):
                model.train()
                model.to(args['device'])
                train_prob = model(node_embed1=subg_node_embedding.to(args['device']),
                                   node_embed2='',
                                   adj_wei=norm_matrix.to(args['device']),
                                   sample=train_idx,
                                   output_thred=args['output_thred'])
                train_acc = Com_acc(train_prob, tensor_train_label.cuda()).item()
                loss = BCE(train_prob.reshape(-1), tensor_train_label.cuda()) + \
                       LR_GNN.regu(model,  args['lam1'], args['lam2'], args['lam3']).to(args['device'])
                optm.zero_grad()
                loss.backward()
                optm.step()
                if epoch % 10 == 0:
                    print('Epoch: %d' % epoch,
                          'loss: %.4f' % loss.item(),
                          'train acc: %.4f' % train_acc)

                model.cpu()
                model.eval()
                vali_prob = model(node_embed1=subg_node_embedding.to(args['device']),
                                  node_embed2='',
                                  adj_wei=norm_matrix.to(args['device']),
                                  sample=valid_idx,
                                  output_thred=args['output_thred'])
                valid_acc = Com_acc(vali_prob, tensor_valid_label).item()
                if valid_acc > max_valid_acc:
                    print('Epoch :', epoch,
                          'validation acc %.4f' % valid_acc)
                    max_valid_acc = valid_acc

                    optm_epoch = epoch
                    optm_state = {'LR-GNN': model.state_dict(),
                                  'optimizer': optm.state_dict(),
                                  'epoch': epoch}
                    test_prob = model(node_embed1=subg_node_embedding.to(args['device']),
                                      node_embed2='',
                                      adj_wei=norm_matrix.to(args['device']),
                                      sample=test_idx,
                                      output_thred=args['output_thred'])
                    optm_test_prob = test_prob
                    optm_test_acc = Com_acc(test_prob, tensor_test_label).item()
                    optm_test_recall, optm_test_f1 = metric(test_prob, tensor_test_label.numpy())
                    optm_test_auc = get_auc(tensor_test_label.numpy(),
                                            test_prob.detach().numpy())
                    optm_test_aupr = get_aupr(tensor_test_label.numpy(),
                                              test_prob.detach().numpy())
                    print('Epoch: %d' % epoch,
                          'test acc: %.4f' % optm_test_acc,
                          'test recall: %.4f' % optm_test_recall,
                          'test f1: %.4f' % optm_test_f1,
                          'test auc: %.4f' % optm_test_auc,
                          'test aupr: %.4f' % optm_test_aupr)
            # torch.save(optm_state, str(k)+' '+str(optm_epoch)+'model parameter.pth')
            # torch.save(optm_test_prob, str(k)+' '+str(optm_epoch)+'test prob.pth')
            end_time = time.time()
            optm_test_acc_list[k].append(optm_test_acc)
            optm_test_recall_list[k].append(optm_test_recall)
            optm_test_f1_list[k].append(optm_test_f1)
            optm_test_auc_list[k].append(optm_test_auc)
            optm_test_aupr_list[k].append(optm_test_aupr)
            optm_epoch_list[k].append(optm_epoch)
            print('total time: %.2f' % (end_time - start_time),
                  'Epoch: %d' % optm_epoch,
                  'test acc: %.4f' % optm_test_acc,
                  'test recall: %.4f' % optm_test_recall,
                  'test f1 score: %.4f' % optm_test_f1,
                  'test auc: %.4f' % optm_test_auc,
                  'test aupr: %.4f' % optm_test_aupr)
    return optm_test_acc_list, optm_test_recall_list, optm_test_f1_list, \
                   optm_test_auc_list, optm_test_aupr_list, optm_epoch_list

def train_forHomoge():
    graph_test_acc = {}
    graph_test_recall = {}
    graph_test_f1 = {}
    graph_test_auc = {}
    graph_test_aupr = {}
    graph_epoch = {}
    s_time = time.time()
    for i in range(1, 5):
        graph_num = 'graph' + str(i)
        print(graph_num)
        cv_sample, cv_label, cv_norm_mat,\
        subg_node_embedding = data_loader.data_loader_homo(args['data_name'], graph_num)
        optm_test_acc_cv, optm_test_recall_cv, \
        optm_test_f1_cv, optm_test_auc_cv, \
        optm_test_aupr_cv, optm_epoch_cv = train_forHomogeCV(cv_sample=cv_sample,
                                                             cv_label=cv_label,
                                                             cv_norm_mat=cv_norm_mat,
                                                             subg_node_embedding=subg_node_embedding)
        graph_test_acc[i] = optm_test_acc_cv
        graph_test_recall[i] = optm_test_recall_cv
        graph_test_f1[i] = optm_test_f1_cv
        graph_test_auc[i] = optm_test_auc_cv
        graph_test_aupr[i] = optm_test_aupr_cv
        graph_epoch[i] = optm_epoch_cv
    e_time = time.time()
    print('total time: ', int(e_time - s_time))
    return graph_test_acc,graph_test_recall,graph_test_f1,graph_test_auc,graph_test_aupr,graph_epoch

def main():
    if args['data_type'] == 'Heterogeneous network':
        cv_test_acc_list, cv_test_recall_list, cv_test_f1_list, \
        cv_test_auc_list, cv_test_aupr_list, cv_epoch_list=train_forHetero()
        cvMetric(cv_test_acc_list,
                 cv_test_recall_list,
                 cv_test_f1_list,
                 cv_test_auc_list,
                 cv_test_aupr_list)

    elif args['data_type'] == 'Homogeneous network':
        graph_test_acc,graph_test_recall,graph_test_f1,\
        graph_test_auc,graph_test_aupr,graph_epoch=train_forHomoge()
        for g in range(1,5):
            print('graph' + str(g))
            cvMetric(graph_test_acc[g],
                     graph_test_recall[g],
                     graph_test_f1[g],
                     graph_test_auc[g],
                     graph_test_aupr[g])
    else:
        raise ValueError("Error: the data type should be 'Heterogeneous network' or 'Homogeneous network'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='Heterogeneous network')
    parser.add_argument('--data_name', type=str, default='LDA')
    # parser.add_argument('--data_type', type=str, default='Homogeneous network')
    # parser.add_argument('--data_name', type=str, default='PPI')
    parser.add_argument('--hidd_list', default=[64,64,64])
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--num_rela_type', type=int, default=2)
    parser.add_argument('--output_thred', type=int, default=0)
    parser.add_argument('--lam1', type=float, default=0.0)
    parser.add_argument('--lam2', type=float, default=0.0)
    parser.add_argument('--lam3', type=float, default=0.01)

    args = parser.parse_args().__dict__
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
