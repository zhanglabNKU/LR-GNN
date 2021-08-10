import numpy as np
import torch
import torch.nn as nn
import time
import argparse
import LR_GNN
import data_loader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc


def Com_acc(output,lab):
    output = output.reshape(-1)
    lab = lab.reshape(-1)
    result = output.ge(0.5).float() == lab
    acc = result.float().mean()
    return acc

def Com_recall(output,lab):
    pred = output.ge(0.5).float()
    pred = pred.reshape(-1)
    posi_index = np.where(np.array(lab)==1)[0]
    posi_pred,posi_label = np.array(pred)[posi_index],np.array(lab)[posi_index]
    recall = np.sum(posi_pred == posi_label,dtype = np.float64)/(posi_index).shape[0]
    return recall

def get_auc(test_label, test_prob):
    auc = roc_auc_score(test_label, np.array(test_prob))
    return auc

def get_aupr(test_label, test_prob):
    precision, recall, _ = precision_recall_curve(test_label, np.array(test_prob))
    return auc(recall, precision)

def train_forHetero():
    train_sample, valid_sample, test_sample, \
    gene_node_embedding, dis_node_embedding, \
    train_label, valid_label, test_label = data_loader.data_loader_heter(args['data_name'])
    gene_num = gene_node_embedding.shape[0]
    dis_num = dis_node_embedding.shape[0]
    gda_adj = data_loader.GDA_adj(gene_num, dis_num, train_sample, train_label)

    train_idx = torch.tensor(list(zip(*train_sample)))
    valid_idx = torch.tensor(list(zip(*valid_sample)))
    test_idx = torch.tensor(list(zip(*test_sample)))

    tensor_train_label = torch.from_numpy(train_label).float().to(args['device'])
    tensor_valid_label = torch.from_numpy(valid_label).float()
    tensor_test_label = torch.from_numpy(test_label).float()

    BCE = nn.BCELoss()
    for k in range(5):
        model = LR_GNN.LR_GNN(in_feat=256,
                              hidd_list=args['hidd_list'],
                              num_layer=args['num_layer'],
                              num_rela_type=args['num_rela_type'])
        optm = torch.optim.Adam(model.parameters(), lr=0.01)
        start_time = time.time()
        max_valid_acc = 0
        optm_test_acc = 0
        optm_test_recall = 0
        optm_test_auc = 0
        optm_test_aupr = 0
        optm_epoch = 0
        optm_state = 0
        optm_test_prob = 0
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
            if epoch % 10 == 0:
                print('Epoch :',epoch,
                      'valid acc %.4f'%valid_acc.item())
            if valid_acc > max_valid_acc:
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
                optm_test_recall = Com_recall(test_prob, tensor_test_label)
                optm_test_auc = get_auc(test_label, test_prob.detach().numpy())
                optm_test_aupr = get_aupr(test_label, test_prob.detach().numpy())
                optm_test_prob = test_prob
        end_time = time.time()
        torch.save(optm_state, str(k)+' '+str(optm_epoch)+'model parameter.pth')
        torch.save(optm_test_prob, str(k)+' '+str(optm_epoch)+'test prob.pth')

        print('total time: %.2f' % (end_time-start_time),
              'Epoch: %d' % optm_epoch,
              'test acc: %.4f' % optm_test_acc,
              'test recall: %.4f' % optm_test_recall,
              'test auc: %.4f' % optm_test_auc,
              'test aupr: %.4f' % optm_test_aupr)

def train_forHomoge():
    for i in range(1,5):
        graph_num = 'graph' + str(i)
        norm_matrix, \
        train_sample, test_sample, \
        subg_node_embedding, \
        train_label, test_label = data_loader.data_loader_homo(args['data_name'], graph_num)
        train_idx = torch.tensor(list(zip(*train_sample)))
        test_idx = torch.tensor(list(zip(*test_sample)))

        tensor_train_label = torch.tensor(train_label).float().to(args['device'])
        tensor_test_label = torch.tensor(test_label).float()
        BCE = nn.BCELoss()
        for k in range(5):
            model = LR_GNN.LR_GNN(in_feat=256,
                                  hidd_list=args['hidd_list'],
                                  num_layer=args['num_layer'],
                                  num_rela_type=args['num_rela_type'])
            optm = torch.optim.Adam(model.parameters(), lr=0.01)
            start_time = time.time()
            max_train_acc = 0
            optm_test_acc = 0
            optm_test_recall = 0
            optm_test_auc = 0
            optm_test_aupr = 0
            optm_epoch = 0
            optm_state = 0
            optm_test_prob = 0
            for epoch in range(50):
                model.train()
                model.to(args['device'])
                train_prob = model(node_embed1=subg_node_embedding.to(args['device']),
                                   node_embed2='',
                                   adj_wei=norm_matrix.to(args['device']),
                                   sample=train_idx,
                                   output_thred=args['output_thred'])
                train_acc = Com_acc(train_prob, tensor_train_label).item()
                if train_acc > max_train_acc:
                    max_train_acc = train_acc
                if epoch % 10 == 0:
                    print('Epoch :', epoch,
                          'train acc: %.4f' % train_acc.item())
                if train_acc == max_train_acc:
                    optm_epoch = epoch
                    model.cpu()
                    model.eval()
                    test_prob = model(subg_node_embedding.cpu(),
                                       '',
                                       norm_matrix.cpu(),
                                       test_idx,
                                       output_thred=args['output_thred'])
                    optm_test_acc = Com_acc(test_prob, tensor_test_label).item()
                    optm_test_recall = Com_recall(test_prob, tensor_test_label)
                    optm_test_auc = get_auc(test_label, test_prob.detach().numpy())
                    optm_test_aupr = get_aupr(test_label, test_prob.detach().numpy())
                    optm_state = {'LR-GNN': model.state_dict(),
                                 'optimizer': optm.state_dict(),
                                 'epoch': epoch}
                    optm_test_prob = test_prob

                model.train()
                model.to(args['device'])
                loss = BCE(train_prob.reshape(-1), tensor_train_label) + \
                       LR_GNN.regu(model, args['lam1'], args['lam2'], args['lam3']).to(args['device'])
                optm.zero_grad()
                loss.backward()
                optm.step()
                if epoch % 10 == 0:
                    print('Epoch :', epoch,
                          'loss: %.4f',loss.item())
            end_time = time.time()
            torch.save(optm_state, graph_num+' '+str(k)+' '+str(optm_epoch)+
                       'model parameter.pth')
            torch.save(optm_test_prob, graph_num+' '+str(k)+' '+str(optm_epoch)+
                       'test prob.pth')
            print('total time: %.2f' % (end_time - start_time),
                  'Epoch: %d' % optm_epoch,
                  'test acc: %.4f' % optm_test_acc,
                  'test recall: %.4f' % optm_test_recall,
                  'test auc: %.4f' % optm_test_auc,
                  'test aupr: %.4f' % optm_test_aupr)

def main():
    if args['data_type'] == 'Heterogeneous network':
        train_forHetero()
    elif args['data_type'] == 'Homogeneous network':
        train_forHomoge()
    else:
        raise ValueError("Error: the data type should be 'Heterogeneous network' or 'Homogeneous network'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='Heterogeneous network')
    parser.add_argument('--data_name', type=str, default='LDA')
    # parser.add_argument('--data_type', type=str, default='Homogeneous network')
    # parser.add_argument('--data_name', type=str, default='PPI')
    parser.add_argument('--hidd_list', default=[64,32,16])
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--num_rela_type', type=int, default=2)
    parser.add_argument('--output_thred', type=int, default=0)
    parser.add_argument('--lam1', type=float, default=0.0)
    parser.add_argument('--lam2', type=float, default=0.01)
    parser.add_argument('--lam3', type=float, default=0.01)

    args = parser.parse_args().__dict__
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
