import numpy as np
import torch

def LDAdata_loader():
    cv_sample = np.load('LDA\\cv_sample.npy',
                        allow_pickle=True).item()
    cv_label = np.load('LDA\\cv_label.npy',
                       allow_pickle=True).item()
    cv_norm_mat = np.load('LDA\\cv_norm_mat.npy',
                          allow_pickle=True).item()
    gene_node_embedding = torch.load('LDA\\lnc_node_embedding.pth')
    dis_node_embedding = torch.load('LDA\\disForlnc_node_embedding.pth')
    return cv_sample,cv_label,cv_norm_mat,gene_node_embedding,dis_node_embedding


def MDAdata_loader():
    cv_sample = np.load('MDA\\cv_sample.npy',
                        allow_pickle=True).item()
    cv_label = np.load('MDA\\cv_label.npy',
                       allow_pickle=True).item()
    cv_norm_mat = np.load('MDA\\cv_norm_mat.npy',
                          allow_pickle=True).item()
    gene_node_embedding = torch.load('MDA\\mic_node_embedding.pth')
    dis_node_embedding = torch.load('MDA\\disFormic_node_embedding.pth')
    return cv_sample, cv_label, cv_norm_mat, gene_node_embedding, dis_node_embedding

def ppi_load(graph_num):
    cv_sample = np.load('PPI//'+graph_num+'//cv_sample.npy', allow_pickle=True).item()
    cv_label = np.load('PPI//'+graph_num+'//cv_label.npy', allow_pickle=True).item()
    cv_norm_mat = np.load('PPI//'+graph_num+'//cv_norm_mat.npy', allow_pickle=True).item()
    #####################################################
    graph_node = []
    file_graph_node = open('PPI//'+graph_num+'//'+graph_num+' graph node.txt')
    for line in file_graph_node:
        graph_node.append(int(line.split('\n')[0]))
    file_graph_node.close()
    node_num = len(graph_node)
    #####################################################
    all_node_embedding = torch.load('PPI//all_node_embedding.pth')

    graph_node_tensor = torch.tensor(graph_node)

    graph_node_embedding = all_node_embedding[graph_node_tensor,:]

    print('the dimension of node embedding: ', graph_node_embedding.shape)
    return cv_sample,cv_label,cv_norm_mat,graph_node_embedding

def ddi_load(graph_num):
    cv_sample = np.load('DDI//'+graph_num+'//cv_sample.npy', allow_pickle=True).item()
    cv_label = np.load('DDI//'+graph_num+'//cv_label.npy', allow_pickle=True).item()
    cv_norm_mat = np.load('DDI//'+graph_num+'//cv_norm_mat.npy', allow_pickle=True).item()
    #####################################################
    graph_node = []
    file_graph_node = open('DDI//'+graph_num+'//'+graph_num+' graph node.txt')
    for line in file_graph_node:
        graph_node.append(int(line.split('\n')[0]))
    file_graph_node.close()
    node_num = len(graph_node)
    #####################################################
    all_node_embedding = torch.load('DDI//all_node_embedding.pth')

    graph_node_tensor = torch.tensor(graph_node)

    graph_node_embedding = all_node_embedding[graph_node_tensor,:]

    print(graph_node_embedding.shape)
    return cv_sample,cv_label,cv_norm_mat,graph_node_embedding

def data_loader_heter(name):
    if name == 'MDA':
        return MDAdata_loader()
    elif name == 'LDA':
        return LDAdata_loader()
    else:
        raise ValueError("Error: the name is not included. Please input 'LDA' or 'MDA'.")

def data_loader_homo(name,graph_num):
    if name == 'PPI':
        return ppi_load(graph_num)
    elif name == 'DDI':
        return ddi_load(graph_num)
    else:
        raise ValueError("Error: the name is not included. Please input 'PPI' or 'DDI'.")