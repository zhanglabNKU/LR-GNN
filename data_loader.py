import numpy as np
import torch


def LDdata_loader():
    train_sample, valid_sample, test_sample = [], [], []
    file_lncdis_train = open('LDA\\train_lncdis.txt')
    for i in file_lncdis_train:
        train_sample.append((int(i.split()[0]), int(i.split()[1].split('\n')[0])))
    file_lncdis_train.close()

    file_lncdis_valid = open('LDA\\valid_lncdis.txt')
    for i in file_lncdis_valid:
        valid_sample.append((int(i.split()[0]), int(i.split()[1].split('\n')[0])))
    file_lncdis_valid.close()

    file_lncdis_test = open('LDA\\test_lncdis.txt')
    for i in file_lncdis_test:
        test_sample.append((int(i.split()[0]), int(i.split()[1].split('\n')[0])))
    file_lncdis_test.close()
    print(len(train_sample), len(valid_sample), len(test_sample))

    gene_node_embedding = torch.load('LDA\\lnc_node_embedding.pth')
    dis_node_embedding = torch.load('LDA\\disForlnc_node_embedding.pth')
    train_label = np.load('LDA\\train label.npy')
    valid_label = np.load('LDA\\valid label.npy')
    test_label = np.load('LDA\\test label.npy')

    return train_sample, valid_sample, test_sample, gene_node_embedding, dis_node_embedding, train_label, valid_label, test_label


def MDdata_loader():
    train_sample, valid_sample, test_sample = [], [], []
    file_micdis_train = open('MDA\\train_micdis.txt')
    for i in file_micdis_train:
        train_sample.append((int(i.split()[0]), int(i.split()[1].split('\n')[0])))
    file_micdis_train.close()

    file_micdis_valid = open('MDA\\valid_micdis.txt')
    for i in file_micdis_valid:
        valid_sample.append((int(i.split()[0]), int(i.split()[1].split('\n')[0])))
    file_micdis_valid.close()

    file_micdis_test = open('MDA\\test_micdis.txt')
    for i in file_micdis_test:
        test_sample.append((int(i.split()[0]), int(i.split()[1].split('\n')[0])))
    file_micdis_test.close()

    print(len(train_sample), len(valid_sample), len(test_sample))
    gene_node_embedding = torch.load('MDA\\mic_node_embedding.pth')
    dis_node_embedding = torch.load('MDA\\disFormic_node_embedding.pth')
    train_label = np.load('MDA\\train label.npy')
    valid_label = np.load('MDA\\valid label.npy')
    test_label = np.load('MDA\\test label.npy')

    return train_sample, valid_sample, test_sample, gene_node_embedding, dis_node_embedding, train_label, valid_label, test_label

def GDA_adj(gene_num, dis_num, train_sample, train_label):
    adj = torch.zeros((gene_num, dis_num))
    for t in range(len(train_sample)):
        if train_label[t] == 1: # postive sample
            adj[train_sample[t][0],train_sample[t][1]] = 1
    gene_degree = torch.sum(adj, dim=1)
    dis_degree = torch.sum(adj, dim=0)
    norm_gene_deg = torch.pow(gene_degree, -0.5)
    norm_dis_deg = torch.pow(dis_degree, -0.5)
    norm_gene_deg[torch.isinf(norm_gene_deg)] = 0
    norm_dis_deg[torch.isinf(norm_dis_deg)] = 0
    norm_gene_deg = torch.diag(norm_gene_deg)
    norm_dis_deg = torch.diag(norm_dis_deg)
    norm_adj = torch.mm(torch.mm(norm_gene_deg, adj.float()),norm_dis_deg)
    return norm_adj

def ppi_load(graph_num):
    ###########################################
    graph_node = []
    file_graph_node = open('PPI\\'+graph_num+' graph node.txt')
    for line in file_graph_node:
        graph_node.append(int(line.split('\n')[0]))
    file_graph_node.close()
    node_num = len(graph_node)
    ############################################
    norm_ppi_matrix = torch.load('PPI\\'+graph_num+'ppi norm matrix.pth')
    ############################################
    train_sample,test_sample = [],[]
    file_train_sample = open('PPI\\'+graph_num+' train_sample.txt')
    for line in file_train_sample:
        train_sample.append((int(line.split(' ')[0]),int(line.split(' ')[1].split('\n')[0])))
    file_train_sample.close()

    file_test_sample = open('PPI\\'+graph_num+' test_sample.txt')
    for line in file_test_sample:
        test_sample.append((int(line.split(' ')[0]),int(line.split(' ')[1].split('\n')[0])))
    file_test_sample.close()

    print(len(train_sample),len(test_sample))
    ############################################
    all_node_embedding = torch.load('PPI\\all_node_embedding.pth')
    graph_node_tensor = torch.tensor(graph_node)
    graph_node_embedding = all_node_embedding[graph_node_tensor,:]
    print(graph_node_embedding.shape)
    ############################################
    train_label,test_label = [],[]
    file_train_label = open('PPI\\'+graph_num+' train_label.txt')
    for line in file_train_label:
        train_label.append(int(line.split('\n')[0]))
    file_train_label.close()
    file_test_label = open('PPI\\'+graph_num+' test_label.txt')
    for line in file_test_label:
        test_label.append(int(line.split('\n')[0]))
    file_test_label.close()
    ############################################
    return norm_ppi_matrix,train_sample,test_sample,graph_node_embedding,train_label,test_label

def ddi_load(graph_num):
    ############################################
    graph_node = []
    file_graph_node = open('DDI\\'+graph_num+' graph node.txt')
    for line in file_graph_node:
        graph_node.append(int(line.split('\n')[0]))
    file_graph_node.close()
    node_num = len(graph_node)
    ############################################
    norm_ddi_matrix = torch.load('DDI\\'+graph_num+'ddi norm matrix.pth')
    ############################################
    train_sample,test_sample = [],[]
    file_train_sample = open('DDI\\'+graph_num+' train_sample.txt')
    for line in file_train_sample:
        train_sample.append((int(line.split(' ')[0]),int(line.split(' ')[1].split('\n')[0])))
    file_train_sample.close()
    file_test_sample = open('DDI\\'+graph_num+' test_sample.txt')
    for line in file_test_sample:
        test_sample.append((int(line.split(' ')[0]),int(line.split(' ')[1].split('\n')[0])))
    file_test_sample.close()
    print(len(train_sample),len(test_sample))
    ############################################
    all_node_embedding = torch.load('DDI\\all_node_embedding.pth')
    graph_node_tensor = torch.tensor(graph_node)
    graph_node_embedding = all_node_embedding[graph_node_tensor,:]
    print(graph_node_embedding.shape)
    ############################################
    train_label,test_label = [],[]
    file_train_label = open('DDI\\'+graph_num+' train_label.txt')
    for line in file_train_label:
        train_label.append(int(line.split('\n')[0]))
    file_train_label.close()
    file_test_label = open('DDI\\'+graph_num+' test_label.txt')
    for line in file_test_label:
        test_label.append(int(line.split('\n')[0]))
    file_test_label.close()
    ############################################
    return norm_ddi_matrix,train_sample,test_sample,graph_node_embedding,train_label,test_label

def data_loader_heter(name):
    if name == 'MDA':
        return MDdata_loader()
    elif name == 'LDA':
        return LDdata_loader()
    else:
        raise ValueError("Error: the name of is not included. Please input 'LDA' or 'MDA'.")

def data_loader_homo(name,graph_num):
    if name == 'PPI':
        return ppi_load(graph_num)
    elif name == 'DDI':
        return ddi_load(graph_num)
    else:
        raise ValueError("Error: the name is not included. Please input 'PPI' or 'DDI'.")