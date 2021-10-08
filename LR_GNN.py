import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN_encoder(nn.Module):
    def __init__(self,
                 num_rela_type,
                 in_size,
                 out_size):
        super(GCN_encoder,self).__init__()
        self.rela_nei_update = nn.ModuleList()
        self.rela_self_update = nn.ModuleList()
        for i in range(num_rela_type):
            self.rela_nei_update.append(nn.Linear(in_size, out_size, bias=False))
            self.rela_self_update.append(nn.Linear(in_size, out_size, bias=False))
    def forward(self,
                rela_type,
                adj_wei,
                node_embedding1,
                node_embedding2):
        self_information = self.rela_self_update[rela_type](node_embedding1)
        nei_agg = torch.mm(adj_wei, node_embedding2)
        nei_information = self.rela_nei_update[rela_type](nei_agg)
        new_node_embedding = F.relu(nei_information + self_information)
        return new_node_embedding

class propagation_rule(nn.Module):
    def __init__(self,
                 pre_size,
                 next_size):
        super(propagation_rule,self).__init__()
        self.com_feat = nn.Linear(pre_size, next_size, bias=True)
        self.ew1 = nn.Linear(pre_size,next_size,bias=False)
        self.ew2 = nn.Linear(pre_size,next_size,bias=False)
    def forward(self,
                sample,
                node_embedding1,
                node_embedding2):
        idx1 = sample[0]
        idx2 = sample[1]
        gather_information = torch.add(node_embedding1[idx1],node_embedding2[idx2])
        ew1 = torch.sigmoid(self.ew1(node_embedding1[idx1]))
        ew2 = torch.sigmoid(self.ew2(node_embedding2[idx2]))
        com_feat = torch.tanh(self.com_feat(gather_information))
        lr = ew1*com_feat + ew2*com_feat
        return lr

class LR_GNN(nn.Module):
    def __init__(self, in_feat, hidd_list, num_layer,
                 num_rela_type):
        super(LR_GNN, self).__init__()
        self.num_layer = num_layer
        self.num_rela_type = num_rela_type
        self.encoder = nn.ModuleList()
        self.propa_lr = nn.ModuleList()
        self.fusing = nn.ModuleList()

        for i in range(num_layer):
            if i == 0:
                self.encoder.append(GCN_encoder(num_rela_type, in_feat, hidd_list[i]))
            else:
                self.encoder.append(GCN_encoder(num_rela_type, hidd_list[i - 1], hidd_list[i]))

        for i in range(num_layer):
            if i == num_layer - 1:
                self.propa_lr.append(propagation_rule(hidd_list[i], 1))
            else:
                self.propa_lr.append(propagation_rule(hidd_list[i], hidd_list[i + 1]))

        for i in range(1, num_layer):
            if i == num_layer - 1:
                self.fusing.append(nn.Linear(hidd_list[i], 1, bias=True))
            else:
                self.fusing.append(nn.Linear(hidd_list[i], hidd_list[i + 1], bias=True))

    def forward(self,
                node_embed1,
                node_embed2,
                adj_wei,
                sample,
                output_thred):
        if self.num_rela_type == 2:
            node_embedding = [node_embed1, node_embed2]
        elif self.num_rela_type == 1:
            assert node_embed2 == '', \
                "If the number of relation types is one, the node_embed2 must be '' "
            node_embedding = node_embed1
        else:
            raise ValueError("Error: the num_rela_type is 2 for LDA, MDA data and 1 for PPI, DDI data.")
        lr_dict = {}
        lr_fused_dict = {}
        for i in range(self.num_layer):
            if self.num_rela_type > 1:
                # gene conv dis
                ge_embed = self.encoder[i](0, adj_wei,
                                           node_embedding[0], node_embedding[1])
                # dis conv gene
                di_embed = self.encoder[i](1, adj_wei.t(),
                                           node_embedding[1], node_embedding[0])
                node_embedding = [ge_embed, di_embed]

                link_repre = self.propa_lr[i](sample, node_embedding[0], node_embedding[1])

            else:
                # prot,drug conv prot,drug
                node_embedding = self.encoder[i](0, adj_wei, node_embedding, node_embedding)

                link_repre = self.propa_lr[i](sample, node_embedding, node_embedding)

            lr_dict['link_repre' + str(i)] = link_repre

        if self.num_layer > 1:

            output = torch.sigmoid(self.fusing[0](lr_dict['link_repre' + str(0)]) +
                                   lr_dict['link_repre' + str(1)])
            lr_fused_dict['fused_link_repre'+str(0)] = output
            if len(lr_dict) > 2:
                for j in range(2, len(lr_dict)):
                    output = torch.sigmoid(self.fusing[j - 1](output) + lr_dict['link_repre' + str(j)])
                    if j < len(lr_dict) - 1:
                        lr_fused_dict['fused_link_repre' + str(j - 1)] = output
        else:
            output = torch.sigmoid(link_repre)

        if output_thred == 1:
            return lr_dict, lr_fused_dict, output
        elif output_thred == 0:
            return output
        else:
            raise ValueError("Error: the output_thred should be 0 or 1. 0: return output. 1: return link representations and output")

def regu(model, lam1, lam2, lam3):
    reg_loss = 0
    for name,param in model.named_parameters():
        if 'encoder' in name and 'weight' in name:
            l2_reg = torch.norm(param,p=2)
            reg_loss += lam1*l2_reg
        if 'com_feat' in name and 'weight' in name:
            l2_reg = torch.norm(param,p=2)
            reg_loss += lam2*l2_reg
        if 'fusing' in name and 'weight' in name:
            l2_reg = torch.norm(param,p=2)
            reg_loss += lam3*l2_reg
    return reg_loss