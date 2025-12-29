from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class cos_loss(nn.Module):
    ''' Compute the KL contrastive loss within each batch  
    '''
    def __init__(self):
        super(cos_loss, self).__init__()

    def forward(self, x1, x2):
        device = (torch.device('cuda')
                  if x1.is_cuda
                  else torch.device('cpu'))
        batchSize = x1.size(0)#（20）
        batch = (batchSize-1)
        # diag_mat = (1 - torch.eye(batchSize)).to(device)

        x1_f = F.normalize(x1, dim=1)#5,3
        x2_f = F.normalize(x2, dim=1)#5,3
        loss1 = nn.CosineEmbeddingLoss()(x1,x2, torch.ones(x1.shape[0]).to(device))

        all_prob =( 1- torch.mm(x1_f, x2_f.t())).to(device)#所有相似性
        pos = 1 - torch.sum(x1_f * x2_f, dim = 1).to(device)
        aa = all_prob - pos
        loss2 = torch.mean((2- aa)/(2*batch))
        loss = loss1 + loss2
        return loss2
        
class BatchCriterion(nn.Module):
    ''' Compute the loss within each batch  
    '''
    def __init__(self, negM, T):
        super(BatchCriterion, self).__init__()
        self.negM = negM
        self.T = T
              #获得（20,20）的单位矩阵
        
    def forward(self, x):
        batchSize = x.size(0)#（20）
        
        #get positive innerproduct
        reordered_x = torch.cat((x.narrow(0,batchSize//2,batchSize//2),\
                x.narrow(0,0,batchSize//2)), 0)                 #打乱顺序 重新排序
        #reordered_x = reordered_x.data
        pos = (x*reordered_x.data).sum(1).div_(self.T).exp_()   #正样本的相似性（20）
        batch_size = x.size(0)
        diag_mat = 1 - torch.eye(batchSize).cuda()
        #get all innerproduct, remove diag
        all_prob = torch.mm(x,x.t().data).div_(self.T).exp_()*diag_mat  #获得所有样例的内积 20,20

        # x_sort,_ = torch.sort(all_prob)
        # x_sort[:,-7:-1]=0
        # all_prob = x_sort

        if self.negM==1:
            all_div = all_prob.sum(1)    #以第一个样例为中心  与其它样例的相似性 对其进行求和  所有正负样例 20
        else:
            #remove pos for neg
            all_div = (all_prob.sum(1) - pos)*self.negM + pos

        lnPmt = torch.div(pos, all_div)  #正样例比重    20

        # negative probability
        Pon_div = all_div.repeat(batchSize,1)    #20,20  每一个 所占的比重 以自己为中心
        lnPon = torch.div(all_prob, Pon_div.t())  #20,20
        lnPon = -lnPon.add(-1)
        
        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        # also remove the pos term
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_()#去除正的样例
        lnPmt.log_()

        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.sum(0)

        # negative multiply m
        lnPonsum = lnPonsum * self.negM
        loss = - (lnPmtsum + lnPonsum)/batchSize
        return loss



class statm_loss(nn.Module):
    def __init__(self, eps=2):
        super(statm_loss, self).__init__()
        self.eps = eps

    def forward(self,x, y):
        x = x.view(x.size(0),x.size(1),-1)
        y = y.view(y.size(0),y.size(1),-1)
        x_mean = x.mean(dim=2)
        y_mean = y.mean(dim=2)
        mean_gap = (x_mean-y_mean).pow(2).mean(1)
        return mean_gap.mean()

class statm_loss(nn.Module):
    def __init__(self, eps=2):
        super(statm_loss, self).__init__()
        self.eps = eps

    def forward(self,x, y):
        x = x.view(x.size(0),x.size(1),-1)
        y = y.view(y.size(0),y.size(1),-1)
        x_mean = x.mean(dim=2)
        y_mean = y.mean(dim=2)
        mean_gap = (x_mean-y_mean).pow(2).mean(1)
        return mean_gap.mean()
        
# 原来为[10,64]  [10,64] 然后用以下一行  改成对应的形式
# features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)#10,2,128
# SupConLoss(features, labels)    #label[1,23,56,9,....,20] 共10个

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.

    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss