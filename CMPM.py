import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.CMPM = args.CMPM
        self.epsilon = args.epsilon
        self.num_classes = args.num_classes
        if args.resume:
            checkpoint = torch.load(args.resume)
            self.W = Parameter(checkpoint['W'])
            print('=========> Loading in parameter W from pretrained models')
        else:
            self.W = Parameter(torch.randn(args.feature_size, args.num_classes))
            self.init_weight()
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
    def init_weight(self):
        nn.init.xavier_uniform_(self.W.data, gain=1)

    def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels):
        """
        Cross-Modal Projection Matching Loss(CMPM)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
            i2t_loss: cmpm loss for image projected to text
            t2i_loss: cmpm loss for text projected to image
            pos_avg_sim: average cosine-similarity for positive pairs
            neg_avg_sim: averate cosine-similarity for negative pairs
        """

        batch_size = image_embeddings.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)

        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True) 
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        #------------

        #-----------
        image_proj_text = torch.matmul(image_embeddings, text_norm.t())
        text_proj_image = torch.matmul(text_embeddings, image_norm.t())

        # normalize the true matching distribution 
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)# 64 64

        i2t_pred = F.softmax(image_proj_text, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + self.epsilon))
        t2i_pred = F.softmax(text_proj_image, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + self.epsilon))

        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
        # print(cmpm_loss)
        return cmpm_loss



    def mutual_compute_cmpm_loss(self, image_embeddings, text_embeddings,labels):

        batch_size = image_embeddings.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)
        #---------- 0
        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        image_proj_text = torch.matmul(image_embeddings, text_norm.t())
        text_proj_image = torch.matmul(text_embeddings, image_norm.t())
        #-----------
        i2t_pred = F.softmax(image_proj_text, dim=1)
        t2i_pred = F.softmax(text_proj_image, dim=1)
        ####################### mutual KL(img0, img1) kl1
        # i2t_loss = F.softmax(image_embeddings, dim=1) * (F.log_softmax(image_embeddings, dim=1) - F.log_softmax(text_embeddings, dim=1))
        # mutual_loss = torch.mean(torch.sum(i2t_loss, dim=1))
        # print("loss 1 :",mutual_loss)
        ####################### mutual KL(img0-txt0, img1-txt1) kl2
        it2it_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - F.log_softmax(Variable(text_proj_image), dim=1))
        it2it_loss_mean = torch.mean(torch.sum(it2it_loss, dim=1))
        mutual_loss = it2it_loss_mean
        # print("loss 1 :",mutual_loss)
        #######################
        all_loss = mutual_loss

        return all_loss

    #--------------------------------------- with pcb -------------------------------------------
    # def forward(self, img_f3, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46, img_f4,
    #                   txt_f3, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, txt_f4,equel,labels):
    #     loss = 0.0
    #     if self.CMPM:
    #         loss = self.compute_cmpm_loss(img_f3, txt_f3, labels) \
    #                   + self.compute_cmpm_loss(img_f41, txt_f41, labels) \
    #                   + self.compute_cmpm_loss(img_f42, txt_f42, labels) \
    #                   + self.compute_cmpm_loss(img_f43, txt_f43, labels) \
    #                   + self.compute_cmpm_loss(img_f44, txt_f44, labels) \
    #                   + self.compute_cmpm_loss(img_f45, txt_f45, labels) \
    #                   + self.compute_cmpm_loss(img_f46, txt_f46, labels) \
    #                   + self.compute_cmpm_loss(img_f4, txt_f4, labels)

    #     return loss
#=========================================================================
    def forward(self, img_f3, img_f41, img_f42, img_f43, img_f44, img_f45, img_f4,\
                      txt_f3, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f4, labels):
        loss = 0.0
        if self.CMPM:
            loss = self.compute_cmpm_loss(img_f3, txt_f3, labels) \
                      + self.compute_cmpm_loss(img_f41, txt_f41, labels) \
                      + self.compute_cmpm_loss(img_f42, txt_f42, labels) \
                      + self.compute_cmpm_loss(img_f43, txt_f43, labels) \
                      + self.compute_cmpm_loss(img_f44, txt_f44, labels) \
                      + self.compute_cmpm_loss(img_f45, txt_f45, labels) \
                      + self.compute_cmpm_loss(img_f4, txt_f4, labels)

        return loss
    
    
    
    # def forward(self, img_f3, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46, img_f47, img_f48, img_f49, img_f410, img_f411, img_f412, img_f4,\
    #                   txt_f3, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, txt_f47, txt_f48, txt_f49, txt_f410, txt_f411, txt_f412, txt_f4, labels):
    #     loss = 0.0
    #     if self.CMPM:
    #         loss = self.compute_cmpm_loss(img_f3, txt_f3, labels) \
    #                   + self.compute_cmpm_loss(img_f41, txt_f41, labels) \
    #                   + self.compute_cmpm_loss(img_f42, txt_f42, labels) \
    #                   + self.compute_cmpm_loss(img_f43, txt_f43, labels) \
    #                   + self.compute_cmpm_loss(img_f44, txt_f44, labels) \
    #                   + self.compute_cmpm_loss(img_f45, txt_f45, labels) \
    #                   + self.compute_cmpm_loss(img_f46, txt_f46, labels) \
    #                   + self.compute_cmpm_loss(img_f47, txt_f47, labels) \
    #                   + self.compute_cmpm_loss(img_f48, txt_f48, labels) \
    #                   + self.compute_cmpm_loss(img_f49, txt_f49, labels) \
    #                   + self.compute_cmpm_loss(img_f410, txt_f410, labels) \
    #                   + self.compute_cmpm_loss(img_f411, txt_f411, labels) \
    #                   + self.compute_cmpm_loss(img_f412, txt_f412, labels) \
    #                   + self.compute_cmpm_loss(img_f4, txt_f4, labels)

    #     return loss
#=========================================================================
    # def forward(self, img_f3, img_f4, txt_f3, txt_f4, labels):
    #     loss = 0.0
    #     if self.CMPM:
    #         loss = self.compute_cmpm_loss(img_f3, txt_f3, labels) \
    #                   + self.compute_cmpm_loss(img_f4, txt_f4, labels)

    #     return loss


    # def forward(self, img_f41, img_f4,
    #                   txt_f41, txt_f4, labels):
    #     loss = 0.0
    #     if self.CMPM:
    #         loss = self.compute_cmpm_loss(img_f41, txt_f41, labels) \
    #                   + self.compute_cmpm_loss(img_f4, txt_f4, labels)

    #     return loss


    # def forward(self, img_f4, txt_f4, labels):
    #     loss = 0.0
    #     if self.CMPM:
    #         loss =  self.compute_cmpm_loss(img_f4, txt_f4, labels)

    #     return loss


#================== doing
    # def forward(self, img_f3, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46, img_f4,
    #                txt_f3, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, txt_f4, hyp_ratio, labels):
    #     loss = 0.0
    #     if self.CMPM:
    #         loss = self.compute_cmpm_loss(img_f3, txt_f3, labels) \
    #                   + hyp_ratio[0] * self.compute_cmpm_loss(img_f41, txt_f41, labels) \
    #                   + hyp_ratio[1] * self.compute_cmpm_loss(img_f42, txt_f42, labels) \
    #                   + hyp_ratio[2] * self.compute_cmpm_loss(img_f43, txt_f43, labels) \
    #                   + hyp_ratio[3] * self.compute_cmpm_loss(img_f44, txt_f44, labels) \
    #                   + hyp_ratio[4] * self.compute_cmpm_loss(img_f45, txt_f45, labels) \
    #                   + hyp_ratio[5] * self.compute_cmpm_loss(img_f46, txt_f46, labels) \
    #                   + self.compute_cmpm_loss(img_f4, txt_f4, labels) 

    #     return loss



    # def forward(self, img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46,
    #                txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46,hyp_ratio, labels):
    #     loss = 0.0
    #     if self.CMPM:
    #         loss =  hyp_ratio[0] * self.compute_cmpm_loss(img_f41, txt_f41, labels) \
    #                   + hyp_ratio[1] * self.compute_cmpm_loss(img_f42, txt_f42, labels) \
    #                   + hyp_ratio[2] * self.compute_cmpm_loss(img_f43, txt_f43, labels) \
    #                   + hyp_ratio[3] * self.compute_cmpm_loss(img_f44, txt_f44, labels) \
    #                   + hyp_ratio[4] * self.compute_cmpm_loss(img_f45, txt_f45, labels) \
    #                   + hyp_ratio[5] * self.compute_cmpm_loss(img_f46, txt_f46, labels) \
    #                   + self.compute_cmpm_loss(img_f4, txt_f4, labels)

    #     return loss
    #--------------------------------------- with pcb 回字型 -------------------------------------------
    # def forward(self, img_f41, img_f42, img_f43, img_f44, img_f4,
    #                   txt_f41, txt_f42, txt_f43, txt_f44, txt_f4, labels):
    #     loss = 0.0
    #     if self.CMPM:
    #         loss =  0.1*self.compute_cmpm_loss(img_f41, txt_f41, labels) \
    #                   + 0.2*self.compute_cmpm_loss(img_f42, txt_f42, labels) \
    #                   + 0.3*self.compute_cmpm_loss(img_f43, txt_f43, labels) \
    #                   + 0.6*self.compute_cmpm_loss(img_f44, txt_f44, labels) \
    #                   + self.compute_cmpm_loss(img_f4, txt_f4, labels)

    #     return loss

    #--------------------------------------- no pcb -------------------------------------------
    # def forward(self, img_f4, txt_f4, labels):
    #     loss = 0.0
    #     if self.CMPM:
    #         loss = self.compute_cmpm_loss(img_f4, txt_f4, labels)

    #     return loss

    ########################################################## origin deep mutual ############################################################
    # origin，表示使用了deep mutual learning论文中互相迭代优化方法，loss = true_loss（真实标签） + mutual_loss（另一个模型的预测标签），

    # def forward(self,  img_0, txt_0, labels):
    #     loss = 0.0
    #     if self.CMPM:

    #         true_loss = self.compute_cmpm_loss(img_0, txt_0, labels)
    #         mutual_loss = self.mutual_compute_cmpm_loss(img_0, txt_0, labels)

    #         loss = true_loss + mutual_loss

    #     return loss, mutual_loss