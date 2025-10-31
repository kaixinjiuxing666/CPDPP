import errno
import sys
import os.path as osp
import torch.utils.data as data
import os
import torch
import numpy as np
import random

from BERT_token_process import CUHKPEDES_BERT_token
# from BERT_token_process_cuhk import CUHKPEDES_BERT_token 
# from BERT_token_process_icfg import CUHKPEDES_BERT_token
# from BERT_token_process_rstp import CUHKPEDES_BERT_token

import time
import os
import pickle
from PIL import Image, ImageOps

import matplotlib.pyplot as plt

import tqdm
import pickle
from test_config import parse_args

def data_config(dir, batch_size, split, max_length, embedding_type,transform):
    print("The word length is", max_length)
    if embedding_type == 'BERT':
        print("The word embedding type is BERT")
        data_split = CUHKPEDES_BERT_token(dir, split, max_length, transform)
    print("the number of", split, ":", len(data_split))
    if split == 'train':
        shuffle = True
    else:
        shuffle = False
    loader = data.DataLoader(data_split, batch_size, shuffle=shuffle, num_workers=2)
    return loader

def optimizer_function(args, model):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.adam_lr, betas=(args.adam_alpha, args.adam_beta), eps=args.epsilon)
        print("optimizer is：Adam")
    return optimizer

def lr_scheduler(optimizer, args):

    if args.lr_decay_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min', factor=args.lr_decay_ratio,
                                                           patience=5, min_lr=args.end_lr)
        print("lr_scheduler is ReduceLROnPlateau")
    ################    
    elif args.lr_decay_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoches)
        print("lr_scheduler is CosineAnnealingLR")
    ################
    else:
        if '_' in args.epoches_decay:
            epoches_list = args.epoches_decay.split('_')
            epoches_list = [int(e) for e in epoches_list]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, epoches_list, gamma=args.lr_decay_ratio)
            print("lr_scheduler is MultiStepLR")
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(args.epoches_decay), gamma=args.lr_decay_ratio)
            print("lr_scheduler is StepLR")
    return scheduler

def load_checkpoint(model,resume):
    start_epoch=0
    if os.path.isfile(resume):
        # checkpoint = torch.load(resume,map_location={'cuda:6':'cuda:0'})
        checkpoint= torch.load(resume, map_location='cpu')
        # checkpoint= torch.load(resume, map_location='cuda:0')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print('Load checkpoint at epoch %d.' % (start_epoch))
    return start_epoch,model


# def load_checkpoint(model,resume):
#     start_epoch=0
#     if os.path.isfile(resume):
#         #checkpoint = torch.load(resume,map_location={'cuda:6':'cuda:0'})
#         checkpoint= torch.load(resume, map_location='cpu')
#         #checkpoint= torch.load(resume, map_location='cpu')
#         start_epoch = checkpoint['epoch']
#         model[0].load_state_dict(checkpoint['state_dict_0'])
#         model[1].load_state_dict(checkpoint['state_dict_1'])
#         print('Load checkpoint at epoch %d.' % (start_epoch))
#     return start_epoch,model

class AverageMeter(object):
    """
    Computes and stores the averate and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py #L247-262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += n * val
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, epoch, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    filename = os.path.join(dst, str(epoch)) + '.pth.tar'
    torch.save(state, filename)

def gradual_warmup(epoch,init_lr,optimizer,epochs):
    lr = init_lr
    if epoch < epochs:
        warmup_percent_done = (epoch+1) / epochs
        warmup_learning_rate = init_lr * warmup_percent_done
        lr = warmup_learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

# def compute_topk(query, gallery, target_query, target_gallery, k=[1,10], reverse=False):
#     result = []
#     query = query / (query.norm(dim=1,keepdim=True)+1e-12)
#     gallery = gallery / (gallery.norm(dim=1,keepdim=True)+1e-12)
#     sim_cosine = torch.matmul(query, gallery.t())
#     result.extend(topk(sim_cosine, target_gallery, target_query, k))
#     if reverse:
#         result.extend(topk(sim_cosine, target_query, target_gallery, k, dim=0))
#     return result


# def topk(sim, target_gallery, target_query, k=[1,10], dim=1):
#     result = []
#     maxk = max(k)
#     size_total = len(target_gallery)
#     _, pred_index = sim.topk(maxk, dim, True, True)
#     pred_labels = target_gallery[pred_index]
#     if dim == 1:
#         pred_labels = pred_labels.t()
#     correct = pred_labels.eq(target_query.view(1,-1).expand_as(pred_labels))

#     for topk in k:
#         correct_k = torch.sum(correct[:topk], dim=0)
#         correct_k = torch.sum(correct_k > 0).float()
#         result.append(correct_k * 100 / size_total)
#     return result

def check_exists(root):
    if os.path.exists(root):
        return True
    return False

def load_embedding(path):
    word_embedding=torch.from_numpy(np.load(path))
    (vocab_size,embedding_size)=word_embedding.shape
    print('Load word embedding,the shape of word embedding is [{},{}]'.format(vocab_size,embedding_size))
    return word_embedding

def load_part_model(model,path):
    model_dict = model.state_dict()
    checkpoint = torch.load(path)
    pretrained_dict = checkpoint["state_dict"]
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

############################################################################ important

def test_map(query_feature,query_label,gallery_feature, gallery_label):
    query_feature = query_feature / (query_feature.norm(dim=1, keepdim=True) + 1e-12)
    gallery_feature = gallery_feature / (gallery_feature.norm(dim=1, keepdim=True) + 1e-12)
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):# 6148
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i],  gallery_feature, gallery_label)
        # print(CMC_tmp[:5])
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    CMC = CMC.float()
    CMC = CMC / len(query_label)
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))
    return CMC[0], CMC[4], CMC[9], ap / len(query_label)

def evaluate(qf, ql, gf, gl):# 查询特征，查询标签，库特征，库标签
    query = qf.view(-1, 1) # 2048,1
    score = torch.mm(gf, query) # 3074,2048   2048,1   
    score = score.squeeze(1).cpu() # 3074
    score = score.numpy() # 3074
    index = np.argsort(score) # 从小到大排序，并返回索引
    index = index[::-1] # 索引倒序，即变成了从大到小#################
    gl=gl.cuda().data.cpu().numpy() # 3074
    ql=ql.cuda().data.cpu().numpy() # 1
    query_index = np.argwhere(gl == ql) # 返回与查询标签相同的库标签的索引
    CMC_tmp = compute_mAP(index, query_index)
    return CMC_tmp

#================================================================ load top-1 top-5 top-10 img

def test_map(query_feature,query_label,gallery_feature, gallery_label):
    query_feature = query_feature / (query_feature.norm(dim=1, keepdim=True) + 1e-12)
    gallery_feature = gallery_feature / (gallery_feature.norm(dim=1, keepdim=True) + 1e-12)
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    cmc10_test_6148_dic = {}
    #------
    save_path = "/home/xs/Documents/codes/TIPCB-00/data/img_rank/"

    # dict_path_6148 = save_path + "test_img_index_path_dict_6148.npz"
    # with open(dict_path_6148, 'rb') as f_dict_path_6148:
    #     data_6148 = pickle.load(f_dict_path_6148)

    dict_path_3074 = save_path + "test_img_index_path_dict_3074.npz"
    with open(dict_path_3074, 'rb') as f_dict_path_3074:
        data_3074 = pickle.load(f_dict_path_3074)

    args = parse_args()
    #------
    for i in tqdm.tqdm(range(len(query_label))):# 6148
        #----- 可选，保存真实图
        # img_path_6148 = data_6148[str(i)]
        # print("true label: ",img_path_6148)

        # test_img_6148 = Image.open(img_path_6148)
        # test_img_6148.save(save_path + "label_true.jpg")
        #-----
        ap_tmp, CMC_tmp = evaluate(i, query_feature[i], query_label[i],  gallery_feature, gallery_label,data_3074,args)
        #---------- 必要，保存top-10结果到字典
        cmc10_test_6148_dic[str(i)] = CMC_tmp[:10]
        #---------- 
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    CMC = CMC.float()
    CMC = CMC / len(query_label)
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))
    #--------- 必要，保存top-10结果到字典，以及加载
    # cmc_save_path = r'/home/xs/Documents/codes/TIPCB-00/data/img_rank/cmc10_test_6148_dic.npz'
    cmc_save_path = "/home/xs/Documents/codes/TIPCB-00/single/log/"+args.name+"/cmc10_test_6148_dic.npz"
    with open(cmc_save_path, 'wb') as ff:
        pickle.dump(cmc10_test_6148_dic, ff)

    # with open(cmc_save_path, 'rb') as fff:
        # loaded_cmc10_test_6148_dic = pickle.load(fff)
        # print(loaded_cmc10_test_6148_dic["0"])
    #---------
    return CMC[0], CMC[4], CMC[9], ap / len(query_label)


def evaluate(i, qf, ql, gf, gl,data_3074,args):# 查询特征，查询标签，库特征，库标签
    query = qf.view(-1, 1) # 2048,1
    score = torch.mm(gf, query) # 3074,2048   2048,1   
    score = score.squeeze(1).cpu() # 3074
    score = score.numpy() # 3074
    index = np.argsort(score) # 从小到大排序，并返回索引
    index = index[::-1] # 索引倒序，即变成了从大到小==================
    gl=gl.cuda().data.cpu().numpy() # 3074
    ql=ql.cuda().data.cpu().numpy() # 1
    query_index = np.argwhere(gl == ql) # 返回与查询标签相同的库标签的索引
    CMC_tmp = compute_mAP(index, query_index)
    #---------------- 必要，根据top5结果生成对应的排序图
    n=0
    # fig, axs = plt.subplots(1,5,figsize=(18,10),layout="constrained",subplot_kw={'xticks': [], 'yticks': []})
    fig, axs = plt.subplots(1,5, figsize=(10,5),subplot_kw={'xticks': [], 'yticks': []},)
    # plt.box(on=False)

    for j in index[:5]:
        img_path_3074 = data_3074[str(j)]
        test_img_3074 = Image.open(img_path_3074).resize((64,192))
        if j in query_index:
            test_img_3074 = ImageOps.expand(test_img_3074, border=(3,3), fill="#00FF00")
        else:
            test_img_3074 = ImageOps.expand(test_img_3074, border=(3,3), fill="#FF0000")
        axs.flat[n].imshow(test_img_3074)
        axs.flat[n]._frameon = False

        n=n+1

    plt.tight_layout(pad=1, w_pad=0.5)
    plt.savefig("/home/xs/Documents/codes/TIPCB-00/single/log/"+args.name+"/top5_img/" + str(i) + "_" + str(int(ql)) + ".jpg", dpi=100)
    plt.close()
    #----------------
    return CMC_tmp
########################################################################## important

def compute_mAP(index, good_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc
    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index) # 在index中查找与good_index相同的值，为真，其他为假
    rows_good = np.argwhere(mask == True) # 正确标签在库中的索引，按索引大小从前往后排序
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1 # 将第一个索引之后的值全赋为1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
