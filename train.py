import sys

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
import yaml
import time
import shutil
# from tensorboard_logger import configure, log_value
from test_model import start_test
from train_config import parse_args
from function import data_config, optimizer_function, load_checkpoint, lr_scheduler, AverageMeter, save_checkpoint, \
    gradual_warmup, fix_seed, Logger


# from models.model import Network
# from models.model_adapcb_perspective import Network
# from models.model_balancepcb_dmm import Network
from models.model_suturepcb_attention import Network



from CMPM import Loss
from torch import nn
from torch.autograd import Variable
import time
# conda activate tipcb38
# cd /home/xs/Documents/codes/TIPCB-00/single
# nohup python -u train.py > /home/xs/Documents/codes/TIPCB-00/single/log/tipcb_60_11004.log 2>&1 &

def train(epoch, train_loader, network, opitimizer, compute_loss, args, checkpoint_dir):
    train_loss = AverageMeter()
    # switch to train mode
    network.train()

    for step, (images, captions, labels, mask) in enumerate(train_loader):
        images = images.cuda()
        captions = captions.cuda()
        labels = labels.cuda()
        mask = mask.cuda()
        opitimizer.zero_grad()

        #---------------------- compute loss with pcb ---------------------------------------------
        # img_f3, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46, img_f4,\
        # txt_f3, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, txt_f4 = network(images, captions, mask)
        # loss = compute_loss(
        #     img_f3, img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46,
        #     txt_f3, txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, labels)
        # train_loss.update(loss, images.shape[0])

        img_f3, img_f41, img_f42, img_f43, img_f44, img_f45, img_f4,\
        txt_f3, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f4 = network(images, captions, mask)
        # n1,n2,n3,n4,n5,n6,n_all,n_1,n_2,n_3,n_4,n_5,n_6, = network(images, captions, mask)
        loss = compute_loss(
            img_f3, img_f41, img_f42, img_f43, img_f44, img_f45, img_f4,
            txt_f3, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f4, labels)
        train_loss.update(loss, images.shape[0])


        # img_f3, img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46, \
        # txt_f3, txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46 = network(images, captions, mask)
        # loss = compute_loss(
        #     img_f3, img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46,
        #     txt_f3, txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, labels)
        # train_loss.update(loss, images.shape[0])


        # img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46, \
        # txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, hyp_ratio = network(images, captions, mask)
        # # n1,n2,n3,n4,n5,n6,n_all,n_1,n_2,n_3,n_4,n_5,n_6, = network(images, captions, mask)
        # loss = compute_loss(
        #     img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46,
        #     txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, hyp_ratio, labels)
        # train_loss.update(loss, images.shape[0])

#================================= doing
        # img_f3, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46, img_f4,\
        # txt_f3, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, txt_f4, hyp_ratio = network(images, captions, mask)
        # # n1,n2,n3,n4,n5,n6,n_all,n_1,n_2,n_3,n_4,n_5,n_6, = network(images, captions, mask)
        # loss = compute_loss(
        #     img_f3, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46, img_f4,
        #     txt_f3, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, txt_f4,hyp_ratio, labels)
        # train_loss.update(loss, images.shape[0])


        # img_f3, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46, img_f47, img_f4,\
        # txt_f3, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, txt_f47, txt_f4, hyp_ratio = network(images, captions, mask)
        # # n1,n2,n3,n4,n5,n6,n_all,n_1,n_2,n_3,n_4,n_5,n_6, = network(images, captions, mask)
        # loss = compute_loss(
        #     img_f3, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46, img_f47, img_f4,
        #     txt_f3, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, txt_f47, txt_f4,hyp_ratio, labels)
        # train_loss.update(loss, images.shape[0])


        # img_f3, img_f41, img_f4, txt_f3, txt_f41, txt_f4 = network(images, captions, mask)
        # loss = compute_loss(
        #     img_f3, img_f41, img_f4, txt_f3, txt_f41, txt_f4, labels)
        # train_loss.update(loss, images.shape[0])


        # img_f41, img_f4, txt_f41, txt_f4 = network(images, captions, mask)
        # loss = compute_loss(
        #     img_f41, img_f4, txt_f41, txt_f4, labels)
        # train_loss.update(loss, images.shape[0])


        # img_f3, img_f4, txt_f3, txt_f4 = network(images, captions, mask)
        # loss = compute_loss(
        #     img_f3, img_f4, txt_f3, txt_f4, labels)
        # train_loss.update(loss, images.shape[0])
        #---------------------- compute loss no pcb ---------------------------------------------
        # img_f4, txt_f4 = network(images, captions, mask)
        # loss = compute_loss(img_f4, txt_f4, labels)
        # train_loss.update(loss, images.shape[0])


        # graduate
        loss.backward()
        opitimizer.step()

        if step % 100 == 0:
            print(
                "Train Epoch:[{}/{}] iteration:[{}/{}] cmpm_loss:{:.4f}".format(epoch + 1, args.num_epoches, step,
                                                                                len(train_loader), train_loss.avg,))
            # print(
            #     "one:{:.4f}  two:{:.4f}  three:{:.4f}  four:{:.4f}  five:{:.4f}  six:{:.4f}".format(hyp_ratio[0],hyp_ratio[1],hyp_ratio[2],hyp_ratio[3],hyp_ratio[4],hyp_ratio[5],))
            # print("{}-----{}-----{}-----{}-----{}-----{}-------------------{}, ".format(n1,n2,n3,n4,n5,n6,n_all))
            # print("Percent:{:.1%},{:.1%},{:.1%},{:.1%},{:.1%},{:.1%}".format(n_1,n_2,n_3,n_4,n_5,n_6))
            # print("Float:{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(hyp_ratio[0],hyp_ratio[1],hyp_ratio[2],hyp_ratio[3],hyp_ratio[4],hyp_ratio[5]))
    if epoch + 1 >= 75:
    # if epoch + 1 >= 0:

        state = {"epoch": epoch + 1,
                "state_dict": network.state_dict(),
                "W": compute_loss.W
                }

        save_checkpoint(state, epoch+1, checkpoint_dir)


def main(network, dataloader, compute_loss, optimizer, scheduler, start_epoch, args, checkpoint_dir):
    start = time.time()
    for epoch in range(start_epoch, args.num_epoches):
        print("**********************************************************")

        if epoch < args.warm_epoch:
            print('learning rate warm_up')
            if args.optimizer == 'sgd':
                optimizer = gradual_warmup(epoch, args.sgd_lr, optimizer, epochs=args.warm_epoch)
            else:
                optimizer = gradual_warmup(epoch, args.adam_lr, optimizer, epochs=args.warm_epoch)
        ###########
        print("lr : {}".format(opitimizer.state_dict()['param_groups'][0]['lr']))
        ###########
        train(epoch, dataloader['train'], network, optimizer, compute_loss, args, checkpoint_dir)

        scheduler.step()

        Epoch_time = time.time() - start
        start = time.time()
        print('Epoch_training complete in {:.0f}m {:.0f}s'.format(
            Epoch_time // 60, Epoch_time % 60))

if __name__=='__main__':

    # time.sleep(39000)
    args = parse_args()

    # load GPU
    str_ids = args.gpus.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True  # make the training speed faster
    fix_seed(args.seed)

    name = args.name
    # set some paths
    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir = os.path.join(checkpoint_dir, name)
    log_dir = args.log_dir
    log_dir = os.path.join(log_dir, name)
    
    if not os.path.isdir('/home/xs/Documents/codes/TIPCB-00/single/log/'+args.name+'/models'):
        os.makedirs('/home/xs/Documents/codes/TIPCB-00/single/log/'+args.name+'/models')
        os.makedirs('/home/xs/Documents/codes/TIPCB-00/single/log/'+args.name+'/top5_img')
        os.makedirs('/home/xs/Documents/codes/TIPCB-00/single/log/'+args.name+'/heat_map')

    shutil.copyfile("/home/xs/Documents/codes/TIPCB-00/single/train.py","/home/xs/Documents/codes/TIPCB-00/single/log/"+args.name+"/train.py")
    shutil.copyfile("/home/xs/Documents/codes/TIPCB-00/single/CMPM.py","/home/xs/Documents/codes/TIPCB-00/single/log/"+args.name+"/CMPM.py")
    shutil.copyfile("/home/xs/Documents/codes/TIPCB-00/single/test_model.py","/home/xs/Documents/codes/TIPCB-00/single/log/"+args.name+"/test_model.py")
    shutil.copyfile("/home/xs/Documents/codes/TIPCB-00/single/BERT_token_process.py","/home/xs/Documents/codes/TIPCB-00/single/log/"+args.name+"/BERT_token_process.py")
    shutil.copyfile("/home/xs/Documents/codes/TIPCB-00/single/function.py","/home/xs/Documents/codes/TIPCB-00/single/log/"+args.name+"/function.py")

    shutil.copyfile("/home/xs/Documents/codes/TIPCB-00/single/train_config.py","/home/xs/Documents/codes/TIPCB-00/single/log/"+args.name+"/train_config.py")
    shutil.copyfile("/home/xs/Documents/codes/TIPCB-00/single/test_config.py","/home/xs/Documents/codes/TIPCB-00/single/log/"+args.name+"/test_config.py")

    shutil.copyfile("/home/xs/Documents/codes/TIPCB-00/single/models/model.py","/home/xs/Documents/codes/TIPCB-00/single/log/"+args.name+"/models/model.py")
    shutil.copyfile("/home/xs/Documents/codes/TIPCB-00/single/models/model_adapcb_perspective.py","/home/xs/Documents/codes/TIPCB-00/single/log/"+args.name+"/models/model_adapcb_perspective.py")
    shutil.copyfile("/home/xs/Documents/codes/TIPCB-00/single/models/model_balancepcb_dmm.py","/home/xs/Documents/codes/TIPCB-00/single/log/"+args.name+"/models/model_balancepcb_dmm.py")
    shutil.copyfile("/home/xs/Documents/codes/TIPCB-00/single/models/model_suturepcb_attention.py","/home/xs/Documents/codes/TIPCB-00/single/log/"+args.name+"/models/model_suturepcb_attention.py")

    shutil.copyfile("/home/xs/Documents/codes/TIPCB-00/single/models/CNN_text.py","/home/xs/Documents/codes/TIPCB-00/single/log/"+args.name+"/models/CNN_text.py")

    sys.stdout = Logger(os.path.join(log_dir, "train_log.txt"))
    opt_dir = os.path.join('log', name)
    if not os.path.exists(opt_dir):
        os.makedirs(opt_dir)
    with open('%s/opts_train.yaml' % opt_dir, 'w') as fp:
        yaml.dump(vars(args), fp, default_flow_style=False)

    # pre-process the dataset
    transform_train_list = [
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((args.height, args.width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    transform_val_list = [
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    # define dictionary: data_transforms
    # data_transforms = {
    #     'train': transforms.Compose(transform_train_list),
    #     'val': transforms.Compose(transform_val_list),
    # }

    # dataloaders = {x: data_config(args.dir, args.batch_size, x, args.max_length, args.embedding_type, transform=data_transforms[x])
    #                for x in ['train', 'val']}

    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'test': transforms.Compose(transform_val_list),
    }
    dataloaders = {x: data_config(args.dir, args.batch_size, x, args.max_length, args.embedding_type, transform=data_transforms[x])
                   for x in ['train', 'test']}
    # loss function
    if args.CMPM:
        print("import CMPM")

    compute_loss = Loss(args).cuda()
    model = Network(args).cuda()

    # compute the model size:
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # load checkpoint:
    if args.resume is not None:
        start_epoch, model = load_checkpoint(model, args.resume)
    else:
        print("Do not load checkpoint,Epoch start from 0")
        start_epoch = 0

    # opitimizer:
    opitimizer = optimizer_function(args, model)
    exp_lr_scheduler = lr_scheduler(opitimizer, args)
    main(model, dataloaders, compute_loss, opitimizer, exp_lr_scheduler, start_epoch, args, checkpoint_dir)
    start_test()

