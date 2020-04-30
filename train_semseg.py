"""
Author: Benny
Date: Nov 2019
"""
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from BigredDataSet import BigredDataSet

import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
import torch.nn.parallel
import pdb
from metrics import AverageMeter
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


classes = ['Non-pedestrain','Pedestrain']
class2label = {cls: i for i,cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i,cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat
    
def mIoU(y_pred,y):
    ioU = []
    class_num = [0,1]
    for num in class_num:
        I = np.sum(np.logical_and(y_pred == num, y == num))
        U = np.sum(np.logical_or(y_pred == num, y == num))
        ioU.append(I / float(U))
    ave = np.mean(ioU)
    return(ave)


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_msg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=128, type=int, help='Epoch to run [default: 128]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0,1', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=20000, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int,  default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--root', type=str, default='../bigRed_h5_pointnet/', help='root')
    parser.add_argument('--num_channel', type=int, default=4, help="use feature transform")
    parser.add_argument('--num_gpu', type=int, default=2, help='GPU to use [default: GPU 0]')
    parser.add_argument('--num_worker', type=int, default=32, help='GPU to use [default: GPU 0]')



    return parser.parse_args()




def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root_dataset = args.root
    NUM_CLASSES = 2
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    print("start loading training data ...")

    # def __init__(self,
    #              root,
    #              is_train=True,
    #              is_validation=False,
    #              is_test=False,
    #              num_channel=5
    #              ):

    TRAIN_DATASET = BigredDataSet(
    root=root_dataset,
    is_train=True,
    is_validation=False,
    is_test=False,
    num_channel=args.num_channel,
    test_code = False
    )

    print("start loading test data ...")

    TEST_DATASET = BigredDataSet(
    root=root_dataset,
    is_train=False,
    is_validation=True,
    is_test=False,
    num_channel=args.num_channel,
    test_code = False
    )
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_worker, pin_memory=True, drop_last=True, worker_init_fn = lambda x: np.random.seed(x+int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_worker, pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()
    #pdb.set_trace()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet_util.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES,num_channel = args.num_channel)
    gpu_list = list(range(int(max(args.gpu))+1))


    classifier = torch.nn.DataParallel(classifier, device_ids=gpu_list).cuda()
    criterion = MODEL.get_loss().cuda()

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_value = 0
    writer = SummaryWriter()
    counter_play= 0

    mean_miou = AverageMeter()
    mean_acc = AverageMeter()
    mean_loss = AverageMeter()

    print("len(trainDataLoader)",len(trainDataLoader))
    print("len(trainDataLoader)",len(testDataLoader))


    for epoch in range(start_epoch,args.epoch):
        num_batches = len(trainDataLoader)

        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x,momentum))
        classifier.train()
        mean_miou.reset()
        mean_acc.reset()
        mean_loss.reset()
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points, target = points.float().cuda(),target.long().cuda()
            points = points.transpose(2, 1)
            optimizer.zero_grad()
            classifier = classifier.train()
            seg_pred, trans_feat = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            batch_label2 = target.cpu().data.numpy()

            pred_val = np.argmax(pred_val, 2)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)

            current_seen_class = [0 for _ in range(NUM_CLASSES)]
            current_correct_class = [0 for _ in range(NUM_CLASSES)]
            current_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            for l in range(NUM_CLASSES):
                current_seen_class[l] = np.sum((batch_label2 == l))
                current_correct_class[l] = np.sum((pred_val == l) & (batch_label2 == l))
                current_iou_deno_class[l] = np.sum(((pred_val == l) | (batch_label2 == l)))

            m_iou = np.mean(np.array(current_correct_class) / (np.array(current_iou_deno_class, dtype=np.float) + 1e-6))
            loss_num = loss.item()
            acc_num = correct / float(args.batch_size * args.npoint)
            writer.add_scalar('training_loss', loss_num, counter_play)
            writer.add_scalar('training_accuracy', acc_num, counter_play)
            writer.add_scalar('training_mIoU', m_iou, counter_play)
            counter_play = counter_play + 1
            mean_miou.update(m_iou)
            mean_acc.update(acc_num)
            mean_loss.update(loss_num)

        train_ave_miou = mean_miou.avg
        train_ave_loss = mean_loss.avg
        train_ave_acc = mean_acc.avg
        log_string('Training point avg class IoU: %f' % train_ave_miou)
        log_string('Training mean loss: %f' % train_ave_loss)
        log_string('Training accuracy: %f' % train_ave_acc)
        # logger.info('Save model...')
        # savepath = str(checkpoints_dir) + '/traningmiou_'+str(mIoU)+'.pth'
        # # savepath = str(checkpoints_dir) + '/model.pth'
        #
        # log_string('Saving at %s' % savepath)
        # state = {
        #     'epoch': epoch,
        #     'model_state_dict': classifier.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        # }
        # torch.save(state, savepath)
        # log_string('Saving model....')

        print("----------------------Validation----------------------")
        mean_miou.reset()
        mean_acc.reset()
        mean_loss.reset()
        classifier.eval()
        with torch.no_grad():
            num_batches = len(testDataLoader)
            labelweights = np.zeros(NUM_CLASSES)
            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points, target = data
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()
                tic = time.perf_counter()
                pred, _ = classifier(points)
                toc = time.perf_counter()
                # print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
                pred = pred.view(-1, NUM_CLASSES)
                target = target.view(-1, 1)[:, 0]
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                pred_np = pred_choice.cpu().data.numpy()
                target_np = target.cpu().data.numpy()
                m_iou = mIoU(pred_np, target_np)
                mean_miou.update(m_iou)
                mean_acc.update(correct.item() / float(args.batch_size * 20000))

        val_ave_miou = mean_miou.avg
        val_ave_acc = mean_acc.avg
        writer.add_scalar('Validation_ave_miou', val_ave_miou, epoch)
        writer.add_scalar('Validation_ave_acc', val_ave_acc, epoch)
        print('Epoch: %d' % epoch)
        print('Validation_ave_miou: %f' % val_ave_miou)
        print('Train_ave_val_ave_miou: %f' % train_ave_miou)
        # labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        # iou_per_class_str = '------- IoU --------\n'
        # for l in range(NUM_CLASSES):
        #     iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
        #         seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
        #         total_correct_class[l] / float(total_iou_deno_class[l]))
        package = dict()
        package['state_dict'] = classifier.state_dict()
        package['optimizer'] = optimizer.state_dict()
        package['Train_ave_val_ave_miou'] = train_ave_miou
        package['Train_ave_acc'] = train_ave_acc
        package['Train_ave_loss'] = train_ave_loss
        package['Validation_ave_miou'] = val_ave_miou
        package['Validation_ave_acc'] = val_ave_acc
        package['epoch'] = epoch
        package['global_epoch'] = global_epoch
        package['time'] = time.ctime()
        package['num_channel'] = args.num_channel
        package['num_gpu'] = args.num_gpu
        # torch.save(package, save_dir + '/val_miou_%f_val_acc_%f_%d.pth' % (val_ave_miou, val_ave_acc, epoch))


        savepath = str(checkpoints_dir)+'/val_miou_'+str(val_ave_miou)+'_val_acc_'+str(val_ave_acc)+'_'+str(epoch)+'.pth'
        torch.save(package, savepath)

        print('Is Best? ', best_value < val_ave_miou)
        if (best_value < val_ave_miou):
            best_value = val_ave_miou
            savepath = str(checkpoints_dir) + '/best_model_valmiou_' + str(val_ave_miou) +'_'+str(epoch)+'.pth'
            log_string('Saving at %s' % savepath)
            torch.save(package, savepath)


        # if val_ave_miou >= best_iou:
        #     best_iou = val_ave_miou
        #     logger.info('Save model...')
        #     savepath = str(checkpoints_dir) + '/best_model_testmiou_' + str(mIoU) + '.pth'
        #     log_string('Saving at %s' % savepath)
        #     state = {
        #         'epoch': epoch,
        #         'class_avg_iou': mIoU,
        #         'model_state_dict': classifier.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #     }
        #     torch.save(state, savepath)
        #     log_string('Saving model....')
        log_string('Best mIoU: %f' % best_value)

    global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)