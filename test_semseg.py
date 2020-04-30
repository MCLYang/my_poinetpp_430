"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
# from data_utils.S3DISDataLoader import ScannetDatasetWholeScene
# from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np

from torch.utils.tensorboard import SummaryWriter

import argparse
import os
# from data_utils.S3DISDataLoader import S3DISDataset
from BigredDataSet import BigredDataSet
from metrics import AverageMeter

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
# import torch.nn.parallel
import pdb
import glob
import time
from collections import OrderedDict
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def mIoU(y_pred, y):
    ioU = []
    class_num = [0, 1]
    for num in class_num:
        I = np.sum(np.logical_and(y_pred == num, y == num))
        U = np.sum(np.logical_or(y_pred == num, y == num))
        ioU.append(I / float(U))
    ave = np.mean(ioU)
    return (ave)

def convert_state_dict(state_dict):

    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_msg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 16]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--num_point', type=int,  default=20000, help='Point Number [default: 4096]')
    parser.add_argument('--num_channel', type=int,  default=5, help='Point Number [default: 4096]')

    parser.add_argument('--visual', action='store_true', default=False, help='Whether visualize result [default: False]')
    parser.add_argument(
        '--num_workers', type=int, help='number of data loading workers', default=32)
    parser.add_argument('--dataset', type=str, default='../bigRed_h5_pointnet', help="dataset path")
    parser.add_argument('--load_model_dir', type=str, default='2020-04-26_07-36/checkpoints/', help="retest")
    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    temp_package = torch.load(args.load_model_dir + 'best_model_valmiou_0.48365261753776134_1.pth')
    args.num_channel = temp_package['num_channel']

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = '2020-04-26_07-36/'
    # best_model_valmiou_0.48365261753776134_1.pth
    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 2
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point


    TEST_DATASET = BigredDataSet(
        root=args.dataset,
        is_train=False,
        is_validation=True,
        is_test=False,
        num_channel=args.num_channel
    )
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                                 pin_memory=True, drop_last=True)

    log_string("The number of test data is: %d" %  len(testDataLoader))
#best_model_valmiou_0.48365261753776134_1.pth
    '''MODEL LOADING'''

    MODEL = importlib.import_module(args.model)

    classifier = MODEL.get_model(NUM_CLASSES,num_channel = 5).cuda()
    temp_dict = temp_package['state_dict']
    temp_dict = convert_state_dict(temp_dict)
    classifier.load_state_dict(temp_dict)
    # pdb.set_trace()

    num_batches = len(testDataLoader)
    mean_miou = AverageMeter()
    mean_acc = AverageMeter()
    mean_time = AverageMeter()
    classifier = classifier.eval()
    with torch.no_grad():
        labelweights = np.zeros(NUM_CLASSES)
        print('---- TEST ----')
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
            mean_time.update(toc - tic)
        test_time = mean_time.avg
        val_ave_miou = mean_miou.avg
        val_ave_acc = mean_acc.avg
        print('val_miou: %f' % temp_package['Validation_ave_miou'])
        print('Test_miou: %f' % val_ave_miou)
        print('Test_acc: %f' % val_ave_acc)
        print('Test ave time(sec/frame): %f' % (test_time))
        print('Test ave time(frame/sec): %f' % (1 / test_time))

        #
        #
        #
        #     pred_val = seg_pred.contiguous().cpu().data.numpy()
        #     seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
        #     batch_label = target.cpu().data.numpy()
        #     target = target.view(-1, 1)[:, 0]
        #
        #     # pdb.set_trace()
        #     pred_val = np.argmax(pred_val, 2)
        #     correct = np.sum((pred_val == batch_label))
        #     tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
        #     labelweights += tmp
        #     for l in range(NUM_CLASSES):
        #         total_seen_class[l] += np.sum((batch_label == l))
        #         total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
        #         total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
        # labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        # mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
        # print('Test_accuracy:', (total_correct / float(total_seen)))
        # print('Test_mIoU:', mIoU)

        # iou_per_class_str = '------- IoU --------\n'
        # for l in range(NUM_CLASSES):
        #     iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
        #         seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
        #         total_correct_class[l] / float(total_iou_deno_class[l]))

        # log_string('Eval mean loss: %f' % (loss_sum / num_batches))
        # log_string(iou_per_class_str)
        # log_string('Eval mean loss: %f' % (loss_sum / num_batches))
        # log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))


    # with torch.no_grad():
    #     num_batches = len(TEST_DATASET_WHOLE_SCENE)
    #     print("num_batches",num_batches)
    #
    #     total_seen_class = [0 for _ in range(NUM_CLASSES)]
    #     total_correct_class = [0 for _ in range(NUM_CLASSES)]
    #     total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
    #
    #     log_string('---- EVALUATION WHOLE SCENE----')
    #
    #     for batch_idx in range(num_batches):
    #         print("visualize [%d/%d] %s ..." % (batch_idx+1, num_batches))
    #         total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
    #         total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
    #         total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
    #         if args.visual:
    #             fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
    #             fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')
    #
    #         whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
    #         whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
    #         vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
    #         for _ in tqdm(range(args.num_votes), total=args.num_votes):
    #             scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
    #             num_blocks = scene_data.shape[0]
    #             s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
    #             batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))
    #
    #             batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
    #             batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
    #             batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))
    #             for sbatch in range(s_batch_num):
    #                 start_idx = sbatch * BATCH_SIZE
    #                 end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
    #                 real_batch_size = end_idx - start_idx
    #                 batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
    #                 batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
    #                 batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
    #                 batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
    #                 batch_data[:, :, 3:6] /= 1.0
    #
    #                 torch_data = torch.Tensor(batch_data)
    #                 torch_data= torch_data.float().cuda()
    #                 torch_data = torch_data.transpose(2, 1)
    #                 seg_pred, _ = classifier(torch_data)
    #                 batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()
    #
    #                 vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
    #                                            batch_pred_label[0:real_batch_size, ...],
    #                                            batch_smpw[0:real_batch_size, ...])
    #
    #         pred_label = np.argmax(vote_label_pool, 1)
    #
    #         for l in range(NUM_CLASSES):
    #             total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
    #             total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
    #             total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l)))
    #             total_seen_class[l] += total_seen_class_tmp[l]
    #             total_correct_class[l] += total_correct_class_tmp[l]
    #             total_iou_deno_class[l] += total_iou_deno_class_tmp[l]
    #
    #         iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float) + 1e-6)
    #         print(iou_map)
    #         arr = np.array(total_seen_class_tmp)
    #         tmp_iou = np.mean(iou_map[arr != 0])
    #         log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
    #         print('----------------------------')
    #
    #         filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
    #         with open(filename, 'w') as pl_save:
    #             for i in pred_label:
    #                 pl_save.write(str(int(i)) + '\n')
    #             pl_save.close()
    #         for i in range(whole_scene_label.shape[0]):
    #             color = g_label2color[pred_label[i]]
    #             color_gt = g_label2color[whole_scene_label[i]]
    #             if args.visual:
    #                 fout.write('v %f %f %f %d %d %d\n' % (
    #                 whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1],
    #                 color[2]))
    #                 fout_gt.write(
    #                     'v %f %f %f %d %d %d\n' % (
    #                     whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color_gt[0],
    #                     color_gt[1], color_gt[2]))
    #         if args.visual:
    #             fout.close()
    #             fout_gt.close()
    #
    #     IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
    #     iou_per_class_str = '------- IoU --------\n'
    #     for l in range(NUM_CLASSES):
    #         iou_per_class_str += 'class %s, IoU: %.3f \n' % (
    #             seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
    #             total_correct_class[l] / float(total_iou_deno_class[l]))
    #     log_string(iou_per_class_str)
    #     log_string('eval point avg class IoU: %f' % np.mean(IoU))
    #     log_string('eval whole scene point avg class acc: %f' % (
    #         np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
    #     log_string('eval whole scene point accuracy: %f' % (
    #                 np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))
    #
    #     print("Done!")

if __name__ == '__main__':
    args = parse_args()
    main(args)
