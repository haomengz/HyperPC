# @file main.py
# @author Junming Zhang, junming@umich.edu; Haomeng Zhang, haomeng@umich.edu
# @brief main file for training and testing
# @copyright Copyright University of Michigan, Ford Motor Company (c) 2020-2021

import torch
import numpy as np
import os
import time
import yaml
import shutil
import argparse
import torch.nn.functional as F
import torch_geometric.transforms as T
import random

from tqdm import tqdm
from utils.model import Model
from utils.completion3D_dataset import completion3D_class
from utils.mvp_dataset import MVP
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch_geometric.datasets import ShapeNet, ModelNet
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def seed_everything(seed=1234):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    g = torch.Generator()
    g.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(args, loader, optimizer, logger, epoch, check_dir):
    '''
    Note: only complete point clouds are loaded during training, so data.x is
    both the input and label for the point cloud completion task. While partial
    point clouds (data.y) are loaded at testing.
    '''

    model.train()
    loss_summary = {}
    global i
    cos_sim = []
    grad_cls = []
    grad_comp = []

    for j, data in enumerate(loader, 0):
        data = data.to(device)

        if args.dataset == 'c3d':
            pos, batch, pc_label, category = data.pos, data.batch, data.y, data.category
        elif args.dataset == 'm40':
            pos, batch, pc_label, category = data.pos, data.batch, data.pos, data.y
        elif args.dataset == 'snet':
            pos, batch, pc_label, category = data.pos, data.batch, data.pos, data.y
        elif args.dataset == 'mvp':
            pos, batch, pc_label, category = data.pos, data.batch, data.y, data.category
        else:
            raise ValueError('{} dataset has not been supported yet.'.format(args.dataset))

        # training
        model.zero_grad()
        model(None, pos, batch)
        if args.uncertainty_flag:
            loss = model.compute_loss(category, pc_label, model.w1, model.w2).mean()
            loss_summary['W1'] = model.w1.item()
            loss_summary['W2'] = model.w2.item()
            loss_summary['expW1'] = torch.exp(model.w1).item()
            loss_summary['expW2'] = torch.exp(model.w2).item()
            loss_summary['W1^2'] = torch.square(model.w1).item()
            loss_summary['W2^2'] = torch.square(model.w2).item()
            loss_summary['Total Loss'] = loss
        elif args.optimal_search:
            w1 = args.ratio / (args.ratio + 1)
            w2 = 1 - w1
            loss = model.compute_loss(category, pc_label, w1, w2).mean()
        else:
            loss = model.compute_loss(category, pc_label).mean()

        if 'classification' in args.task:
            loss_summary['loss_cls'] = model.loss_classification.mean()
        if 'segmentation' in args.task:
            loss_summary['loss_seg'] = model.loss_segmentation.mean()
        if 'completion' in args.task:
            loss_summary['loss_chamfer'] = model.loss_completion.mean()
            if model.loss_sec is not None:
                loss_summary['loss_sec'] = model.loss_sec.mean()

        # gradients for shared layers
        if args.compute_gradient_norm:
            weights = [w for w in model.encoder.parameters()]
            if args.use_hyperspherical_module:
                weights += [w for w in model.hyperspherical_module.parameters()]

            if 'classification' in args.task:
                model.zero_grad()
                grad_classification = torch.autograd.grad(model.loss_classification.mean(), weights, retain_graph=True)
                flatten_grad_classification = torch.cat([g.flatten() for g in grad_classification])
                grad_norm_classification = torch.norm(flatten_grad_classification)
                loss_summary['grad_norm_classification'] = grad_norm_classification.item()  
                grad_cls.append(grad_norm_classification.item())
            if 'completion' in args.task:
                model.zero_grad()
                grad_completion = torch.autograd.grad(model.loss_completion.mean(), weights, retain_graph=True)
                flatten_grad_completion = torch.cat([g.flatten() for g in grad_completion])
                grad_norm_completion = torch.norm(flatten_grad_completion)
                loss_summary['grad_norm_completion'] = grad_norm_completion.item()  
                grad_comp.append(grad_norm_completion.item())
            if len(args.task) > 1:
                cosine = torch.sum(flatten_grad_classification * flatten_grad_completion) / (1e-9 + grad_norm_completion * grad_norm_classification)
                cos_sim.append(cosine.item())
                loss_summary['grad_cosine'] = cosine.item()

        model.zero_grad()
        loss.backward()

        if args.compute_gradient_norm and args.grad_surgey_flag and cosine < 0:
            new_grad_classification = torch.subtract(flatten_grad_classification, flatten_grad_completion * torch.sum(flatten_grad_completion * flatten_grad_classification) / (torch.sum(flatten_grad_completion * flatten_grad_completion)))
            new_grad_completion = torch.subtract(flatten_grad_completion, flatten_grad_classification * torch.sum(flatten_grad_completion * flatten_grad_classification) / (torch.sum(flatten_grad_classification * flatten_grad_classification)))
            new_cosine = torch.sum(new_grad_classification * new_grad_completion) / (torch.norm(new_grad_classification) * torch.norm(new_grad_completion))
            loss_summary['new_grad_cosine'] = new_cosine    
            start = 0
            for index, w in enumerate(weights):     
                temp_size = w.shape
                temp_num = w.numel()
                new_temp_grad_classification = new_grad_classification[start:start + temp_num].reshape(temp_size)
                new_temp_grad_completion = new_grad_completion[start:start + temp_num].reshape(temp_size)
                start += temp_num
                if args.uncertainty_flag:
                    w.grad = torch.add(1 / (torch.square(model.w1)).item() * new_temp_grad_classification, 1 / (2 * torch.square(model.w2)).item() * new_temp_grad_completion)
                else:
                    w.grad = torch.add(new_temp_grad_classification, new_temp_grad_completion)

        optimizer.step()
        
        # write summary
        if i % 100 == 0:
            for item in loss_summary:
                logger.add_scalar(item, loss_summary[item], i)
            logger.add_scalar('lr', get_lr(optimizer), i)
            print(''.join(['{}: {:.4f}, '.format(k, v) for k,v in loss_summary.items()]))
        i = i + 1
    
    if args.compute_gradient_norm:
        logger.add_scalar('grad_cls_epoch', sum(grad_cls)/len(grad_cls), epoch)
        logger.add_scalar('grad_comp_epoch', sum(grad_comp)/len(grad_comp), epoch)
        logger.add_scalar('cos_epoch', sum(cos_sim)/len(cos_sim), epoch)


def val_one_epoch(args, loader, logger, epoch):
    print()
    print('Evaluating on {}'.format(args.model_name))

    model.eval()
    results = []
    results_classification = []
    results_segmentation = []
    results_completion = []

    for j, data in enumerate(loader, 0):
        data = data.to(device)

        if args.dataset == 'c3d':
            pos, batch, pc_label, category = data.pos, data.batch, data.y, data.category
        elif args.dataset == 'm40':
            pos, batch, pc_label, category = data.pos, data.batch, data.pos, data.y
        elif args.dataset == 'snet':
            pos, batch, pc_label, category = data.pos, data.batch, data.pos, data.y
        elif args.dataset == 'mvp':
            pos, batch, pc_label, category = data.pos, data.batch, data.y, data.category
        else:
            raise ValueError('{} dataset has not been supported yet.'.format(args.dataset))

        # inference
        with torch.no_grad():
            model(None, pos, batch)
            if args.uncertainty_flag:
                loss = model.compute_loss(category, pc_label, model.w1, model.w2)
            elif args.optimal_search:
                w1 = args.ratio / (args.ratio + 1)
                w2 = 1 - w1
                loss = model.compute_loss(category, pc_label, w1, w2)
            else:
                loss = model.compute_loss(category, pc_label)

        if 'completion' in args.task:
            results_completion.append(model.loss_completion)
        if 'classification' in args.task:
            pred = model.pred_classification
            results_classification.append(pred.eq(category).float())
        if 'segmentation' in args.task:
            pred = model.pred_segmentation
            results_segmentation.append(pred.eq(category).float())
        results.append(loss)

    print('Epoch: {:03d}, '.format(epoch), end='')
    
    if 'completion' in args.task:
        results_completion = torch.cat(results_completion, dim=0).mean().item()
        logger.add_scalar('test_chamfer_dist', results_completion, epoch)
        print('Test Chamfer: {:.5f}, '.format(results_completion), end='')
        acc = -results_completion
    if 'classification' in args.task:
        results_classification = torch.cat(results_classification, dim=0).mean().item()
        logger.add_scalar('test_acc', results_classification, epoch)
        print('Test Acc: {:.4f}'.format(results_classification), end='')
        acc = results_classification
    if 'segmentation' in args.task:
        results_segmentation = torch.cat(results_segmentation, dim=0).mean().item()
        logger.add_scalar('test_seg_acc', results_segmentation, epoch)
        print('Test Seg Acc: {:.4f}'.format(results_segmentation), end='')
        acc = results_segmentation
    if len(args.task) > 1 and 'completion' in args.task:
        acc = -results_completion
    print()

    return acc


def train(args, train_dataloader, test_dataloader):

    check_dir, log_dir = check_overwrite(args)
    logger = SummaryWriter(log_dir=log_dir)
    backup(log_dir, parser)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    global i
    i = 1
    acc = -1000000
    for epoch in range(1, args.max_epoch+1):
        # do training
        train_one_epoch(args, train_dataloader, optimizer, logger, epoch, check_dir)
        # reduce learning rate
        scheduler.step()
        # validation
        acc_ = val_one_epoch(args, test_dataloader, logger, epoch)

        if epoch % 1 == 0:
            torch.save(model.state_dict(), os.path.join(check_dir,
                'model_{}_epoch.pth'.format(epoch)))

        if acc_ > acc:
            # save model
            torch.save(model.state_dict(), os.path.join(check_dir,
                'model.pth'))
            acc = acc_


def fake_partial_point_clouds(pos):
    '''
    pos: 2048 x 3
    '''
    inputs = pos
    pos = pos[pos[:, 0]>0.35]
    if pos.size(0)==0:
        pos = inputs
    pos = pos[np.random.choice(pos.size(0), 2048)]
    return pos


def evaluate(args, loader, save_dir):
    print()
    print('Evaluating on {}'.format(args.model_name))

    model.eval()
    results = []
    results_classification = []
    results_completion = []
    intersections, unions, categories = [], [], []

    categories_summary = {k:[] for k in loader.dataset.idx2cat.keys()}
    categories_encoding = {v:[] for v in loader.dataset.idx2cat.values()}
    idx2cat = loader.dataset.idx2cat

    encoding_norm = []
    
    for _ in range(1):
        for j, data in enumerate(tqdm(loader)):
            data = data.to(device)

            if args.dataset == 'c3d':
                pos, batch, pc_label, category = data.pos, data.batch, data.y, data.category
            elif args.dataset == 'm40':
                pos, batch, pc_label, category = data.pos, data.batch, data.pos, data.y
            elif args.dataset == 'snet':
                pos, batch, pc_label, category = data.pos, data.batch, data.pos, data.y
            elif args.dataset == 'mvp':
                pos, batch, pc_label, category = data.pos, data.batch, data.y, data.category
            else:
                raise ValueError('{} dataset has not been supported yet.'.format(args.dataset))
    
            with torch.no_grad():
                model(None, pos, batch)
                if args.uncertainty_flag:
                    model.compute_loss(category, pc_label, model.w1, model.w2)
                else:
                    model.compute_loss(category, pc_label)
                
                encoding = model.encoding_feature.cpu().detach().numpy()
                encoding_norm.append(np.linalg.norm(encoding, axis=-1))
                pos = pos.cpu().detach().numpy().reshape(-1, 2048*3)
                pc_label = pc_label.cpu().detach().numpy().reshape(-1, 2048*3)
                out_info = np.concatenate([encoding, pos, pc_label], axis=-1)
                if 'completion' in args.task:
                    if args.completion_decoder_choice=='folding':
                        pc_pred = model.pred_completion.cpu().detach().numpy().reshape(-1, 2025*3)
                    else:
                        pc_pred = model.pred_completion.cpu().detach().numpy().reshape(-1, 2048*3)
                    out_info = np.concatenate([encoding, pos, pc_label, pc_pred], axis=-1)

            if 'completion' in args.task:
                results_completion.append(model.loss_completion)
            if 'classification' in args.task:
                pred = model.pred_classification
                results_classification.append(pred.eq(category).float())

            category = category.cpu().detach().numpy()
            for idx in range(category.shape[0]):
                categories_encoding[idx2cat[category[idx]]].append(out_info[idx])

            categories.append(category)

    categories = np.concatenate(categories, axis=0)
    print('Encoding norm avg:', np.stack(encoding_norm).mean())

    if 'completion' in args.task:
        results = torch.cat(results_completion, dim=0)
        for i in range(len(categories)):
            categories_summary[categories[i]].append(results[i])
        total_chamfer_distance = 0
        for idx in categories_summary:
            chamfer_distance_cat = torch.stack(categories_summary[idx], dim=0).mean().item()
            total_chamfer_distance += chamfer_distance_cat
            print('{}: {:.7f}'.format(idx2cat[idx], chamfer_distance_cat))
        print('Mean Class Chamfer Distance: {:.6f}'.format(total_chamfer_distance/len(categories_summary)))

    if 'classification' in args.task:
        results = torch.cat(results_classification, dim=0).mean().item()
        print('Test Acc: {:.4f}'.format(results))

    for cat in categories_encoding:
        encodings = np.concatenate(categories_encoding[cat], 0)
        np.save(os.path.join(save_dir, '{}.npy'.format(cat)), encodings)

    print('Sample results are saved to: {}'.format(save_dir))
    print('{} point clouds are evaluated.'.format(len(loader.dataset)))


def load_dataset(args):

    if args.dataset == 'c3d':
        pre_transform, transform = None, None
        categories ='plane,cabinet,car,chair,lamp,couch,table,watercraft'
        categories = categories.split(',')
        train_dataset = completion3D_class('data_root/Completion3D', categories, split='train',
                            include_normals=False, pre_transform=pre_transform, transform=transform)
        test_dataset = completion3D_class('data_root/Completion3D', categories, split='val',
                            include_normals=False, pre_transform=pre_transform, transform=transform)

    elif args.dataset == 'm40':
        pre_transform, transform = T.NormalizeScale(), T.SamplePoints(2048)
        train_dataset = ModelNet('data_root/ModelNet40', name='40', train=True,
                                 pre_transform=pre_transform, transform=transform)
        test_dataset = ModelNet('data_root/ModelNet40', name='40', train=False,
                                 pre_transform=pre_transform, transform=transform)

    elif args.dataset == 'snet':
        pre_transform, transform = T.NormalizeScale(), T.FixedPoints(2048)
        train_dataset = ShapeNet('data_root/ShapeNet', split='trainval', include_normals=False,
                                 pre_transform=pre_transform, transform=transform)
        test_dataset = ShapeNet('data_root/ShapeNet', split='test', include_normals=False,
                                pre_transform=pre_transform, transform=transform)

    elif args.dataset == 'mvp':
        train_dataset = MVP('data_root/MVP', split='train', npoints=2048)
        test_dataset = MVP('data_root/MVP', split='test', npoints=2048)

    train_dataloader = DataLoader(train_dataset, batch_size=args.bsize, shuffle=True,
                                  num_workers=6, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bsize, shuffle=False,
                                 num_workers=6, drop_last=False)

    return train_dataloader, test_dataloader


def create_model_name(args):
    model_name = args.dataset
    for task, name in [('classification','cls'), ('segmentation','seg'), ('completion', 'pc')]:
        if task in args.task:
            model_name += '_{}'.format(name)
    model_name += '_b{}ep{}lr{}s{}g{}'.format(args.bsize, args.max_epoch,
            args.lr, args.lr_step_size, args.lr_gamma)
    model_name += '_{}'.format(args.encoder_choice)

    model_name += '_HyperModule{}'.format(args.use_hyperspherical_module)
    if args.use_hyperspherical_module:
        model_name += '_HyperEncode{}'.format(args.use_hyperspherical_encoding)
    model_name += '_MaxPool{}'.format(args.maxpool_bottleneck)
    if args.use_hyperspherical_module:
        model_name += '-Hyper{}'.format(args.hyper_bottleneck)
        
    model_name += '-NormOrder{}'.format(args.norm_order)
    model_name += '-LayerNum{}'.format(args.hyperspherical_module_layers)
    model_name += '-HyperModuleBN{}'.format(args.hyperspherical_module_BN)

    if 'completion' in args.task:
        model_name += '_{}'.format(args.completion_decoder_choice)

    if 'classification' in args.task:
        model_name += '_Classifier{}BN{}'.format(args.mlps_classifier.replace(',', '-'), args.use_BN_classifier)

    if 'segmentation' in args.task:
        model_name += '_Seg{}BN{}'.format(args.mlps_segmentator.replace(',', '-'), args.use_BN_segmentator)

    if args.use_hyperspherical_module \
            and args.use_hyperspherical_encoding \
            and args.weight_sec_loss is not None:
        model_name += '_{}SecLoss'.format(args.weight_sec_loss)

    if args.grad_surgey_flag:
        model_name += '_grad_surgey'

    if args.uncertainty_flag:
        model_name += '_uncertainty'
    
    if args.optimal_search:
        model_name += '_search'
        model_name += str(args.ratio)
    
    if args.compute_gradient_norm and not args.grad_surgey_flag:
        model_name += '_gradient'

    return model_name


def check_overwrite(args):
    model_name = args.model_name
    check_dir = '{}/{}'.format(args.check_dir, model_name)
    log_dir = '{}/{}'.format(args.log_dir, model_name)
    if os.path.exists(check_dir) or os.path.exists(log_dir):
        valid = ['y', 'yes', 'no', 'n']
        inp = None
        while inp not in valid:
            inp = input('{} already exists. Do you want to overwrite it? (y/n)'.format(model_name))
            if inp.lower() in ['n', 'no']:
                raise Exception('Please create new experiment.')
            
        # remove the existing dir if overwriting.
        if os.path.exists(check_dir):
            shutil.rmtree(check_dir)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

        # waiting for updating of tensorboard
        time.sleep(2)

    # create directory
    os.makedirs(check_dir)
    os.makedirs(log_dir)
    return check_dir, log_dir


def backup(log_dir, parser):
    shutil.copyfile('main.py', os.path.join(log_dir, 'main.py'))
    shutil.copytree('utils', os.path.join(log_dir, 'utils'))

    file = open(os.path.join(log_dir, 'parameters.txt'), 'w')
    adict = vars(parser.parse_args())
    keys = list(adict.keys())
    keys.sort()
    for item in keys:
        file.write('{0}:{1}\n'.format(item, adict[item]))
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', help='path to config file', required=True)
    args = parser.parse_args()

    seed_everything()

    with open('./cfgs/config.yaml') as f:
        config = yaml.safe_load(f)

    print('\n**************************\n')
    for k, v in config.items():
        setattr(args, k, v)
        print('\t{}:{}'.format(k, v))
    print('\n**************************\n')

    args.task = args.task.split(',')
    for item in args.task:
        assert item in ['completion', 'classification', 'segmentation']
    args.model_name = create_model_name(args)
    print('Model name: {}'.format(args.model_name))

    # construct data loader
    train_dataloader, test_dataloader = load_dataset(args)

    model = Model(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # evaluation
    if args.eval:
        model_path = os.path.join(args.check_dir, args.model_name)
        if model_path.endswith('.pth'):
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')))
        print('Successfully load model from: {}'.format(model_path))

        save_dir = os.path.join(model_path, 'eval_sample_results')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        evaluate(args=args, loader=test_dataloader, save_dir=save_dir)

    # training
    else:
        if args.pretrained_path is None:
            print('\nStart training!\n')
            train(args=args, train_dataloader=train_dataloader, test_dataloader=test_dataloader)
            print('Training is done: {}'.format(args.model_name))

        else:
            pretrained_dict = torch.load(args.pretrained_path)
            print('Successfully load pretrained.')
            model.load_state_dict(pretrained_dict)
            train(args=args, train_dataloader=train_dataloader, test_dataloader=test_dataloader)

