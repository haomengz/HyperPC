# @file completion3D_dataset.py
# @author Junming Zhang, junming@umich.edu; Haomeng Zhang, haomeng@umich.edu
# @brief completion3D dataset class
# @copyright Copyright University of Michigan, Ford Motor Company (c) 2020-2021

import os
import os.path as osp
import shutil
import h5py
import torch

from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
from torch_geometric.io import read_txt_array


class completion3D_class(InMemoryDataset):
    """The Completion3D benchmark is a platform for evaluating state-of-the-art 3D 
    Object Point Cloud Completion methods. Participants are given a partial 3D object 
    point cloud and tasked to infer a complete 3D point cloud for the object.
    <https://completion3d.stanford.edu/> 
    Completion3D dataset contains 3D shape point clouds of 8 shape categories.

    Args:
        root (string): Root directory where the dataset should be saved.
        categories (string or [string], optional): The category of the CAD
            models (one or a combination of :obj:`"plane"`, :obj:`"cabinet"`,
            :obj:`"car"`, :obj:`"chair"`, :obj:`"lamp"`, :obj:`"couch"`,
            :obj:`"table"`, :obj:`"watercraft"`).
            Can be explicitly set to :obj:`None` to load all categories.
            (default: :obj:`None`)
        include_normals (bool, optional): If set to :obj:`False`, will not
            include normal vectors as input features. (default: :obj:`True`)
        split (string, optional): 
            If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = ('http://download.cs.stanford.edu/downloads/completion3d/'
            'dataset2019.zip')

    category_ids = {
        'plane': '02691156',
        'cabinet': '02933112',
        'car': '02958343',
        'chair': '03001627',
        'lamp': '03636649',
        'couch': '04256520',
        'table': '04379243',
        'watercraft': '04530566',
    }

    def __init__(self, root, categories=None, include_normals=True,
                 split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.category_ids for category in categories)

        self.categories = categories
        categories_ids = [self.category_ids[cat] for cat in self.categories]
        cat_idx = {categories_ids[i]: i for i in range(len(categories_ids))}
        ids_category = {v:k for k,v in self.category_ids.items()}
        self.idx2cat = {v:ids_category[k] for k,v in cat_idx.items()}
        self.split = split
        super(completion3D_class, self).__init__(root, transform, pre_transform,
                                       pre_filter)

        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        else:
            raise ValueError((f'Split {split} found, but expected either '
                              'train, val, trainval or test'))

        self.data, self.slices = torch.load(path)
        self.data.x = self.data.x if include_normals else None

    @property
    # all the folder names except xxx.txt
    def raw_file_names(self):
        # return list(self.category_ids.values()) + ['train_test_split']
        return list(['train', 'val', 'train.list', 'val.list'])

    @property
    # naming the pt files, eg : cha_air_car_test.pt, cha_air_car_train.pt
    def processed_file_names(self):
        cats = '_'.join([cat[:3].lower() for cat in self.categories])
        return [
            os.path.join('{}_{}.pt'.format(cats, split))
            for split in ['train', 'val']
        ]

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        name = 'shapenet'
        os.rename(osp.join(self.root, name), self.raw_dir)

    def process_filenames(self, filenames, split_in_loop):
        data_list = []
        # categories_ids is the IDs of the categories user selected 
        # eg: ['02691156', '02933112', '02958343', '03001627',
        # '03636649', '04256520', '04379243', '04530566']
        categories_ids = [self.category_ids[cat] for cat in self.categories]

        # cat_idx: {ID of the categores -> index}
        # eg: {'02691156': 0, '02933112': 1, '02958343': 2, '03001627': 3,
        # '03636649': 4, '04256520': 5, '04379243' : 6, '04530566': 7}
        cat_idx = {categories_ids[i]: i for i in range(len(categories_ids))}

        self.idx2cat = {k:v for k,v in cat_idx.items()}

        #name : the path of the point clouds
        # eg: 04530566/786f18c5f99f7006b1d1509c24a9f631
        # the point clouds is saves in train/04530566/786f18c5f99f7006b1d1509c24a9f631.h5

        #name.split(osp.sep) : ['04530566', '786f18c5f99f7006b1d1509c24a9f631']
        for name in filenames:
            cat = name.split(osp.sep)[0]

            if split_in_loop == 'train' or split_in_loop == 'val':
                if cat not in categories_ids:
                    continue

            fpos = h5py.File(osp.join(osp.join(self.raw_dir,
                            '{}/partial'.format(split_in_loop)), name), 'r')
            pos = torch.tensor(fpos['data'], dtype=torch.float32)
            fy = h5py.File(osp.join(osp.join(self.raw_dir,
                            '{}/gt'.format(split_in_loop)), name), 'r')
            y = torch.tensor(fy['data'], dtype=torch.float32)
            data = Data(pos=pos, y=y, category=cat_idx[cat])

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list


    def process(self):
        trainval = []
        for i, split in enumerate(['train', 'val']):
            path = osp.join(self.raw_dir, f'{split}.list')
            with open(path, 'r') as f:
                tmp = ".h5"
                filenames = [
                    (name[0: -1] + tmp)
                    for name in f
                ]
            data_list = self.process_filenames(filenames, split)
            torch.save(self.collate(data_list), self.processed_paths[i])

    def __repr__(self):
        return '{}({}, categories={})'.format(self.__class__.__name__,
                                              len(self), self.categories)
