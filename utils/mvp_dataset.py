# @file mvp_dataset.py
# @author Junming Zhang, junming@umich.edu;
# @brief MVP class
# @copyright Copyright University of Michigan, Ford Motor Company (c) 2020-2021

import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset


def random_rotation(data):
    pos = data.pos
    degree = 2 * np.pi * np.random.uniform()
    sin, cos = np.sin(degree), np.cos(degree)
    matrix = torch.Tensor([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
    data.pos = torch.matmul(pos, matrix)
    return data


class MVP(InMemoryDataset):

    def __init__(self, root, split='train', npoints=2048, novel_input=True,
            pre_transform=None, transform=None):
        self.root = root
        self.split = split
        self.npoints = npoints
        self.novel_input = novel_input

        self.idx2cat = {
            0: 'airplane',
            1: 'cabinet',
            2: 'car',
            3: 'chair',
            4: 'lamp',
            5: 'sofa',
            6: 'table',
            7: 'watercraft',
            8: 'bed',
            9: 'bench',
            10: 'bookshelf',
            11: 'bus',
            12: 'guitar',
            13: 'motorbike',
            14: 'pistol',
            15: 'skateboard',
        }

        super().__init__(root, transform=transform, pre_transform=pre_transform, pre_filter=None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_names = [
            'mvp_{}_gt_{}pts.h5'.format(self.split, self.npoints),
            'mvp_{}_input.h5'.format(self.split)]
        return file_names

    @property
    def processed_file_names(self):
        return ['{}.pt'.format(self.split)]

    def process(self):

        data_list = []

        print('Processing {} data...'.format(self.split))
        self.input_path = os.path.join(self.root, 'raw/mvp_{}_input.h5'.format(self.split))
        self.gt_path = os.path.join(self.root, 'raw/mvp_{}_gt_{}pts.h5'.format(self.split, self.npoints))

        input_file = h5py.File(self.input_path, 'r')
        gt_file = h5py.File(self.gt_path, 'r')

        input_data = torch.tensor((input_file['incomplete_pcds'][()]), dtype=torch.float32)
        novel_input_data = torch.tensor((input_file['novel_incomplete_pcds'][()]), dtype=torch.float32)
        labels = torch.tensor((input_file['labels'][()]), dtype=torch.long)
        novel_labels = torch.tensor((input_file['novel_labels'][()]), dtype=torch.long)
        gt_data = torch.tensor((gt_file['complete_pcds'][()]), dtype=torch.float32)
        novel_gt_data = torch.tensor((gt_file['novel_complete_pcds'][()]), dtype=torch.float32)

        input_file.close()
        gt_file.close()

        if self.novel_input:
            input_data = torch.cat((input_data, novel_input_data), axis=0)
            gt_data = torch.cat((gt_data, novel_gt_data), axis=0)
            labels = torch.cat((labels, novel_labels), axis=0)

        for idx in tqdm(range(input_data.shape[0])):
            data = Data(pos=input_data[idx], y=gt_data[idx//26], category=labels[idx])
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


if __name__ == "__main__":
    root = '../data_root/MVP/'
    dataset = MVP(root, split='train')

    data = dataset[10]
    print(data.pos.size())
    print(data.y.size())
    print(data.category.size())
    print(data.category)