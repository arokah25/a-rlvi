import numpy as np
import torch
import torch.utils.data as Data
from PIL import Image
import os
import sys
import errno
from torchvision import datasets, transforms


if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import data_tools as tools

# --- data_load.py ----------------------------------------------------
class Food101(torch.utils.data.Dataset):
    """
    Food-101 loader that respects the official noisy-train / clean-test split.

    Args
    ----
    split : {"train", "val", "test"}
        * "train" → first `split_per` fraction of the 75 750 noisy training imgs
        * "val"   → remaining fraction of the noisy training imgs
        * "test"  → 25 250 clean test imgs (always the same, no noise)
    """
    def __init__(self, root, split="train", transform=None,
                 split_per=0.75, random_seed=1, stratified=False, download=True):

        from torchvision.datasets import Food101 as TorchFood101
        assert split in {"train", "val", "test"}
        self.transform = transform

        base_split = "train" if split in {"train", "val"} else "test"
        full_ds = TorchFood101(root, split=base_split, download=download)

        # be robust to torchvision version differences
        if hasattr(full_ds, "_image_files") and hasattr(full_ds, "_labels"):
            self.images = full_ds._image_files
            self.labels = np.array(full_ds._labels, dtype=int)
        elif hasattr(full_ds, "imgs"):  # list[(path, label)]
            pairs = full_ds.imgs
            self.images = [p for p, _ in pairs]
            self.labels = np.array([y for _, y in pairs], dtype=int)
        else:
            raise RuntimeError("Unsupported torchvision.Food101 structure")

        rng = np.random.default_rng(random_seed)

        if split in {"train", "val"} and 0. < split_per < 1.:
            if stratified:
                train_idx, val_idx = [], []
                for c in np.unique(self.labels):
                    idx_c = np.where(self.labels == c)[0]
                    rng.shuffle(idx_c)
                    cut = int(len(idx_c) * split_per)
                    train_idx.append(idx_c[:cut])
                    val_idx.append(idx_c[cut:])
                train_idx = np.concatenate(train_idx)
                val_idx   = np.concatenate(val_idx)
                self.indices = train_idx if split == "train" else val_idx
            else:
                idx = np.arange(len(self.images))
                rng.shuffle(idx)
                cut = int(len(idx) * split_per)
                self.indices = idx[:cut] if split == "train" else idx[cut:]
        else:
            self.indices = np.arange(len(self.images))
    # ------------------------------------------------------------
    
    def __getitem__(self, i):
        real_idx = self.indices[i]
        img  = Image.open(self.images[real_idx]).convert("RGB")
        lbl  = self.labels[real_idx]
        if self.transform: img = self.transform(img)
        return img, lbl, i, real_idx         



    def __len__(self):
        return len(self.indices)


class Mnist(torch.utils.data.Dataset):
    def __init__(self, root, train=True, download=True, transform=None, target_transform=None,
                 dataset=None, noise_type=None, noise_rate=None, split_per=0.9, random_seed=1):

        if transform is None:
            transform = transforms.ToTensor()
        if target_transform is None:
            target_transform = lambda y: y

        raw_dataset = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform,
            target_transform=target_transform
        )

        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # Extract data and labels
        images = raw_dataset.data
        labels = np.array(raw_dataset.targets)

        # Apply label noise if requested
        if noise_rate > 0:
            data_split = tools.dataset_split(
                images, labels, dataset, noise_type, noise_rate,
                split_per, random_seed, 10
            )
        else:
            data_split = tools.dataset_split_without_noise(
                images, labels, split_per, random_seed
            )

        self.train_data, self.val_data, self.train_labels, self.val_labels, self.noise_mask = data_split

    def __getitem__(self, index):
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]

        img = Image.fromarray(img.numpy())
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label, index

    def __len__(self):
        return len(self.train_data) if self.train else len(self.val_data)

class MnistTest(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        if transform is None:
            transform = transforms.ToTensor()
        if target_transform is None:
            target_transform = lambda y: y

        self.dataset = datasets.MNIST(
            root=root,
            train=False,
            download=True,
            transform=transform,
            target_transform=target_transform
        )

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, index

    def __len__(self):
        return len(self.dataset)
  
    
class Cifar10(Data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,
                 dataset='cifar10', noise_type='symmetric', noise_rate=0.5, 
                 split_per=0.9, random_seed=1, num_class=10
                ):
        self.root = os.path.join(os.path.expanduser(root), 'CIFAR')
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        self.train_data = []
        self.train_labels = []
        for fentry in self.train_list:
            f = fentry[0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.train_data.append(entry['data'])
            if 'labels' in entry:
                self.train_labels += entry['labels']
            else:
                self.train_labels += entry['fine_labels']
            fo.close()

        self.train_data = np.concatenate(self.train_data)
        self.train_labels = np.array(self.train_labels)
        # self.train_data = self.train_data.reshape((50000, 3, 32, 32))
        # self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

        # clean images and noisy labels (training and validation)
        if noise_rate > 0:
            self.train_data, self.val_data, self.train_labels, self.val_labels, self.noise_mask = tools.dataset_split(
                self.train_data, self.train_labels, dataset, noise_type, noise_rate, split_per, random_seed, num_class
            )
        else:
            self.train_data, self.val_data, self.train_labels, self.val_labels, self.noise_mask = tools.dataset_split_without_noise(
                self.train_data, self.train_labels, split_per, random_seed
            )

        if self.train:      
            self.train_data = self.train_data.reshape((-1, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))
        
        else:
            self.val_data = self.val_data.reshape((-1, 3, 32, 32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))
        
    def __getitem__(self, index):
           
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
            
        else:
            img, label = self.val_data[index], self.val_labels[index]
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.val_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not tools.check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        tools.download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

        
class Cifar10Test(Data.Dataset):
    base_folder = 'cifar-10-batches-py'
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.join(os.path.expanduser(root), 'CIFAR')
        self.transform = transform
        self.target_transform = target_transform

        f = self.test_list[0][0]
        file = os.path.join(self.root, self.base_folder, f)
        fo = open(file, 'rb')
        if sys.version_info[0] == 2:
            entry = pickle.load(fo)
        else:
            entry = pickle.load(fo, encoding='latin1')
        self.test_data = entry['data']
        if 'labels' in entry:
            self.test_labels = entry['labels']
        else:
            self.test_labels = entry['fine_labels']
        fo.close()
        self.test_data = self.test_data.reshape((-1, 3, 32, 32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1))
        self.test_labels = np.array(self.test_labels)

    def __getitem__(self, index):
        img, label = self.test_data[index], self.test_labels[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index
    
    def __len__(self):
        return len(self.test_data)

    
class Cifar100(Data.Dataset):

    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,
                 dataset='cifar100', noise_type='symmetric', noise_rate=0.5, 
                 split_per=0.9, random_seed=1, num_class=100
                ):
        self.root = os.path.join(os.path.expanduser(root), 'CIFAR')
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        self.train_data = []
        self.train_labels = []
        for fentry in self.train_list:
            f = fentry[0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.train_data.append(entry['data'])
            if 'labels' in entry:
                self.train_labels += entry['labels']
            else:
                self.train_labels += entry['fine_labels']
            fo.close()

        self.train_data = np.concatenate(self.train_data)
        self.train_labels = np.array(self.train_labels)

        # clean images and noisy labels (training and validation)
        if noise_rate > 0:
            self.train_data, self.val_data, self.train_labels, self.val_labels, self.noise_mask = tools.dataset_split(
                self.train_data, self.train_labels, dataset, noise_type, noise_rate, split_per, random_seed, num_class
            )
        else:
            self.train_data, self.val_data, self.train_labels, self.val_labels, self.noise_mask = tools.dataset_split_without_noise(
                self.train_data, self.train_labels, split_per, random_seed
            )

        if self.train:      
            self.train_data = self.train_data.reshape((-1, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1)) 
        
        else:
            self.val_data = self.val_data.reshape((-1, 3, 32, 32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]   
        else:
            img, label = self.val_data[index], self.val_labels[index]
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.val_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not tools.check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        tools.download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
        
        
class Cifar100Test(Data.Dataset):

    base_folder = 'cifar-100-python'
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.join(os.path.expanduser(root), 'CIFAR')
        self.transform = transform
        self.target_transform = target_transform

        f = self.test_list[0][0]
        file = os.path.join(self.root, self.base_folder, f)
        fo = open(file, 'rb')
        if sys.version_info[0] == 2:
            entry = pickle.load(fo)
        else:
            entry = pickle.load(fo, encoding='latin1')
        self.test_data = entry['data']
        if 'labels' in entry:
            self.test_labels = entry['labels']
        else:
            self.test_labels = entry['fine_labels']
        fo.close()
        self.test_data = self.test_data.reshape((-1, 3, 32, 32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1))
        self.test_labels = np.array(self.test_labels)

    def __getitem__(self, index):
        img, label = self.test_data[index], self.test_labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index
    
    def __len__(self):
        return len(self.test_data)
