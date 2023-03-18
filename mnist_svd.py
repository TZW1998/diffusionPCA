import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
import pickle
import numpy as np

class MnistSVD(Dataset):
    def __init__(self, image_paths, train = True, train_preprocess = False):
        prefix = 'train' if train else 'test'
        image_data_paths = os.path.join(image_paths, f'{prefix}_data.pkl')

        if not os.path.exists(image_data_paths):
            transform = transforms.Compose([transforms.ToTensor()])
            mnist = datasets.MNIST(root=image_paths, train=train, download=True, transform=transform)
            print(f"Calculating SVD of {prefix} MNIST images...")
            self.mnist_data = [(torch.linalg.svd(image[0].float(), full_matrices=False), label) for image, label in mnist]
            print("Done!")
            with open(image_data_paths, 'wb') as f:
                pickle.dump(self.mnist_data, f)
        else:
            with open(image_data_paths, 'rb') as f:
                self.mnist_data = pickle.load(f)
            print("Done!")
        
        self.train_preprocess = train_preprocess

        tau = 9
        p = np.exp((28 - np.arange(1,28))/tau)
        self.sample_prob = p/p.sum()
        
    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        if self.train_preprocess:
            use_rank = np.random.choice(np.arange(1,28),p=self.sample_prob)
            (U, S, Vh), label = self.mnist_data[idx]
            inp = U[:,:use_rank] @ torch.diag(S[:use_rank]) @ Vh[:use_rank,:]
            out = U[:,use_rank:] @ torch.diag(S[use_rank:]) @ Vh[use_rank:,:]

            inp += torch.randn_like(inp) * 0.1

            new_U, new_S, new_Vh = torch.linalg.svd(inp, full_matrices=False)
            inp = new_U[:,:use_rank] @ torch.diag(new_S[:use_rank]) @ new_Vh[:use_rank,:]
            inp *= S[:use_rank].sum() / new_S[:use_rank].sum() 
            
            return inp.flatten(), out.flatten(), np.float32(use_rank), label
        else:
            image_svd, label = self.mnist_data[idx]
            return *image_svd, label
    
# mnist_train = MnistSVD("data")