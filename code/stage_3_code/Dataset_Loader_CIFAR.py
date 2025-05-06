from code.base_class.dataset import dataset
import pickle

class Dataset_Loader_CIFAR(dataset):
    def __init__(self):
        super().__init__('CIFAR', '')
        self.dataset_source_file_path = 'data/stage_3_data/CIFAR'

    def load(self):
        with open(self.dataset_source_file_path, 'rb') as f:
            data = pickle.load(f)
        return data
