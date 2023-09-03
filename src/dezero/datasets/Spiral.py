from .Dataset import Dataset
from .toy_dataset import get_spiral


class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)
    
