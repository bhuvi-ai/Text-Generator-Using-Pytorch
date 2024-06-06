from datasets  import load_dataset
import typing
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset , DataLoader
import json
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence


def load_data(n: int = 1_000_000) -> typing.List[str]:
    data = load_dataset('bookcorpus',split='train')[:n]
    return data['text']



def find_character(data: typing.List[str]) -> typing.List[str]:
    characters = set()
    for sentence in data:
        characters.update(set(sentence))
    return sorted(list(characters))

class CharTokenizer:
    
    def __init__(self , characters: typing.List[str]):
        self.characters = characters
        
        self.pad_token = 0
        self.bos_token = 1
        self.unk_token = 2
        
        self.vocab_size = len(characters) + 3
        
    def encode(self, sentence: str, add_bos_token: bool = False) -> torch.LongTensor:
        encoded = []
        if add_bos_token:
            encoded.append(self.bos_token)
        for char in sentence:
            if char not in self.characters:
                encoded.append(self.unk_token)
            else:
                encoded.append(self.characters.index(char) + 3)
        return torch.LongTensor(encoded)


    def decode(self, encoded: torch.LongTensor) -> str:
        output = ''
        for idx in encoded:
            if idx < 3:
                continue
            
            char = self.characters[idx - 3]
            output = output + char
        return output

    
    
    def save(self, path: str):
        with open(path,'w') as file:
            json.dump(self.characters, file)
    
    @staticmethod
    def load(path: str) -> 'CharTokenizer':
        with open(path, 'r') as file:
            characters = json.load(file)
        return CharTokenizer(characters)


class CharDataset(Dataset):
    
    def __init__(self , data: typing.List[str], tokenizer: CharTokenizer):
        super().__init__()   
        self.data = data
        self.tokenizer = tokenizer
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index:int) -> torch.LongTensor:
        sentence = self.data[index]
        encoded = self.tokenizer.encode(sentence)
        return encoded
    
    
    
class CharDatamodule(pl.LightningDataModule):
    
    def __init__(self, data: typing.List[str], tokenizer:CharTokenizer, batch_size: int = 128):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        
        
        train_data, val_data, test_data = self.spilt(data)
        
        self.train_dataset = CharDataset(train_data, tokenizer)
        self.val_dataset = CharDataset(val_data, tokenizer)
        self.test_dataset = CharDataset(test_data, tokenizer)
    
    def collate_fn(self, samples: typing.List[torch.LongTensor]) -> torch.LongTensor:
        return pad_sequence(samples, batch_first=True , padding_value=self.tokenizer.pad_token)        
    
    
    def common_dataloader(self , split:str) -> DataLoader:
        dataset = getattr(self, f'{split}_dataset')
        return DataLoader(dataset , batch_size=self.batch_size, shuffle=(split=='train'), collate_fn=self.collate_fn)
    
    
       
    def train_dataloader(self) -> DataLoader:
        return self.common_dataloader('train')
    
    def val_dataloader(self) -> DataLoader:
        return self.common_dataloader('val')

    
    def test_dataloader(self) -> DataLoader:
        return self.common_dataloader('test')

    
    def spilt(self, data: typing.List[str]) -> typing.Tuple[typing.List[str], typing.List[str], typing.List[str]]:
        n_train = int(len(data) * 0.8)
        n_val = int(len(data) * 0.1)
        train_data = data[:n_train]
        val_data = data[n_train:n_train+n_val] 
        test_data = data[n_train+n_val:]
        return train_data,val_data,test_data
         
    
    
    
    
if __name__ == '__main__':
    data = load_data()
    
    # file_path = 'bookcorpus.txt'
    # with open(file_path, 'w') as f:
    #     for line in data:
    #         f.write(line + '\n')
    
    # print(f"Data saved to {file_path}")
    # print(data)
    
    characters = find_character(data)
    tokenizer = CharTokenizer(characters)
    # sentence = 'i like coding'
    # encoded = tokenizer.encode(sentence)
    
    
    # decoded = tokenizer.decode(encoded)
    # print(encoded)
    # print(decoded)
    
    # tokenizer.save('tokenizer.json')
    # loaded_tokenizer = CharTokenizer.load('tokenizer.json')
    
    # print(loaded_tokenizer)
    
    # dataset = CharDataset(data , tokenizer)
    
    # print(dataset)
    datamodule = CharDatamodule(data, tokenizer)
    
    datamodule.setup('fit')
    
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    #size of the training, validation, and test datasets
    print(f"Training data size: {len(datamodule.train_dataset)} samples")
    print(f"Validation data size: {len(datamodule.val_dataset)} samples")
    print(f"Test data size: {len(datamodule.test_dataset)} samples")

    #number of batches for training, validation, and test datasets
    train_batches = len(train_loader)
    val_batches = len(val_loader)
    test_batches = len(test_loader)

    print(f"Number of training batches: {train_batches}")
    print(f"Number of validation batches: {val_batches}")
    print(f"Number of test batches: {test_batches}")

    # first batch from the training data
    batch = next(iter(train_loader))
    print("First batch from training data:")
    print(batch)
    
    print(datamodule)
    print(DataLoader)