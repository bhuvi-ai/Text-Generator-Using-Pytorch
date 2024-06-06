from typing import Any
import pytorch_lightning as pl
import torch.nn as nn
from data import CharTokenizer
import torch
import typing
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint




class Generator(pl.LightningModule):
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size : int,tokenizer: CharTokenizer):
        
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        self.emb_layer = nn.Embedding(vocab_size, embedding_dim)
        self.rnn_layer = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.out_layer = nn.Linear(hidden_size, vocab_size)
        self.tokenizer = tokenizer
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token)
        
    def forward(self,encoded: torch.LongTensor, hidden: torch.LongTensor = None) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        #----------------------------------
        encoded = encoded.to(self.device) 
        ##############
        emb = self.emb_layer(encoded)
        rnn_out, hidden = self.rnn_layer(emb, hidden)
        out = self.out_layer(rnn_out)
        
        return out, hidden
    
    
    def prepend_bos(self,batch: torch.LongTensor) -> torch.LongTensor:
        bs = batch.shape[0]
        bos_tokens = torch.full((bs, 1), self.tokenizer.bos_token, device=batch.device)
        output = torch.cat((bos_tokens, batch), dim=1)[:, :-1]
        return output
    
    def training_step(self, batch: torch.LongTensor, batch_idx: int) -> torch.Tensor:
        input = self.prepend_bos(batch)
        out, _ = self(input)
        loss = self.loss_fn(out.transpose(2,1),batch)
        self.log('loss',loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: torch.LongTensor, batch_idx: int) -> torch.Tensor:
        input = self.prepend_bos(batch)
        out, _ = self(input)
        loss = self.loss_fn(out.transpose(2,1),batch)
        self.log('val_loss',loss, prog_bar=True)
    
    
    def test_step(self, batch: torch.LongTensor, batch_idx: int) -> torch.Tensor:
        input = self.prepend_bos(batch)
        out, _ = self(input)
        loss = self.loss_fn(out.transpose(2,1),batch)
        self.log('test_loss',loss, prog_bar=True)
    
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)
    
    
    def generate(self,prompt: str , n_tokens :int = 200) -> str:
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # encoded_prompt = self.tokenizer.encode(prompt, add_bos_token=True)
            encoded_prompt = torch.tensor(self.tokenizer.encode(prompt, add_bos_token=True)).to(self.device)  # Ensure input tensor is on the correct device

            out, hidden = self(encoded_prompt.unsqueeze(0))
            # out = out[-1:]        
            out = out[:, -1, :]

            next_token = torch.distributions.Categorical(out.softmax(-1)).sample()
            generated_tokens = [next_token]
            
            for _ in range(n_tokens - 1):
                next_token = next_token.unsqueeze(0).to(self.device)
                out, hidden = self(next_token, hidden)
                out = out[:, -1, :]
                next_token = torch.distributions.Categorical(out.softmax(-1)).sample()
                generated_tokens.append(next_token)
            
            generated_tokens = torch.cat(generated_tokens, dim=0)
            return self.tokenizer.decode(generated_tokens.tolist())
    
    
    
    
if __name__ == '__main__':
    
    tokenizer = CharTokenizer.load('tokenizer.json')
    generator = Generator(tokenizer.vocab_size,4,5, tokenizer)
    
    fake_batch = torch.randint(0, tokenizer.vocab_size - 1, (3,17))
    
    out = generator(fake_batch)
    
    # print(out)
    # print(generator.loss_fn(out[0].transpose(2,1), fake_batch))
    print("--------------------------------------------------------")
    # print(out[0].shape)
    from data import CharDatamodule,load_data
    
    data = load_data()
    tokenizer = CharTokenizer.load('tokenizer.json')
    
    datamodule = CharDatamodule(data, tokenizer)
    
#     generator = Generator(tokenizer.vocab_size, 128,512, tokenizer)
    
    
#     checkpoint_callback = ModelCheckpoint(
#     monitor='val_loss', 
#     dirpath='checkpoints/',  
#     filename='best-checkpoint',  
#     save_top_k=1,  
#     mode='min'  
# )
#     # trainer = pl.Trainer(max_epochs=30)
#     trainer = pl.Trainer(
#         max_epochs=20,
#         callbacks=[checkpoint_callback]
#     )    

#     trainer.fit(model=generator, datamodule=datamodule)

#     print(f"Best model saved at: {checkpoint_callback.best_model_path}")

#Best Model Load For making Sentence generation
# print('------------Best CheckPoint-----------------')
    generator = Generator.load_from_checkpoint("checkpoints/best-checkpoint.ckpt", tokenizer=tokenizer)


# print(generator)

# print('-----------Metrics on Test Data-----------------')
# trainer = pl.Trainer(max_epochs=1)

# trainer.fit(model=generator, datamodule=datamodule)
# trainer.test(model=generator , datamodule=datamodule)

# print('-------------------------GPU CPU Issue----------------------------')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)



    prompt= 'I want say'

    # encoded = tokenizer.encode(prompt,add_bos_token=True)

    # encoded_tensor = torch.tensor(encoded).to(device)



    # out, _ = generator(encoded_tensor)

    # print(out)

    output = generator.generate(prompt)
    print('------------------Prompt Testing------------------------------')

    print(output)