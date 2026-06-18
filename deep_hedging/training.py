import torch
from tqdm.auto import tqdm


def get_optimizer(model):
    optimizer= torch.optim.AdamW(model.parameters() , lr=1e-2 , weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer , T_max=64*157)
    return optimizer , scheduler

def training_loop(model , dataloader , batch_size , optimizer , scheduler , num_epochs , device):
  train_states , train_dS , train_payoff = next(iter(dataloader))
  J = train_dS.size(1)

  for epoch in tqdm(range(num_epochs)):
    train_loss=0
    model.train()
    for states_b , dS_b , payoff_b in dataloader :
        # Move batch tensors to the same device as the model
        states_b = states_b.to(device)
        dS_b = dS_b.to(device)
        payoff_b = payoff_b.to(device)

        optimizer.zero_grad()
        current_batch_size = states_b.size(0)
        features = states_b.size(2)
        PnL=torch.zeros(current_batch_size, device=device) # Initialize PnL on the correct device
        #for j in range(J):
        #  alpha_j=model(states_b[:,j,:]).squeeze()
        #  PnL+=alpha_j*dS_b[:,j]
        # (batch, J, 3) → (batch*J, 3)
        inputs = states_b[:, :J, :]
        states_flat = inputs.reshape(-1, features)
        alpha = model(states_flat).squeeze()          # (batch*J,)
        alpha = alpha.reshape(current_batch_size, J)          # (batch, J)
        PnL = (alpha * dS_b).sum(dim=1)              # (batch,)
        loss=torch.mean((model.X0 + PnL -payoff_b)**2)
        train_loss+=loss
        loss.backward()
        optimizer.step()
        scheduler.step()
    train_loss/=len(dataloader)
    if epoch %4 ==0 :
      print(f'Epoch ={epoch} , Loss ={loss}')

def training_rnn(model , dataloader , batch_size , optimizer , scheduler , num_epochs , device):
    for epoch in tqdm(range(num_epochs)):
        train_loss=0
        model.train()
        for states_b , dS_b , payoff_b in dataloader :
            states_b =states_b.to(device)
            dS_b =dS_b.to(device)
            payoff_b=payoff_b.to(device)

            optimizer.zero_grad()
            current_batch_size =states_b.size(0)
            PnL=torch.zeros(current_batch_size , device=device)
            alpha=model(states_b.permute(1,0,2)).squeeze()
            alpha=alpha[:-1, :].permute(1,0)
            PnL= torch.sum(alpha*dS_b , dim=1)
            loss= torch.mean((model.X0 + PnL-payoff_b)**2)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss+=loss
        train_loss/=len(dataloader)
        if epoch %4 ==0 :  
            print(f'Epoch ={epoch} , Loss ={train_loss}')
