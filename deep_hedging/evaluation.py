import torch


def evaluate(model , dataloader , device ):
  model.eval()
  with torch.inference_mode():
    all_PnL=[]
    all_payoff=[]
    for states_b , dS_b , payoff_b in dataloader :
      states_b = states_b.to(device)
      dS_b = dS_b.to(device)
      payoff_b = payoff_b.to(device)
      current_batch_size = states_b.size(0)
      features = states_b.size(2)
      J = dS_b.size(1)
      PnL=torch.zeros(current_batch_size, device=device)
      #for j in range(J):
      #  alpha_j=model(states_b[:,j,:]).squeeze()
      #  PnL+=alpha_j*dS_b[:,j]
      # (batch, J, 3) → (batch*J, 3)
      inputs = states_b[:, :J, :]
      states_flat = inputs.reshape(-1, features)
      alpha = model(states_flat).squeeze()          # (batch*J,)
      alpha = alpha.reshape(current_batch_size, J)          # (batch, J)
      PnL = (alpha * dS_b).sum(dim=1)              # (batch,)
      all_PnL.append(PnL)
      all_payoff.append(payoff_b)
    all_PnL=torch.cat(all_PnL)
    all_payoff=torch.cat(all_payoff)
    HedgingPnL = model.X0 + all_PnL - all_payoff
    mean_squarred_pnl=torch.mean((model.X0 + all_PnL -all_payoff)**2)
    mean_pnl=torch.mean(model.X0 +all_PnL - all_payoff)
    return HedgingPnL, mean_squarred_pnl , mean_pnl
  
def evaluate_rnn(model, dataloader, device):
    model.eval()
    with torch.inference_mode():
        all_PnL = []
        all_payoff = []
        for states_b, dS_b, payoff_b in dataloader:
            states_b = states_b.to(device)
            dS_b = dS_b.to(device)
            payoff_b = payoff_b.to(device)
            current_batch_size = states_b.size(0)
            alpha = model(states_b.permute(1, 0, 2)).squeeze()
            alpha = alpha[:-1, :].permute(1, 0)
            PnL = (alpha * dS_b).sum(dim=1)
            all_PnL.append(PnL)
            all_payoff.append(payoff_b)
        all_PnL = torch.cat(all_PnL)
        all_payoff = torch.cat(all_payoff)
        HedgingPnL = model.X0 + all_PnL - all_payoff
        mean_squared_pnl = torch.mean((model.X0 + all_PnL - all_payoff)**2)
        mean_pnl = torch.mean(model.X0 + all_PnL - all_payoff)
        return HedgingPnL, mean_squared_pnl, mean_pnl  
