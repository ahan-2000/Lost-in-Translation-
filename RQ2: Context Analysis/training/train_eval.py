import numpy as np, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import wandb
from tqdm import tqdm

def train_model(model, train_loader, val_loader, lr=2e-5, weight_decay=0.01, epochs=3, warmup_ratio=0.06, device="cpu", experiment_name="occitan_gender", fold=None, class_weights=None, save_dir=None, patience=3):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_steps = max(1, len(train_loader)*epochs)
    num_warm = max(1, int(warmup_ratio*num_steps))
    from transformers import get_linear_schedule_with_warmup
    sch = get_linear_schedule_with_warmup(opt, num_warm, num_steps)
    if class_weights is not None:
        ce = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    else:
        ce = nn.CrossEntropyLoss()
    best = None; best_loss = 1e9; patience_counter = 0
    
    # Initialize WandB
    run_name = f"{experiment_name}_fold_{fold}" if fold is not None else experiment_name
    wandb.init(
        project="occitan-gender-classification",
        name=run_name,
        config={
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "warmup_ratio": warmup_ratio,
            "batch_size": train_loader.batch_size,
            "fold": fold,
            "experiment_name": experiment_name
        }
    )
    
    for ep in tqdm(range(epochs), desc="Epochs", position=0):
        model.train(); tot=0; n=0
        for batch in tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}", position=1, leave=False):
            for k in batch: batch[k] = batch[k].to(device)
            out = model(**batch)
            loss = ce(out["logits"], batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Tighter gradient clipping
            opt.step(); sch.step(); opt.zero_grad()
            tot += loss.item()*batch["label"].size(0); n += batch["label"].size(0)
        tr_loss = tot/max(1,n)
        vl_loss, vl_acc, vl_f1 = eval_model(model, val_loader, device)
        if vl_loss < best_loss:
            best_loss = vl_loss
            best = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {ep+1}")
                break
        
        # Log metrics to WandB
        wandb.log({
            "epoch": ep + 1,
            "train_loss": tr_loss,
            "val_loss": vl_loss,
            "val_accuracy": vl_acc,
            "val_f1_macro": vl_f1,
            "learning_rate": sch.get_last_lr()[0]
        })
        
        print(f"[epoch {ep+1}] train_loss={tr_loss:.4f} val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} val_f1={vl_f1:.4f}")
    if best is not None: model.load_state_dict(best)
    
    # Save model weights if save_dir is provided
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{experiment_name}_fold_{fold}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
    
    # Finish WandB run
    wandb.finish()
    return model

def eval_model(model, loader, device="cpu"):
    model.eval(); ce = nn.CrossEntropyLoss(reduction="sum")
    tot=0; n=0; y_true=[]; y_pred=[]
    with torch.no_grad():
        for batch in loader:
            for k in batch: batch[k] = batch[k].to(device)
            out = model(**batch)
            logits = out["logits"]
            loss = ce(logits, batch["label"])
            tot += loss.item(); n += batch["label"].size(0)
            y_true.extend(batch["label"].cpu().tolist())
            y_pred.extend(logits.argmax(-1).cpu().tolist())
    loss = tot/max(1,n)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return loss, acc, f1

def predict_probs(model, loader, device="cpu"):
    model.eval()
    all_row=[]; all_prob=[]; all_logp=[]; all_gold=[]
    with torch.no_grad():
        for batch in loader:
            row_id = batch["row_id"].cpu().numpy().tolist()
            gold = batch["label"].cpu().numpy().tolist()
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            out = model(**batch)
            logits = out["logits"]
            prob = torch.softmax(logits, dim=-1)
            logp = torch.log_softmax(logits, dim=-1)
            idx = batch["label"].unsqueeze(-1)
            p_true = torch.gather(prob, dim=-1, index=idx).squeeze(-1).cpu().tolist()
            lp_true = torch.gather(logp, dim=-1, index=idx).squeeze(-1).cpu().tolist()
            all_row.extend(row_id); all_prob.extend(p_true); all_logp.extend(lp_true); all_gold.extend(gold)
    import numpy as np
    return np.array(all_row), np.array(all_prob), np.array(all_logp), np.array(all_gold)

