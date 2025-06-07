# train.py (оптимизированная версия)
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import random
import os
import gc # Garbage Collector

from bs_roformer import BSRoformer
from config_12gb import config # Импортируем конфиг для 12 ГБ

# --- 1. Датасет и Загрузчик данных (без изменений) ---
class MUSDBDataset(Dataset):
    def __init__(self, root_dir, chunk_size, is_train=True, num_stems=1):
        self.root_dir = root_dir
        self.chunk_size = chunk_size
        self.num_stems = num_stems
        self.musdb_dataset = torchaudio.datasets.MUSDB_HQ(root=root_dir, subset='train' if is_train else 'test', download=True)

    def __len__(self):
        return len(self.musdb_dataset)

    def __getitem__(self, idx):
        track = self.musdb_dataset[idx]
        
        mix_waveform = track['mixture']
        
        if self.num_stems == 1:
            target_waveform = track['vocals']
        else:
            # Этот блок сейчас не используется с конфигом для 12ГБ, но оставлен для гибкости
            target_waveform = torch.stack([
                track['drums'],
                track['bass'],
                track['other'],
                track['vocals']
            ], dim=0)

        full_length = mix_waveform.shape[-1]
        if full_length > self.chunk_size:
            start = random.randint(0, full_length - self.chunk_size)
            mix_chunk = mix_waveform[:, start:start + self.chunk_size]
            target_chunk = target_waveform[..., start:start + self.chunk_size]
        else:
            mix_chunk = torch.nn.functional.pad(mix_waveform, (0, self.chunk_size - full_length))
            target_chunk = torch.nn.functional.pad(target_waveform, (0, self.chunk_size - full_length))

        return mix_chunk, target_chunk

# --- Основная функция ---
def main():
    device = torch.device(config['training']['device'])
    
    model = BSRoformer(**config['model']).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    if config['training']['use_compile']:
        compile_mode = config['training'].get('compile_mode', 'default') # Получаем режим компиляции
        print(f"Compiling the model with mode: {compile_mode}...")
        model = torch.compile(model, mode=compile_mode)

    train_dataset = MUSDBDataset(
        root_dir="./data", 
        chunk_size=config['training']['audio_chunk_size'], 
        is_train=True,
        num_stems=config['model']['num_stems']
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=2, # Уменьшаем кол-во воркеров, чтобы экономить RAM
        pin_memory=True
    )
    
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * config['training']['num_epochs'] // config['training']['grad_accumulation_steps'])

    scaler = torch.cuda.amp.GradScaler()
    grad_accum_steps = config['training']['grad_accumulation_steps']
    optimizer.zero_grad() # Обнуляем градиенты перед циклом

    for epoch in range(config['training']['num_epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        for i, (mix, target) in enumerate(pbar):
            mix, target = mix.to(device), target.to(device)

            with torch.cuda.amp.autocast():
                loss = model(mix, target=target)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            pbar.set_postfix(loss=f"{loss.item() * grad_accum_steps:.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")
            
            # Очистка памяти (может помочь, но не всегда)
            del mix, target, loss
            gc.collect()
            torch.cuda.empty_cache()


        print(f"Epoch {epoch+1} finished. Saving checkpoint...")
        os.makedirs("checkpoints_12gb", exist_ok=True)
        model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save(model_to_save.state_dict(), f"checkpoints_12gb/bs_roformer_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()
