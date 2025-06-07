# train.py
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import random
import os
import gc

# Импорт вашей модели и конфигурации
from bs_roformer import BSRoformer
from config import config

# --- Класс Датасета: здесь происходит вся магия подготовки данных ---
class TwoStemMUSDBDataset(Dataset):
    """
    Этот датасет загружает треки из MUSDB18-HQ и на лету создает два целевых стема:
    1. Барабаны
    2. Все остальное (микс минус барабаны)
    """
    def __init__(self, root_dir, chunk_size, is_train=True):
        self.root_dir = root_dir
        self.chunk_size = chunk_size
        
        # Инициализируем датасет torchaudio.
        # download=False, так как датасет у вас уже есть.
        self.musdb_dataset = torchaudio.datasets.MUSDB_HQ(
            root=self.root_dir,
            subset='train' if is_train else 'test',
            download=False 
        )

    def __len__(self):
        return len(self.musdb_dataset)

    def __getitem__(self, idx):
        # Получаем один трек из датасета
        track = self.musdb_dataset[idx]
        
        # Извлекаем необходимые аудиодорожки
        mix_waveform = track['mixture']
        drums_waveform = track['drums']
        
        # --- Создание целевых стемов ---
        # Стем 1: Барабаны (уже готов)
        # Стем 2: "Все остальное" = Микс - Барабаны.
        # Этот подход гарантирует, что сумма стемов всегда равна миксу.
        other_instruments_waveform = mix_waveform - drums_waveform

        # Объединяем два целевых стема в один тензор
        # Форма итогового тензора: (num_stems, channels, time) -> (2, 2, L)
        target_waveform = torch.stack([drums_waveform, other_instruments_waveform], dim=0)

        # --- Вырезаем случайный фрагмент для обучения ---
        full_length = mix_waveform.shape[-1]
        if full_length > self.chunk_size:
            start = random.randint(0, full_length - self.chunk_size)
            mix_chunk = mix_waveform[:, start:start + self.chunk_size]
            target_chunk = target_waveform[..., start:start + self.chunk_size]
        else: # Если трек короче, дополняем его тишиной
            pad_length = self.chunk_size - full_length
            mix_chunk = torch.nn.functional.pad(mix_waveform, (0, pad_length))
            target_chunk = torch.nn.functional.pad(target_waveform, (0, pad_length))

        return mix_chunk, target_chunk

# --- Основная функция для запуска обучения ---
def main():
    # Настройки из конфига
    train_cfg = config['training']
    model_cfg = config['model']
    data_cfg = config['data']
    
    device = torch.device(train_cfg['device'])
    
    # 1. Инициализация модели
    model = BSRoformer(**model_cfg).to(device)
    print(f"Model for {model_cfg['num_stems']} stems created. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if train_cfg['use_compile']:
        compile_mode = train_cfg.get('compile_mode', 'default')
        print(f"Compiling the model with mode: {compile_mode}...")
        model = torch.compile(model, mode=compile_mode)

    # 2. Создание датасета и загрузчика
    train_dataset = TwoStemMUSDBDataset(
        root_dir=data_cfg['root_dir'], 
        chunk_size=train_cfg['audio_chunk_size'], 
        is_train=True
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_cfg['batch_size'], 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    # 3. Настройка оптимизатора и планировщика
    optimizer = AdamW(model.parameters(), lr=train_cfg['learning_rate'])
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * train_cfg['num_epochs'] // train_cfg['grad_accumulation_steps'])
    scaler = torch.cuda.amp.GradScaler() # Для смешанной точности
    
    # 4. Цикл обучения
    print("Starting training...")
    for epoch in range(train_cfg['num_epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['num_epochs']}")
        optimizer.zero_grad()
        
        for i, (mix, target) in enumerate(pbar):
            mix, target = mix.to(device), target.to(device)

            with torch.cuda.amp.autocast():
                loss = model(mix, target=target)
                loss = loss / train_cfg['grad_accumulation_steps']

            scaler.scale(loss).backward()

            if (i + 1) % train_cfg['grad_accumulation_steps'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            pbar.set_postfix(loss=f"{loss.item() * train_cfg['grad_accumulation_steps']:.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")
        
        # 5. Сохранение модели
        checkpoint_dir = "checkpoints_drums_vs_other"
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save(model_to_save.state_dict(), f"{checkpoint_dir}/bs_roformer_epoch_{epoch+1}.pth")
        print(f"Epoch {epoch+1} finished. Checkpoint saved.")

if __name__ == "__main__":
    main()
