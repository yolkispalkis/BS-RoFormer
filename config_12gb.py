# config_12gb.py
config = {
    "model": {
        "dim": 384,                 # Уменьшаем размерность (было 512)
        "depth": 4,                 # Значительно уменьшаем глубину (было 6)
        "time_transformer_depth": 1,
        "freq_transformer_depth": 1,
        "stereo": True,
        "num_stems": 1,             # Начнем с одной цели (вокал), это экономит память в `target`
        
        # Улучшения
        "use_conv_stem": True,
        "drop_path_rate": 0.05,     # Меньше регуляризации для маленькой модели

        # Параметры Attention
        "heads": 6,                 # 384 / 6 = 64
        "dim_head": 64,
        "flash_attn": True,

        # STFT параметры (стандартные)
        "stft_n_fft": 2048,
        "stft_hop_length": 512,     # Можно увеличить до 1024, чтобы сократить временную размерность, но это ухудшит качество
        "stft_win_length": 2048,
        
        "multi_stft_resolution_loss_weight": 1.0,
    },
    "training": {
        "batch_size": 1,           # Минимально возможный размер батча
        "learning_rate": 5e-4,     # Можно немного увеличить LR для меньшей модели
        "num_epochs": 150,         # Возможно, понадобится больше эпох для маленькой модели
        "grad_accumulation_steps": 16, # Эффективный батч = 1 * 16 = 16. Можно увеличить до 32 или 64.
        "audio_chunk_size": 44100 * 4, # 4 секунды аудио (было 6). Это КЛЮЧЕВОЙ параметр.
        "device": "cuda",
        "use_compile": True,       # Использовать torch.compile()
        "compile_mode": "reduce-overhead" # Экономный режим компиляции
    }
}
