# config.py
config = {
    "data": {
        "root_dir": "./MUSDB18-HQ"  # Путь к вашей папке с датасетом
    },
    "model": {
        "dim": 384,
        "depth": 4,
        "time_transformer_depth": 1,
        "freq_transformer_depth": 1,
        "stereo": True,
        "num_stems": 2,  # <--- Ключевой параметр: 2 выходных стема
        "use_conv_stem": True,
        "drop_path_rate": 0.05,
        "heads": 6,
        "dim_head": 64,
        "flash_attn": True,
        "stft_n_fft": 2048,
        "stft_hop_length": 512,
        "stft_win_length": 2048,
        "multi_stft_resolution_loss_weight": 1.0,
    },
    "training": {
        "batch_size": 1,
        "learning_rate": 5e-4,
        "num_epochs": 150,
        "grad_accumulation_steps": 16,
        "audio_chunk_size": 44100 * 3, # 3 секунды аудио
        "device": "cuda",
        "use_compile": True,
        "compile_mode": "reduce-overhead"
    }
}
