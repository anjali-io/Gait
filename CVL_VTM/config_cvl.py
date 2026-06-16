conf = {
    "WORK_PATH": "/home/himanshu/CVL_VST",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",

    "data": {
        "dataset_path": "/home/himanshu/CVL_VST/CASIA-B-64",
        "resolution": 64,
        "dataset": "CASIA-B",
        "pid_num": 90,          # increased for better accuracy
        "pid_shuffle": False,
    },

    "model": {
        # Backbone
        "hidden_dim": 256,
        "vtm_hidden": 512,      # kept (harmless even if CVL unused)

        # Training
        "lr": 1e-4,
        "hard_or_full_trip": "hard",   # use hard triplet after warmup
        "batch_size": (16, 8),         # better identity diversity
        "restore_iter": 0,
        "total_iter": 150000,          # train long
        "margin": 0.2,
        "num_workers": 3,
        "frame_num": 30,
        "model_name": "GaitSet",
    }
}



