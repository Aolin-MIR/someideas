import numpy as np
from SeqGAN.train import Trainer
from SeqGAN.get_config import get_config

# config = get_config('config.ini')
config = get_config('config_big4.ini')
trainer = Trainer(config["batch_size"],
                config["max_length"],
                config["g_e"],
                config["g_h"],
                config["d_e"],
                config["d_h"],
                config["c_e"],
                d_dropout=config["d_dropout"],
                path_pos=config["path_pos"],
                path_neg=config["path_neg"],
                path_gctrain=config["path_gctrain"],
                path_label=config["path_label"],
                g_lr=config["g_lr"],
                d_lr=config["d_lr"],
                n_sample=config["n_sample"],
                generate_samples=config["generate_samples"],
                cls_num=config["cls_num"],
                c_filter_sizes=config["c_filter_sizes"],c_num_filters=config["c_num_filters"],k=config["topk"],clip=config["clip"])


# Pretraining for adversarial training 对抗训练的pretrain
trainer.pre_train(g_epochs=config["g_pre_epochs"],
                d_epochs=config["d_pre_epochs"],
                g_pre_path=config["g_pre_weights_path"],
                d_pre_path=config["d_pre_weights_path"],
                g_lr=config["g_pre_lr"],
                d_lr=config["d_pre_lr"],
                c_epochs=config["c_pre_epochs"],
                c_pre_path=config["c_pre_weights_path"],
                c_lr=config["c_pre_lr"],
                train_c=config["pre_train_c"],
                train_d=config["pre_train_d"],
                train_g=config["pre_train_g"],
                )


trainer.load_pre_train(config["g_pre_weights_path"], config["d_pre_weights_path"],config["c_pre_weights_path"])
trainer.reflect_pre_train()

trainer.train(steps=config["step"],
            g_steps=config["g_step"],
            head=10,
            g_weights_path=config["g_weights_path"],
            d_weights_path=config["d_weights_path"],
            gc=config["gc"],gd=config["gd"])

trainer.save(config["g_weights_path"], config["d_weights_path"])


trainer.load(config["g_weights_path"], config["d_weights_path"])

trainer.test()

trainer.generate_txt(config["g_test_path"], config["generate_samples"], k=config["topk"])

