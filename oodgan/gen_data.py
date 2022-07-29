from SeqGAN.train import Trainer 
from SeqGAN.get_config import get_config

config = get_config('config-infer.ini')

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
                c_filter_sizes=config["c_filter_sizes"],c_num_filters=config["c_num_filters"],k=config["topk"],mode=config["mode"])

print("trainer init, done")
trainer.agent.load(config["g_weights_path"])
print("load weight, done")
trainer.generate_txt(config["g_test_path"], config["generate_samples"],k=config["topk"],use_eos=True)
