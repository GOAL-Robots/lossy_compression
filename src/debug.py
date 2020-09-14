from omegaconf import OmegaConf
import hydra


OmegaConf.register_resolver("plus", lambda a,b: int(a) + int(b))
OmegaConf.register_resolver("times", lambda a,b: int(a) * int(b))
OmegaConf.register_resolver("plus_times", lambda a,b,c: (int(a) + int(b)) * int(c))

@hydra.main(config_path='../config/', config_name='config.yaml')
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
