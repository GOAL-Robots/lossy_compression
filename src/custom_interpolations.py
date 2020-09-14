from omegaconf import OmegaConf


OmegaConf.register_resolver("plus", lambda a,b: int(a) + int(b))
OmegaConf.register_resolver("times", lambda a,b: int(a) * int(b))
OmegaConf.register_resolver("plus_times", lambda a,b,c: (int(a) + int(b)) * int(c))
