from omegaconf import OmegaConf


OmegaConf.register_resolver("plus", lambda a,b: int(a) + int(b))
OmegaConf.register_resolver("times", lambda a,b: int(a) * int(b))
OmegaConf.register_resolver("plus_times", lambda a,b,c: (int(a) + int(b)) * int(c))
OmegaConf.register_resolver("times_plus", lambda a,b,c: (int(a) * int(b)) + int(c))
OmegaConf.register_resolver("slash_to_dot", lambda s: s.replace('/', '.'))
OmegaConf.register_resolver("get_dim_correlate", lambda dim_sources, dim_shared, dim_correlate, correlate_dilation_factor, auto: int(correlate_dilation_factor) * (int(dim_sources) + int(dim_shared)) if bool(auto) else dim_correlate)
