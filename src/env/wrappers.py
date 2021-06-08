def make_pad_env(*args, **kwargs):
    if kwargs["domain_name"] == "metaworld":
        from env import metaworld_wrappers
        return metaworld_wrappers.make_pad_env(*args, **kwargs)
    else:
        from env import dmc_wrappers
        return dmc_wrappers.make_pad_env(*args, **kwargs)
