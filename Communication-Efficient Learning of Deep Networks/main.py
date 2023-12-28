import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig


# We hydrate with config file named conf/base.yaml
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(config: DictConfig): 
    print("We are launch our application based on Federated Learning (FL)")
    print("\n Our configurations")
    # print(OmegaConf.to_yaml(config))
    cfg = OmegaConf.to_object(config)
    print(cfg["num_clients"])


if __name__ == "__main__": 
    # Launch main function 
    main()