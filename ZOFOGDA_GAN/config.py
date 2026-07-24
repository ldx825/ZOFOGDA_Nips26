from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    
    # Experiment name
    name: str = "zofogda_gan_experiment"
    
    # Dataset configurations
    datasets: List[str] = None
    
    # Hyperparameter search
    learning_rates: List[float] = None
    gamma_values: List[float] = None
    zo_delta_values: List[float] = None
    
    # Training
    num_trials: int = 3
    early_stopping_patience: int = 20
    
    # Logging
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    log_dir: str = "./logs"
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["mnist", "fashionmnist", "cifar10"]
        if self.learning_rates is None:
            self.learning_rates = [1e-3, 5e-4, 1e-4]
        if self.gamma_values is None:
            self.gamma_values = [1e-3, 5e-4, 1e-4]
        if self.zo_delta_values is None:
            self.zo_delta_values = [1e-3, 1e-4, 1e-5]


def get_best_config(dataset: str) -> ZOFOGDACfg:
    """Get pre-tuned configuration for each dataset"""
    from zofo_gda_gan import ZOFOGDA_GAN_Cfg
    
    base_config = {
        "mnist": {
            "gamma": 1e-3,
            "eta_k": 1.0,
            "c1": 0.1,
            "c2": 0.1,
            "n_critic": 3,
            "epochs": 100
        },
        "fashionmnist": {
            "gamma": 8e-4,
            "eta_k": 1.0,
            "c1": 0.15,
            "c2": 0.15,
            "n_critic": 3,
            "epochs": 120
        },
        "cifar10": {
            "gamma": 5e-4,
            "eta_k": 0.8,
            "c1": 0.2,
            "c2": 0.2,
            "n_critic": 5,
            "epochs": 200
        },
        "cifar100": {
            "gamma": 3e-4,
            "eta_k": 0.5,
            "c1": 0.3,
            "c2": 0.3,
            "n_critic": 5,
            "epochs": 250
        }
    }
    
    if dataset not in base_config:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    config = base_config[dataset]
    
    return ZOFOGDA_GAN_Cfg(
        dataset=dataset,
        gamma=config["gamma"],
        eta_k=config["eta_k"],
        c1=config["c1"],
        c2=config["c2"],
        n_critic=config["n_critic"],
        epochs=config["epochs"]
    )