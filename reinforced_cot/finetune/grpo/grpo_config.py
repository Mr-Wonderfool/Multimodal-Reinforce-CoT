from reinforced_cot.common.base_config import BaseTrainerConfig
from trl import GRPOConfig

class GRPOTrainerConfig(BaseTrainerConfig, GRPOConfig):
    # additional arguments compared with original GRPOConfig
    model_path: str 
    
