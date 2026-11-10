from enum import StrEnum

class Saves(StrEnum):
    ARCHITECTURE = "modelArchitecture"
    EMBEDDING = "embeddingModel"
    CONFIG = "config"
    EXPERIMENT = "experimentName"
    STATE = "state_dict"