from enum import StrEnum

class Paths(StrEnum):
    SOURCE = "src"
    ARTIFACTS = "src/artifacts"
    DATA = "src/artifacts/data"
    MODELS = "src/artifacts/models"
    CLASSIFIERS = "src/artifacts/models/classifiers"
    EMBEDDING = "src/artifacts/models/embedding"
    RESULTS = "src/artifacts/results"
    VISUALS = "src/artifacts/visuals"
    CONFIGS = "src/configs"
    UTILS = "src/utils"
    ENUMS = "src/utils/enums"