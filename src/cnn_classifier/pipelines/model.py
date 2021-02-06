from kedro.pipeline import node, Pipeline
from cnn_classifier.nodes.model import (
    create_model,
)

def create_model_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=create_model,
                inputs=None,
                outputs=None,
                name="create_model",
            ),
        ]
    )