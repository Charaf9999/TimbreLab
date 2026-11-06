"""Entry point for supervised learning."""


def train_supervised(table, task, target, features, model_name, group_column=None):
    """Return pipeline, metrics, predictions and metadata using the shared result contract."""
    raise NotImplementedError(
        "Supervised training is not connected yet. Waiting for Basma's models.train_supervised "
        "(classification and regression), with Charaf's Random Forest registered through that trainer. "
        "You can save your analysis settings from Export."
    )
