import sys
from pathlib import Path

# Fix import path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.transolver_sonata_lightning_cached import TransolverSonataModel


def inspect_model():
    """Inspect model architecture and parameters"""
    model = TransolverSonataModel()

    # Count parameters by component
    sonata_params = sum(p.numel() for n, p in model.named_parameters() if "sonata" in n)
    transolver_params = sum(
        p.numel() for n, p in model.named_parameters() if "blocks" in n
    )

    print(f"Sonata encoder parameters: {sonata_params:,}")
    print(f"Transolver parameters: {transolver_params:,}")

    # Check frozen status
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)

    print(f"Frozen layers: {frozen}")
    print(f"Trainable layers: {trainable}")

    return model


if __name__ == "__main__":
    model = inspect_model()
