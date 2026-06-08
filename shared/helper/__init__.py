from .dataloaders import CIFAR10, get_sinusoid_example
from .util import download_pretrained_model, plot_regression


def __getattr__(name):
    if name in ("CIFAR10Net", "FMNIST", "QuickDS", "get_dataset"):
        from .util_gp import CIFAR10Net, FMNIST, QuickDS, get_dataset
        return {"CIFAR10Net": CIFAR10Net, "FMNIST": FMNIST, "QuickDS": QuickDS, "get_dataset": get_dataset}[name]
    if name == "WideResNet":
        from .wideresnet import WideResNet
        return WideResNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
