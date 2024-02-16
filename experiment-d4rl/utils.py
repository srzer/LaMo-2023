from torch import optim
from itertools import chain
import timm.optim.optim_factory as optim_factory


def get_optimizer(args, model):
    if args["pretrained_lm"]:
        optimizer = optim.AdamW(
            [
                {
                    "params": list(
                        chain(
                            *[
                                list(
                                    (
                                        filter(
                                            lambda p: p.requires_grad,
                                            module.parameters(),
                                        )
                                    )
                                )
                                for module in model.children()
                                if (
                                    ("transformers" in str(type(module)).lower())
                                    or ("dataparallel" in str(type(module)).lower())
                                )
                            ]
                        )
                    ),
                    "lr": args["lm_learning_rate"]
                    if args["lm_learning_rate"] is not None
                    else args["learning_rate"],
                    "weight_decay": 0.0,
                },
                {
                    "params": list(
                        chain(
                            *[
                                list(
                                    (
                                        filter(
                                            lambda p: p.requires_grad,
                                            module.parameters(),
                                        )
                                    )
                                )
                                for module in model.children()
                                if (
                                    ("transformers" not in str(type(module)).lower())
                                    and (
                                        "dataparallel" not in str(type(module)).lower()
                                    )
                                )
                            ]
                        )
                    ),
                    "weight_decay": args["weight_decay"],
                },
            ],
            lr=args["learning_rate"],
            eps=1e-6,
        )
        # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args["learning_rate"], weight_decay=args["weight_decay"], eps=1e-6)
    else:
        for name, param in model.named_parameters():
            print(f"{name}: {param.requires_grad}")
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args["learning_rate"],
            weight_decay=args["weight_decay"],
        )  
    return optimizer
