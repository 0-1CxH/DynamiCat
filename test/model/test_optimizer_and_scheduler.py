from dynamicat.model.deepspeed_model import DeepSpeedHFModelProvider
from test.model.test_model_load import model

optimizer = DeepSpeedHFModelProvider.get_optimizer(model, 5e-5, True)

sched = DeepSpeedHFModelProvider.get_lr_scheduler(
    optimizer,
    "cosine",

        num_warmup_steps=10,
        num_training_steps=2000
    )

if __name__ == '__main__':

    print(optimizer)
    optimizer2 = DeepSpeedHFModelProvider.get_optimizer(model, 1e-5,  False, 0.5)
    print(optimizer2)


    print(sched)

    sched2 = DeepSpeedHFModelProvider.get_lr_scheduler(
        optimizer,
        "wsd",

        scheduler_specific_kwargs = {
            "warmup_ratio": 0.1,
            "decay_ratio": 0.3
        },
        num_training_steps=2000
    )
    print(sched2)

    sched3 = DeepSpeedHFModelProvider.get_lr_scheduler(
        optimizer,
        "custom",
        num_training_steps=2000,
        lr_lambda=lambda x:x
    )
    print(sched3)