from dynamicat.utils.deepspeed_utils import DeepSpeedConfigBuilder

if __name__ == '__main__':
    DeepSpeedConfigBuilder.make_config_for_training(
        32,
        4,
        3,
        True,
        True,
        True,
        return_dict=False,
        tensorboard_save_path="test_path_0000"
    ).print_user_config()

    print(
    DeepSpeedConfigBuilder.make_config_for_training(
        128,
        1,
        2,
        False,
        False,
        False,
        return_dict=True,
        tensorboard_save_path="test_path_1111",
        enable_hybrid_engine=True
    )
    )

    DeepSpeedConfigBuilder.make_config_for_eval(
        32,
        4,
        3,
        True,
        True,
        return_dict=False,
    ).print_user_config()

    print(
    DeepSpeedConfigBuilder.make_config_for_eval(
        128,
        1,
        2,
        False,
        False,
        return_dict=True,
    )
    )