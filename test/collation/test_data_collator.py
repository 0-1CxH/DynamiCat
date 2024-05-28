from dynamicat.collation.planned_data_collator import GeneralDataCollator
from test.tensorplanning.test_tensor_planner import plan2, plan4

if __name__ == '__main__':
    collator = GeneralDataCollator(
        ["prompt", "chosen"],
        ["prompt"],
        True
    )
    for plan_item in plan2.iterate_plan_items():
        batch = collator.collate(plan_item)
        for k, v in batch.items():
            print(k, v.shape)

    print("*"*50)

    collator2 = GeneralDataCollator(
        ["content"],
        [],
        False
    )
    for plan_item in plan4.iterate_plan_items():
        batch = collator2.collate(plan_item)
        for k, v in batch.items():
            print(k, v.shape, v)
