from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

dpath = "../dev/tensorboards/belle_G5200"
e = EventAccumulator(dpath)
e.Reload()

print(e.Tags())

flops_series = [_.value for _ in e.Scalars('tflops (mean)')]
print(sum(flops_series)/len(flops_series))


print(sum([_.value for _ in e.Scalars('time (mean)')]))