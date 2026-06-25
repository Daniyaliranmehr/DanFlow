# test/test_training.py

from danflow.training import AverageMeter

def test_average_meter():
    meter = AverageMeter()

    meter.update(2)
    meter.update(4)

    assert abs(meter.avg - 3) < 1e-6