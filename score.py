from medpy import metric

def score(predict, ground_truth):
    dice = metric.binary.dc(predict, ground_truth)
    jc = metric.binary.jc(predict, ground_truth)
    hd = metric.binary.hd95(predict, ground_truth)
    asd = metric.binary.asd(predict, ground_truth)

    return dice, jc, hd, asd
