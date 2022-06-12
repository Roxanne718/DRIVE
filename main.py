import cv2
import argparse
from score import score
import numpy as np
from PIL import Image
import time

parser = argparse.ArgumentParser()
parser.add_argument('--method', help ='morphology,matched_filtering')
args = parser.parse_args()

if(args.method=='morphology'):
    import methods.morphology as method
elif(args.method=='matched_filtering'):
    import methods.matchedFiltering as method

numList = list(range(22, 41))

dice_list= []
jc_list = []
hd_list = []
asd_list = []

start = time.time()

for num in numList:
    img_path = './dataset/training/images/' + ('%02d' % num) + '_training.tif'
    gt_path = './dataset/training/1st_manual/' + ('%02d' % num) + '_manual1.gif'
    mask_path = './dataset/training/mask/' + ('%02d' % num) + '_training_mask.gif'
    seg_img = method.run(img_path, mask_path)
    ground_truth = np.array(Image.open(gt_path))

    dice, jc, hd, asd = score(seg_img, ground_truth)
    dice_list.append(dice)
    jc_list.append(jc)
    hd_list.append(hd)
    asd_list.append(asd)

    # write image
    cv2.imwrite(f'./results/{args.method}/{num}.png', seg_img)

end = time.time()

print('Dice: ' + str(np.mean(dice_list)))
print('Jaccard: ' + str(np.mean(jc_list)))
print('Hausdorff Distance: ' + str(np.mean(hd_list)))
print('Average surface distance: ' + str(np.mean(asd_list)))
print('Time: ' + str(end-start))


    


