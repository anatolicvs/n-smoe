
from utils import utils_image as util
from utils.utils_blindsr import degradation_bsrgan, upsample_and_clip
import cv2
import numpy as np
import traceback

if __name__ == '__main__':
    img = util.imread_uint('utils/test.png', 1)
    img = util.uint2single(img)
    sf = 4
    
    for i in range(10000):
        try:
            img_lq, img_hq = degradation_bsrgan(img, sf=sf, lq_patchsize=72)
            print(i)
        except Exception as e:
            print('Error:', e)
            traceback.print_exc()
            continue

        lq_nearest = upsample_and_clip(img_lq, sf)
        img_concat = np.concatenate([util.single2uint(lq_nearest), util.single2uint(img_hq)], axis=1)
        util.imsave(img_concat, str(i)+'.png')