import sys
import numpy as np
import cv2
import sys
sys.path.append('/data/yyang/workspace/Iphone-Warp')
from dycheck.dataset.iphonev3 import iPhoneDataset
import os
if __name__ == "__main__":

    sequence = 'apple'
    dataset = iPhoneDataset(
        data_root='data/iphone_sm',
        scene=sequence,
        use_undistort=False
    )

    valid_index = dataset.valid_index

    img_list = []
    warp_list = []
    mask_list = []
    target_img_list = []

    for idx in valid_index:
        pixel_a, pixel_b, warpped_pixel_b, valid_mask = dataset.warp_a2b(idx,0,idx,1)

        source_img = pixel_a[...,:3].astype(np.uint8)
        target_img = pixel_b[...,:3].astype(np.uint8)
        warp_img = warpped_pixel_b.astype(np.uint8)
        # depth = (depth_a * 255).astype(np.uint8)

        img_list.append(source_img)
        target_img_list.append(target_img)
        warp_list.append(warp_img)
        mask_list.append(valid_mask.astype(np.bool_))

        cv2.imwrite(f'tools/ALTest/warp_{idx}.png', cv2.cvtColor(warp_img, cv2.COLOR_BGR2RGB))
        cv2.imwrite(f"tools/ALTest/mask_{idx}.png", (255.0*valid_mask).astype(np.uint8))
        cv2.imwrite(f"tools/ALTest/source_{idx}.png", cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB))
        cv2.imwrite(f"tools/ALTest/target_{idx}.png", cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
        # cv2.imwrite(f"sample/depth_{idx}.png", depth)
        print(f"Processed index {idx}")

    # Save the images and masks
    imgs = np.array(img_list)
    warps = np.array(warp_list)
    masks = np.array(mask_list)
    target_imgs = np.array(target_img_list)
    
    # Save the images to .npy files
    os.makedirs(f'iphone/dataset/{sequence}', exist_ok=True)
    save_path = f"iphone/dataset/{sequence}/target.npy"
    np.save(save_path, target_imgs)
    save_path = f"iphone/dataset/{sequence}/img.npy"
    np.save(save_path, imgs)
    save_path = f"iphone/dataset/{sequence}/warp.npy"
    np.save(save_path, warps)
    save_path = f"iphone/dataset/{sequence}/mask.npy"
    np.save(save_path, masks)
    print(f"Images saved to {save_path}")


