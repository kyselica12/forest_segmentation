import matplotlib.pyplot as plt
import numpy as np


BLUE = np.array([100,100,255])
GREEN = np.array([100,255,100])
RED = np.array([255,100,100])
ORANGE = np.array([255,150,100])
WHITE = np.array([255,255,255]) 

def show_image_mask(img, mask):
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    _ = [ax.set_axis_off() for ax in axs.ravel()]
    axs[0,0].imshow(img)
    axs[0,1].imshow(mask)
    axs[1,0].set_title('Labled as Forest')
    img_T = img.copy()
    img_T[mask!=1] = WHITE
    img_F = img.copy()
    img_F[mask==1] = WHITE
    img_TT = img_T.copy() * 2
    img_TT[img_TT > 255] = 255
    axs[1,0].imshow(img_TT /255)
    axs[1,1].set_title('Labled as Non-Forest')
    axs[1,1].imshow(img_F)
    plt.show()
    
def show_results(img, original_mask, generated_masks,  new_mask, size=4, intensity=1, output_paht=None):
    img = img * intensity
    img[img > 255] = 255
    img = img.astype(np.uint8)
    
    fig, axs = plt.subplots(5, 2, figsize=(3*size, 6*size))
    _ = [ax.set_axis_off() for ax in axs.ravel()]
    
    axs[0,0].set_title('Original Image')
    axs[0,0].imshow(img)
    axs[0,1].set_title('Generated Masks')
    show_anns(generated_masks, axes=axs[0,1])
    
    axs[1,0].set_title('Original Mask')
    axs[1,0].imshow(original_mask)
    axs[1,1].set_title('New Mask')
    axs[1,1].imshow(new_mask)
    
    orig_img_T, orig_img_F = get_cut_out_images(img, original_mask)
    axs[2,0].set_title('Original Mask: Forest')
    axs[2,0].imshow(orig_img_T)
    axs[3,0].set_title('Original Mask: Background')
    axs[3,0].imshow(orig_img_F)
    
    new_img_T, new_img_F = get_cut_out_images(img, new_mask)
    axs[2,1].set_title('New Mask: Forest')
    axs[2,1].imshow(new_img_T)
    axs[3,1].set_title('New Mask: Background')
    axs[3,1].imshow(new_img_F)
    
    c_img = compare_masks_image(original_mask, new_mask, img)
    axs[4,1].set_title('Comparison')
    axs[4,1].imshow(c_img)
    
    if output_paht is not None:
        plt.savefig(output_paht)
        plt.close(fig)
    else:
        plt.show()
    
    
    
def show_mask_comparison(original_mask, new_mask, img, size=4):
    fig, axs = plt.subplots(4, 2, figsize=(3*size, 4*size))
    _ = [ax.set_axis_off() for ax in axs.ravel()]
    
    axs[0,0].imshow(original_mask)
    axs[0,1].imshow(new_mask)
    
    orig_img_T, orig_img_F = get_cut_out_images(img, original_mask)
    axs[1,0].imshow(orig_img_T)
    axs[1,0].set_title('Original Mask: Forest')
    axs[2,0].imshow(orig_img_F)
    axs[2,0].set_title('Original Mask: Background')
    
    new_img_T, new_img_F = get_cut_out_images(img, new_mask)
    axs[1,1].imshow(new_img_T)
    axs[1,1].set_title('New Mask: Forest')
    axs[2,1].imshow(new_img_F)
    axs[2,1].set_title('New Mask: Background')
    
    c_img = compare_masks_image(original_mask, new_mask, img)
    axs[3,1].imshow(c_img)
    axs[3,1].set_title('Comparison')
    
    plt.show()

def get_cut_out_images(img, mask, intensity=1):
    
    img_T = img.copy() * intensity
    img_T[img_T > 255] = 255
    img_T[np.logical_not(mask)] = WHITE
    img_F = img.copy() * intensity
    img_F[img_F > 255] = 255
    img_F[mask] = WHITE
    
    return img_T, img_F

def compare_masks_image(original_mask, new_mask, img, intensity=1):
    c_img = img.copy() * intensity
    c_img[c_img > 255] = 255
    c_img[np.logical_and(new_mask ==1, original_mask==1)] = WHITE
    c_img[np.logical_and(new_mask ==1, original_mask==0)] = BLUE
    c_img[np.logical_and(new_mask ==0, original_mask==1)] = ORANGE
    
    return c_img

def plot_contours(img, contours, ax):
    ax.imshow(img, cmap=plt.cm.gray)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def show_anns(anns, axes=None):
    if len(anns) == 0:
        return
    if axes:
        ax = axes
    else:
        ax = plt.gca()
        ax.set_autoscale_on(False)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m**0.5)))
        
def show_anns_masked(anns, MASK, axes=None):
    if len(anns) == 0:
        return
    if axes:
        ax = axes
    else:
        ax = plt.gca()
        ax.set_autoscale_on(False)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
            img = img * MASK 
        ax.imshow(np.dstack((img, m**0.5)))