from skimage import color
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
from skimage.filters import threshold_otsu, median, prewitt
from skimage.morphology import closing, disk, remove_small_holes
from skimage.measure import regionprops, label
from skimage.transform import rescale
import numpy as np

def histogram_stretch(img_in):

    img_float = img_as_float(img_in)
    min_val = img_float.min()
    max_val = img_float.max()
    min_desired = 0.0
    max_desired = 1.0
	
    img_out = ((img_float-min_val)*(max_desired-min_desired)/(max_val-min_val))+min_desired
    return img_as_ubyte(img_out)
    
def process_image(image):
    
    img_grey = color.rgb2gray(image)
    
    img_stretched = histogram_stretch(img_grey)
    img_filtered = median(img_stretched, np.ones((5,5)))
    img_outline = prewitt(img_filtered)
    img_closed = closing(img_outline, disk(5))
    threshold = threshold_otsu(img_closed)
    img_tresh = img_closed > threshold
    img_filled = remove_small_holes(img_tresh, connectivity=2)
        
    return img_filled

def center_image(image):
    
    shape = image.shape
    label_image = label(image)
    props = regionprops(label_image)
    
    props_sorted = sorted(props, key=lambda x: x.area, reverse=True)
    
    main_blob = props_sorted[0]
    
    minr, minc, maxr, maxc = main_blob.bbox
    cropped_blob = image[minr:maxr, minc:maxc]
    
    centered_image = np.zeros(shape, dtype=np.bool_)
    
    center_r, center_c = shape[0] // 2, shape[1] // 2
    blob_r, blob_c = cropped_blob.shape
    start_r = center_r - blob_r // 2
    start_c = center_c - blob_c // 2
    
    centered_image[start_r:start_r + blob_r, start_c:start_c + blob_c] = cropped_blob
    
    scale_factor = (2*shape[0]/3) / (maxr - minr)
    centered_image_rescaled = rescale(centered_image, scale_factor, anti_aliasing=False)
    
    new_shape = centered_image_rescaled.shape
    
    if new_shape[0] < shape[0]:
        pad = shape[0] - new_shape[0]
        pad_top = pad // 2
        pad_bottom = pad - pad_top
        centered_image = np.pad(centered_image_rescaled, ((pad_top, pad_bottom), (0, 0)), mode='constant')
        
    if new_shape[0] > shape[0]:
        crop = new_shape[0] - shape[0]
        crop_top = crop // 2
        crop_bottom = crop - crop_top
        centered_image = centered_image_rescaled[crop_top:(new_shape[0] - crop_bottom), :]
        
    if new_shape[1] < shape[1]:
        pad = shape[1] - new_shape[1]
        pad_left = pad // 2
        pad_right = pad - pad_left
        centered_image = np.pad(centered_image, ((0, 0), (pad_left, pad_right)), mode='constant')
        
    if new_shape[1] > shape[1]:
        crop = new_shape[1] - shape[1]
        crop_left = crop // 2
        crop_right = crop - crop_left
        centered_image = centered_image[:, crop_left:(new_shape[1] - crop_right)]
    
    return centered_image