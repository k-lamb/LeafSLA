import os
import glob
import pandas as pd # type: ignore
import numpy as np # type: ignore
import skimage # type: ignore
import cv2 # type: ignore
from segment_anything import sam_model_registry, SamPredictor # type: ignore
import PIL # type: ignore

# import images and create relevant folders as needed
def img_import(img_folder, folders=True):
    img_list = [file for file in glob.glob(f"./{img_folder}/*")
                if os.path.isfile(file) and file.lower().endswith(('.jpg', '.jpeg', '.heic', '.png'))]
    img_names = [os.path.splitext(os.path.basename(file))[0] for file in img_list]

    if folders == True:
        # generate folders for analysis (as needed)
        folders = ["{0}/{1}/color_masks".format(os.getcwd(), img_folder),
                "{0}/{1}/processed".format(os.getcwd(), img_folder),
                "{0}/{1}/bboxes".format(os.getcwd(), img_folder),
                "{0}/{1}/perspective".format(os.getcwd(), img_folder),
                "{0}/{1}/threshold".format(os.getcwd(), img_folder)]

        if not os.path.exists("{0}/data".format(os.getcwd())):
            os.makedirs("{0}/data".format(os.getcwd())) # checks data folder separately
        if not os.path.exists(folders[0]): # checks folders exist
            # Create all folders
            for folder in folders:
                os.makedirs(folder)
            print(f"All directories created: {', '.join(folders)}")
    
    return img_list, img_names

# set up data frames for addending information into
def new_data():
    df = pd.DataFrame(columns=['Image_ID', 'Shape', 'Contour_Area', 'Hull_Area', 'Perimeter', 'Hull_Perimeter', 'Lobing_Ratio']) # meta data for images with some stats
    area_df = pd.DataFrame(columns=['Image_ID', 'Lobing_Ratio', 'Area', 'Prop_Red']) # stats data frame (can't combine because square object is needed for making area scalar)
    
    return df, area_df

# color thresholds for first pass bounding
def color_thresher(reds=True, yellows=True, browns=False):
    results = {}

    # Greens
    results["lower_green"] = np.array([20, 40, 40])   # Allow darker/less saturated greens
    results["upper_green"] = np.array([110, 255, 255]) 

    if reds == True:
        results["lower_red1_light"] = np.array([0, 20, 100])
        results["upper_red1_light"] = np.array([10, 150, 255])

        results["lower_red2_light"] = np.array([170, 20, 100])
        results["upper_red2_light"] = np.array([180, 150, 255])

        # Dark Reds (deep/rich)
        results["lower_red1_dark"] = np.array([0, 50, 50])
        results["upper_red1_dark"] = np.array([10, 255, 150])

        results["lower_red2_dark"] = np.array([170, 50, 50])
        results["upper_red2_dark"] = np.array([180, 255, 150])

    if yellows == True:
        # Yellows
        results["lower_yellow"] = np.array([10, 100, 100])
        results["upper_yellow"]= np.array([35, 255, 200])  

    if browns == True:
        # Browns
        results["lower_brown"] = np.array([10, 70, 30])
        results["upper_brown"] = np.array([20, 200, 150])  

    return results

# finds rough bounding information for leaf and red box
def setup_image(img_src, img_folder, img_name, color_ranges, sq_size, leaf_position, padding = 20, min_size = 50, bbox_multiplier = 2):
    image = cv2.imread("./{0}".format(img_src))
    
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Light red masks
    mask1_light = cv2.inRange(image_hsv, color_ranges['lower_red1_light'], color_ranges['upper_red1_light'])
    mask2_light = cv2.inRange(image_hsv, color_ranges['lower_red2_light'], color_ranges['upper_red2_light'])

    # Dark red masks
    mask1_dark = cv2.inRange(image_hsv, color_ranges['lower_red1_dark'], color_ranges['upper_red1_dark'])
    mask2_dark = cv2.inRange(image_hsv, color_ranges['lower_red2_dark'], color_ranges['upper_red2_dark'])

    # Combine all masks
    red_mask = mask1_light | mask2_light | mask1_dark | mask2_dark

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find red contours

    cnt = ()
    if contours:
        if sq_size == 1:
            for c in contours:
                peri = cv2.arcLength(c, True) # find the perimeter of contour
                approx = cv2.approxPolyDP(c, 0.015 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = max(w, h) / min(w, h)
                if w > min_size and h > min_size:
                    if 0.7 < aspect_ratio < 1.3:
                        cnt = cnt + (approx,)
            red_bbox = cv2.boundingRect(max(cnt, key=cv2.contourArea))
        elif sq_size > 1:
            red_bbox = cv2.boundingRect(max(contours, key=cv2.contourArea))
        output = image.copy() # paste bounding boxes onto image copy
        x, y, w, h = red_bbox
        # cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 3)  # Blue box for red square
    else:
        red_bbox = None

    if red_bbox:
        red_x, red_y, red_w, red_h = red_bbox

        # Define the leaf's bounding box as below the red square
        leaf_x = int(red_x - red_w / (bbox_multiplier))  # Expand width equally on both sides
        leaf_w = int(red_w * 2)  # Twice the width of the red square
        leaf_h = int(red_h * bbox_multiplier)  # Assume the leaf is roughly 2x the square's height
        
        if leaf_position == "above":
            leaf_y = int(red_y - int(red_h * bbox_multiplier) - padding)  # Position above the red square with some padding
        elif leaf_position == "below":
            leaf_y = int(red_y + red_h + padding)  # Start a bit below the red square

        leaf_bbox = (leaf_x, leaf_y, leaf_w, leaf_h)
        # cv2.rectangle(output, (leaf_x, leaf_y), (leaf_x+leaf_w, leaf_y+leaf_h), (0, 255, 0), 3)  # Green box for leaf
    
    # procoessing the rest of the image 
    image_hsv = image_hsv[max(leaf_y, 0):leaf_y+leaf_h, max(leaf_x, 0):leaf_x+leaf_w]

    # Create masks
    mask_green = cv2.inRange(image_hsv, color_ranges['lower_green'], color_ranges['upper_green'])
    mask_yellow = cv2.inRange(image_hsv, color_ranges['lower_yellow'], color_ranges['upper_yellow'])

    if 'lower_brown' in color_ranges:
        mask_brown = cv2.inRange(image_hsv, color_ranges['lower_brown'], color_ranges['upper_brown'])

    # Combine all masks
    mask = cv2.bitwise_or(mask_green, mask_yellow)
    if 'lower_brown' in color_ranges:
        mask = cv2.bitwise_or(mask, mask_brown)

    image_hsv[mask > 0] = [0, 0, 0] # Mutates green pixels to black
    image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    
    ### with basic masks now made, we want to tighten up the leaf bounding boxes before feeding it to SAM
    # convert image to grayscale and apply a threshold to it
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    t_glob_mean = skimage.filters.threshold_minimum(gray) # Thresholding 
    glob_mean = gray >= t_glob_mean # Apply threshold

    seed = np.copy(glob_mean)
    seed[1:-1, 1:-1] = False
    mask = glob_mean

    # morphological repair
    rec = skimage.morphology.reconstruction(seed, mask, method='dilation')
    binary_objects = rec.astype(bool)
    binary_filled = skimage.morphology.remove_small_holes(binary_objects, 6000)
    binary_filled = skimage.img_as_ubyte(binary_filled)

    edges = cv2.Canny(binary_filled, threshold1=50, threshold2=150)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    leaf_bbox_tight = cv2.boundingRect(max_contour)  # Get largest contour

    output = image.copy()
    x, y, w, h = leaf_bbox_tight
    buffer = 0
    leaf_array=np.array([x+leaf_x, y+leaf_y, x+w+leaf_x, y+h+leaf_y+buffer]) # buffer for chunks of leaf that might get cut off
    cv2.rectangle(output, (x+leaf_x, y+leaf_y), (x+w+leaf_x, y+h+leaf_y+buffer), (0, 255, 0), 3)  # Green box for leaf

    x, y, w, h = red_bbox
    box_array=np.array([x, y, x+w, y+h])

    cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 3)
    cv2.imwrite('./{0}/bboxes/{1}.jpg'.format(img_folder, img_name), output)

    return leaf_array, box_array

# apply SAM and threshold 
def SAM_image(img_src, img_folder, img_name, leaf_array, box_array, model_type, checkpoint, device):
    image = cv2.imread("./{0}".format(img_src))
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    predictor = SamPredictor(sam)

    predictor.set_image(image)

    masks, _, _ = predictor.predict(box=box_array)
    leaf_masks, _, _ = predictor.predict(box=leaf_array)

    mask_1 = masks[0].astype(np.uint8) * 255
    mask_2 = leaf_masks[0].astype(np.uint8) * 255
    combined_mask = np.logical_or(mask_1, mask_2).astype(np.uint8) * 255

    combined_mask = skimage.morphology.remove_small_holes(combined_mask.astype(bool), 6000)
    combined_mask = skimage.img_as_ubyte(combined_mask)
    # combined_mask = binary_closing(combined_mask, structure=disk(7))

    cv2.imwrite('./{0}/threshold/{1}.jpg'.format(img_folder, img_name), cv2.bitwise_not(combined_mask))

# doing some morphology repair. defaults are based on some analysis
def image_repair(img_folder, img_name, erode=7, sigma=1.6):
    image = cv2.imread('./{0}/threshold/{1}.jpg'.format(img_folder, img_name)) # get image for contouring
    image = skimage.img_as_ubyte(image)
    image = cv2.erode(image, np.ones((int(erode), int(erode)), np.uint8), iterations=1) # enlarges the image before blurring. should balance out with the blurring but double check contours match leaf margins (and lobing ratio. min should be 1.00)
    image = cv2.GaussianBlur(image, (51, 51), float(sigma)) # last number is sigmaX, changes SD of blurring effect. bigger equals more dispersed blurring. (x,x) is kernel size. must be positive and odd
    return image

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum and difference to sort points
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    
    return rect

# correcting for perspective/foreshortening issues
def perspective_correction(image, img_folder, img_name):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contours, heirarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        peri = cv2.arcLength(c, True) # find the perimeter of contour
        approx = cv2.approxPolyDP(c, 0.015 * peri, True) # create approximated polygon to determine rough shape of each contour (4 = square; >4 leaf) # was 0.015

        if cv2.arcLength(approx, True) >= 200 and len(approx) >= 3 and cv2.contourArea(c) < 1000000:
            if len(approx) == 4:
                pts = approx.reshape(-1, 2).astype("float32")
                pts = order_points(pts)
                h, w = image.shape[:2]  # Original image size... use the original dimensions for the transformation
                
                dst_pts = np.array([ [pts[0][0], pts[0][1]], [pts[1][0], pts[0][1]], [pts[1][0], pts[2][1]], [pts[0][0], pts[2][1]] ], dtype="float32") # Define the corrected quadrilateral inside the original image size

                M = cv2.getPerspectiveTransform(pts, dst_pts) # Compute transformation matrix
                corrected_img = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)) # Apply warp to the full image, keeping original dimensions

                cv2.imwrite('./{0}/perspective/{1}.jpg'.format(img_folder, img_name), corrected_img)
                image = cv2.imread('./{0}/perspective/{1}.jpg'.format(img_folder, img_name)) # get image for contouring
                image = skimage.img_as_ubyte(image)

                return image

# finding length of one edge of square
def find_segment_length(points, length="longest"):
    # Reshape the array to extract points in (x, y) format
    points = points.reshape(-1, 2)  # Flatten to shape (N, 2)
    
    # Calculate the distances between consecutive points
    segment_lengths = np.sqrt((points[1:, 0] - points[:-1, 0])**2 + 
                              (points[1:, 1] - points[:-1, 1])**2)
    
    if length=="longest":
        segment_length = np.max(segment_lengths) # Find the maximum length
    elif length=="shortest":
        segment_length = np.min(segment_lengths)
    return segment_length

# contouring for measurements and finding color fraction of the leaf
def contour_measurement(img_src, img_folder, img_name, image, sq_size, df, area_df, method, color_filter=True, red_low_tuple=(0, 90, 0), red_high_tuple=(15, 255,255), crop_tuple=(400, 200, 600)):
    img_col = cv2.imread("./{0}".format(img_src)) # get image for drawing contours (want regular pre-processed image)
    
    if method == "algo_pipeline":
        height, width, _ = image.shape
        x, y, h_crop = crop_tuple
        h=height-h_crop
        w=width-x
        img_col = img_col[y:h, x:w]
    elif method == "SAM":
        pass

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contours, heirarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find contours in thresholded image

    # for every contour in the image:
    for c in contours:
        peri = cv2.arcLength(c, True) # find the perimeter of contour
        approx = cv2.approxPolyDP(c, 0.015 * peri, True) # create approximated polygon to determine rough shape of each contour (4 = square; >4 leaf) # was 0.015

        # DEFAULT: 300; >3; 1000000
        if cv2.arcLength(approx, True) >= 200 and len(approx) >= 3 and cv2.contourArea(c) < 1000000: # removes small (artefacts) and huge (paper/image) contours
            # gets the contour that matches the square scale
            h_img = img_col.shape[0]
            max_y = max(approx, key=lambda point: point[0][1])[0][1]
            min_y = min(approx, key=lambda point: point[0][1])[0][1]

            # requires leaf to be have some extent below halfway 80% of the image, from the bottom, and be above 25% of the image, from the bottom
            is_below = max_y > (h_img * 0.15)
            is_above = min_y < (h_img * 0.85)

            if len(approx) == 4:
                short = find_segment_length(approx, length="shortest")
                long = find_segment_length(approx, length="longest")
                if ((short * long * 1.5) < (long**2)) == True: # square must be within range of expected area (of a square of area=long^2) to be counted as a square
                    pass

                else:
                    shape_sq = "square"
                    hull_sq = cv2.convexHull(approx)
                    area_sq_1 = cv2.contourArea(hull_sq)
                    longest_segment_length = find_segment_length(approx, length="longest")
                    area_sq_2 = longest_segment_length**2
                    
                    area_sq = area_sq_2

                    cv2.drawContours(img_col, [hull_sq], -1, (0, 255, 0), 2) # more precise contour of square... performs calculations using this
                    cv2.putText(img_col, shape_sq, (int(c[0][0][0]), int(c[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 3)

                    # append data to metadata frame
                    row = pd.DataFrame({
                        'Image_ID': [img_name],
                        'Shape': [shape_sq],
                        'Contour_Area': [np.nan],
                        'Hull_Area': [area_sq],
                        'Perimeter': [np.nan],
                        'Hull_Perimeter': [np.nan],
                        'Lobing_Ratio': [np.nan]
                    })

                    df = pd.concat(df.dropna(axis=1, how='all') for df in [df, row])
                    # df = pd.concat([df, row], ignore_index=True)

            # for the 'leaf' (hopefully only thing with >4 sides)
            elif (len(approx) > 4 and is_below == True) and (len(approx) > 4 and is_above== True):
                # print("image %s contains a leaf" % img_n)
                shape_c = "leaf"
                perimeter_circle = cv2.arcLength(c, True) # was approx
                hull_circle = cv2.convexHull(c)

                # hull_circle = cv2.approxPolyDP(hull_circle, 0.01, True)
                perimeter_hull_circle = cv2.arcLength(hull_circle, True)
                area_circle = cv2.contourArea(c)

                # smoothed_c = cv2.approxPolyDP(c, 6, True)
                # cv2.drawContours(img_col, [smoothed_c], -1, (255, 0, 0), 2)
                
                cv2.drawContours(img_col, [hull_circle], -1, (0, 255, 0), 3)
                cv2.drawContours(img_col, [c], -1, (255, 0, 0), 2)
                cv2.putText(img_col, shape_c, (int(c[0][0][0]), int(c[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 3)
                
                if perimeter_circle >= 100:
                    # add data to metadata frame 
                    row = pd.DataFrame({
                        'Image_ID': [img_name],
                        'Shape': [shape_c],
                        'Contour_Area': [area_circle],
                        'Hull_Area': [cv2.contourArea(hull_circle)],
                        'Perimeter': [perimeter_circle],
                        'Hull_Perimeter': [perimeter_hull_circle],
                        'Lobing_Ratio': [ round((perimeter_circle / perimeter_hull_circle), 2) if 'perimeter_circle' in globals() and perimeter_circle != 0 else np.nan ],
                    })
                    df = pd.concat(df.dropna(axis=1, how='all') for df in [df, row])
            
                # red-green color fraction
                if color_filter == True:
                    mask = np.zeros_like(img_col) # create a mask the size of the image
                    cv2.drawContours(mask, [c], -1, (255, 255, 255), thickness=cv2.FILLED) # draw a contour out of the mask equal to leaf shape
                    result = np.where(mask == 0, 255, img_col) # overlay mask and image

                    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    pixel_count = cv2.countNonZero(mask_gray)
                    # count red pixels and find fraction of total image
                    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV) # Convert the image to HSV

                    # determined using: https://i.sstatic.net/TSKh8.png - currently have this set to all non-green pixels
                    # newly determined using color_picker.ipynb -- picture above is still useful context but color wheel/slider is better for precision
                    mask_red1 = cv2.inRange(hsv, red_low_tuple, red_high_tuple) # red to orangey-yellow... the first number in second list (usually 20-30) is most sensitive one. was (0,10,0) & (25,255,255)
                    # mask_red2 = cv2.inRange(hsv, (90, 0, 0), (180, 255,255)) # was (90,0,0) & (180,255,255)
                    mask_red = mask_red1 # + mask_red2 # combine masks

                    red_count = cv2.countNonZero(mask_red) # count pixels in red range 
                    fraction_red = red_count / pixel_count # fraction of total pixels in mask

                    cv2.imwrite('./{0}/color_masks/{1}_red.jpg'.format(img_folder, img_name), mask_red)

    # append data to the stats data frame
    try:
        lobing_ratio = round((perimeter_circle / perimeter_hull_circle), 2)
        area = round((area_circle / area_sq), 4) * sq_size**2
        prop_red = round(fraction_red, 3)
    except NameError:
        lobing_ratio = np.nan
        area = np.nan
        prop_red = np.nan

    area_row = pd.DataFrame({
        'Image_ID': [img_name],
        'Lobing_Ratio': [ lobing_ratio ],
        'Area': [ area ],
        'Prop_Red': [ prop_red ] # if color_filter == False, this will still run
    })

    # area_df = pd.concat([area_df, area_row], ignore_index=True)
    area_df = pd.concat(df.dropna(axis=1, how='all') for df in [area_df, area_row])
    cv2.imwrite('./{0}/processed/{1}.jpg'.format(img_folder, img_name), img_col)

    return df, area_df


def algo_pipeline(img_src, img_folder, img_name, saturation, crop_tuple=(400, 200, 600), black_white_tuple=(20, 200), small_hole_size=6000):
    image = cv2.imread("./{0}".format(img_src))
    
    # crop edges of photo
    height, width, channel = image.shape
    x, y, h_crop = crop_tuple
    h=height-h_crop
    w=width-x
    image = image[y:h, x:w]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Mask for colored pixels
    threshold_black, threshold_white = black_white_tuple
    color_mask = (hsv[:, :, 1] > saturation) & (hsv[:, :, 2] > threshold_black) & (hsv[:, :, 2] < threshold_white)
    
    # Create the output image: start with all white
    glob_mean = np.full_like(image, 255) # All white
    glob_mean[color_mask] = [0, 0, 0] # Set colored pixels to black
    
    seed = np.copy(glob_mean) # Seed and mask for morphological reconstruction
    seed[1:-1, 1:-1] = False  # Ensure the interior is set to False, not the minimum value
    mask = glob_mean

    # Perform morphological reconstruction
    rec = skimage.morphology.reconstruction(seed, mask, method='dilation')
    binary_objects = rec.astype(bool)
    binary_filled = skimage.morphology.remove_small_holes(binary_objects, small_hole_size)

    binary_filled = skimage.img_as_ubyte(binary_filled)
    skimage.io.imsave("./{0}/threshold/{1}.jpg".format(img_folder, img_name), skimage.img_as_ubyte(binary_filled))

    return binary_filled

