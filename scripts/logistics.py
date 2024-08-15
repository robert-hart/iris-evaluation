import os
from tqdm import tqdm
import cv2
import numpy as np

from generated_iris_evaluation import MasekGaborResponse, CoordinateTransformer, otsu_thresh, segmentation

#main processing, done in parallel
def process_dataset(arguments):
    #shared data
    process_parameters = arguments[0][0] #TODO expand this out
    gabor_kernels = arguments[0][1]

    #image specific data
    image = arguments[1][0]
    image_name = arguments[1][1]
    target = arguments[1][2]

    #make clahe; in this part of code because not serializable
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    img_normalize = None
    img_normalized_color = None
    img_normalized_gray = None

    #TODO try / except = quality control for unflattened images
    if process_parameters[0] == 1:
        try:
            img_thresh = otsu_thresh(image)
            limbic_circle, pupillary_circle = segmentation(img_thresh)
            limbic_radius = np.rint(limbic_circle[0][2]) - 2
            pupillary_radius = np.rint(pupillary_circle[0][2]) + 2 
            center = np.rint((limbic_circle[0][0] + limbic_circle[0][1] + pupillary_circle[0][0] + pupillary_circle[0][1])/4)
            img_normalize = CoordinateTransformer(image, pupillary_radius, limbic_radius, center, process_parameters[2], process_parameters[3])
            img_normalized_color = img_normalize.transformed
        except:
            target = target + '_failed'
            with open(f'{target}.txt', 'a') as file:
                file.write(f'{image_name} failed quality control\n')
            if not process_parameters[4]:
                img_normalize = CoordinateTransformer(image, 45, 83, image.shape[0]//2, process_parameters[2], process_parameters[3]) #estimated center of the eye for failed segmentation
                img_normalized_color = img_normalize.transformed
            else:
                return
    
    #grayscale the image
    img_normalized_gray = cv2.cvtColor(img_normalized_color, cv2.COLOR_BGR2GRAY)
    img_normalized_gray = clahe.apply(img_normalized_gray)
    
    GaborReponse = MasekGaborResponse(img_normalized_gray, gabor_kernels) #pass the filter bank to get the filter responses
    gabor_responses = GaborReponse.quantized_responses
    gabor_barcodes = GaborReponse.iris_barcode

    iris_code = []
    
    if process_parameters[5]: #if verbose
        os.makedirs(target, exist_ok=True)
        cv2.imwrite(f'{target}/[ORIGINAL]-{image_name}.png', image)  #image
        cv2.imwrite(f'{target}/[NORMALIZED-COLOR]-{image_name}.png', img_normalized_color) #image color normalized
        cv2.imwrite(f'{target}/[NORMALIZED-GRAY]-{image_name}-normalized-gray.png', img_normalized_gray) #image gray normalized

    for i in range(gabor_responses.shape[0]):
        iris_code.append(gabor_responses[i])
        if process_parameters[5]: #if verbose
            real_imaginary = None
            for j in range(gabor_responses.shape[1]):
                if j % 2 == 0:
                    real_imaginary = "real"
                else:
                    real_imaginary = "imaginary"
                cv2.imwrite(f'{target}/[BARCODE+FILTER{i+1}+{real_imaginary}]-{image_name}.png', gabor_barcodes[i][j]) #save the barcocde
                np.save(f'{target}/[IRISCODE+FILTER{i+1}+{real_imaginary}-{image_name}.npy', gabor_barcodes[i][j]) #save the code
            
    iris_code = np.array(iris_code, dtype='?')
    np.save(f'{target}.npy', iris_code)

    return