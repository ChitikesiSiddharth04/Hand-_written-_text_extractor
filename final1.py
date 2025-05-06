import cv2
import numpy as np
from pdf2image import convert_from_path
import pandas as pd
from PIL import Image
# import matplotlib.pyplot as plt
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from util import is_blank_space,processor,model,remove_nearby_contours,remove_lines,field_detection,field_detection_2

def recognize_text_trocr(roi_image):
    if is_blank_space(remove_lines(roi_image)):
        return "",1 # Skip OCR for blank regions
    pil_image = Image.fromarray(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Calculate confidence scores for each token
    confidence_scores = []
    
    # Get logits to compute probabilities
    with torch.no_grad():
        outputs = model(pixel_values, decoder_input_ids=generated_ids[:, :-1])  # Provide decoder inputs
        logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)

    # Get predicted token IDs (only for the generated sequence)
    predicted_token_ids = torch.argmax(logits[:, :-1], dim=-1)  # Exclude last token for decoder input

    # Iterate through the sequence to gather confidence scores
    for i in range(predicted_token_ids.size(1)):  # Iterate over tokens in the generated sequence
        token_logits = logits[0, i]  # Get logits for the i-th token in the batch
        probs = torch.nn.functional.softmax(token_logits, dim=-1)  # Compute softmax probabilities
        token_id = predicted_token_ids[0, i].item()  # Get the token ID
        token_prob = probs[token_id].item()  # Get the probability for the predicted token
        confidence_scores.append(token_prob)
    # Optionally map token scores to individual characters
    char_confidences = map_token_scores_to_characters(generated_text, confidence_scores)

    return generated_text, char_confidences

def recognize_text_trocr_2(roi_image):
    # Pre-process image for model input
    pil_image = Image.fromarray(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Calculate confidence scores for each token
    confidence_scores = []
    
    # Get logits to compute probabilities
    with torch.no_grad():
        outputs = model(pixel_values, decoder_input_ids=generated_ids[:, :-1])  # Provide decoder inputs
        logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)

    # Get predicted token IDs (only for the generated sequence)
    predicted_token_ids = torch.argmax(logits[:, :-1], dim=-1)  # Exclude last token for decoder input

    # Iterate through the sequence to gather confidence scores
    for i in range(predicted_token_ids.size(1)):  # Iterate over tokens in the generated sequence
        token_logits = logits[0, i]  # Get logits for the i-th token in the batch
        probs = torch.nn.functional.softmax(token_logits, dim=-1)  # Compute softmax probabilities
        token_id = predicted_token_ids[0, i].item()  # Get the token ID
        token_prob = probs[token_id].item()  # Get the probability for the predicted token
        confidence_scores.append(token_prob)
    # Optionally map token scores to individual characters
    char_confidences = map_token_scores_to_characters(generated_text, confidence_scores)

    return generated_text, char_confidences

def map_token_scores_to_characters(text, token_scores):
    char_confidences = []
    # Tokenize the text to match tokenization method used in the model
    tokens = processor.tokenizer.tokenize(text)

    for token, score in zip(tokens, token_scores):
        # Calculate the number of characters in the token
        for char in token:
            # You might consider a more nuanced way to assign scores if necessary
            char_confidences.append(score)  # Still uniform, but ensuring it maps correctly to tokens
    return char_confidences




#IMAGE FOR CHECK BOXES
images = convert_from_path("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\Different handwritings\\File 6.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:\\Program Files (x86)\\poppler-24.07.0\\Library\\bin")  # Set dpi for higher resolution
# images = convert_from_path("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\Different handwritings\\File 2 - Copy.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:\\Program Files (x86)\\poppler-24.07.0\\Library\\bin")  # Set dpi for higher resolution
# images = convert_from_path("C:/Users/DELL/Downloads/File - 004.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:/Users/DELL/Downloads/poppler-24.07.0/Library/bin")  # Set dpi for higher resolution
image_np = np.array(images[0])
image_np=cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR)
img=cv2.cvtColor(image_np,cv2.COLOR_RGB2GRAY)
_, img_bin = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
img_bin=cv2.dilate(img_bin,np.ones((3,3),np.uint8),iterations=1)
line_min_width = 15  # Adjust based on your checkbox size
kernal_h = np.ones((1, line_min_width), np.uint8)
kernal_v = np.ones((line_min_width, 1), np.uint8)
img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)
img_bin_combined = cv2.add(img_bin_h, img_bin_v)

_, img_bin_cb = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
img_bin_cb =cv2.dilate(img_bin_cb,np.ones((1,1),np.uint8),iterations=1)
line_min_width = 15  # Adjust based on your checkbox size
kernal_h = np.ones((1, line_min_width), np.uint8)
kernal_v = np.ones((line_min_width, 1), np.uint8)
img_bin_h_cb = cv2.morphologyEx(img_bin_cb, cv2.MORPH_OPEN, kernal_h)
img_bin_v_cb = cv2.morphologyEx(img_bin_cb, cv2.MORPH_OPEN, kernal_v)
img_for_checkbox = cv2.add(img_bin_h_cb, img_bin_v_cb)

contours, _ = cv2.findContours(img_bin_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_cb, _ = cv2.findContours(img_for_checkbox, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

text = {}


def remove_nearby_contours(contours, min_distance=1):
    filtered_contours = []
    to_remove = set()
    
    for i, contour1 in enumerate(contours):
        if i in to_remove:
            continue
        
        x1, y1, w1, h1 = cv2.boundingRect(contour1)
        keep = True
        
        for j, contour2 in enumerate(contours):
            if i != j and j not in to_remove:
                x2, y2, w2, h2 = cv2.boundingRect(contour2)
                
                # Calculate the distance between the centers of the bounding boxes
                center1 = np.array([x1 + w1 / 2, y1 + h1 / 2])
                center2 = np.array([x2 + w2 / 2, y2 + h2 / 2])
                distance = np.linalg.norm(center1 - center2)
                # print(distance, i, j)
                
                if distance <= min_distance:
                    keep = False
                    to_remove.add(j)
                    break
        
        if keep:
            filtered_contours.append(contour1)

    # print(to_remove)
    contours = [c for i, c in enumerate(contours) if i not in to_remove]
    return contours



def is_blank_space(roi_image, white_pixel_threshold=0.98):
    # Convert the ROI to grayscale
    # gray_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the grayscale image to get a binary image
    _, binary_image = cv2.threshold(roi_image, 200, 255, cv2.THRESH_BINARY)
    
    # Calculate the total number of pixels in the region
    total_pixels = binary_image.size
    
    # Count the number of white pixels (255 value)
    white_pixels = np.sum(binary_image == 255)
    
    # Calculate the percentage of white pixels
    white_pixel_ratio = white_pixels / total_pixels

    # cv2.imshow("image",binary_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print(white_pixel_ratio)

    
    # If the percentage of white pixels exceeds the threshold, declare it as blank
    if white_pixel_ratio > white_pixel_threshold:
        return True  # The region is blank
    else:
        return False  # The region contains text or is not blank
    
def remove_lines(img):
    # print(img.shape)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    _, img_bin = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    img_bin_c=cv2.dilate(img_bin,np.ones((1,1),np.uint8),iterations=1)
    line_min_width = 150# Adjust based on your checkbox size
    kernal_h = np.ones((1, 70), np.uint8)
    kernal_v = np.ones((line_min_width, 1), np.uint8)
    img_bin_h = cv2.morphologyEx(img_bin_c, cv2.MORPH_OPEN, kernal_h)
    img_bin_v = cv2.morphologyEx(img_bin_c, cv2.MORPH_OPEN, kernal_v)
    img_bin_combined_c = cv2.add(img_bin_h, img_bin_v)
    # print(img_bin_combined_c.shape)
    img_bin_c=255-img_bin_c
    img_bin_c_bgr = cv2.cvtColor(img_bin_c, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("image",img_bin_c_bgr)
    img_bin_combined_c_bgr = cv2.cvtColor(img_bin_combined_c, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("Image",img_bin_combined_c_bgr)
   
    result=cv2.bitwise_or(img_bin_c_bgr, img_bin_combined_c_bgr)
    # cv2.imshow("result",result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return result


def is_ticked(roi):
    roi=cv2.resize(roi,(50,73))
    _, binary = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("binary",roi)
    total_pixels = binary.size
    black_pixels = np.sum(binary == 255)
    black_pixel_ratio = black_pixels / total_pixels
    # print(black_pixel_ratio)
    return black_pixel_ratio > 0.25


# def recognize_text_trocr(roi_image):
#     roi_image=remove_lines(roi_image)
#     if is_blank_space(roi_image):
#         return ""  # Skip OCR for blank regions
#     pil_image = Image.fromarray(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
#     pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
#     generated_ids = model.generate(pixel_values)
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     return generated_text

# def recognize_text_trocr_2(roi_image):
#     # if is_blank_space(roi_image):
#     #     return ""  # Skip OCR for blank regions
#     pil_image = Image.fromarray(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
#     pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
#     generated_ids = model.generate(pixel_values)
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     return generated_text


def highest_education_level(contours,img):
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
        # cv2.putText(image_np,str(i_cb),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        roi_image = img[y:y+h, x:x+w]
    # print(roi_image.shape)
        if is_ticked(roi_image):
        #   cv2.imshow("Feild",img[y:y+h,x+w:x+w+200])
        #   cv2.waitKey(0)
          field=recognize_text_trocr_2(img[y:y+h,x+w:x+w+200])
        #   print(field)
          return field
    return "NOTHING IS CHECKED"

def with_dependent_children(contours, img):
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
        # cv2.putText(image_np,str(i_cb),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        roi_image = img[y:y+h, x:x+w]
        
        # Check if the region is ticked
        if is_ticked(roi_image):
            # # Debugging: Show the field image
            # cv2.imshow("Field", img[y:y+h, x+w:x+w+50])
            # cv2.waitKey(0)  # Wait for a key press to proceed (for debugging)
            
            # Recognize text next to the checkbox
            field = recognize_text_trocr_2(img[y:y+h, x+w:x+w+50])
            # print(f"Recognized field text: '{field}'")
            # print(field)
            
            # Strip whitespace and use case-insensitive comparison
            if "no" in field.strip().lower():
                return "No"
            else:
                # Adjusted ROI for the second text extraction
                # cv2.imshow("Field",img[y:y+h, x+w+300:x+w+560])
                # cv2.waitKey(0)
                return recognize_text_trocr(img[y:y+h, x+w+300:x+w+560])
    
    return "NOTHING IS TICKED"

def marital_status(contours_mt,img):
    ct = contours_mt
    # print(ct)
    for c in ct:
        # print(c)
        x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
        # cv2.putText(image_np,str(i_cb),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        roi_image = img[y:y+h, x:x+w]
        # cv2.imshow(roi_image)
        # cv2.waitKey(0)
        if is_ticked(roi_image):
            # print(i)
            # cv2.imshow("Feild",img[y:y+h,x+w:x+w+100])
            # cv2.waitKey(0)
            field=recognize_text_trocr_2(img[y:y+h,x+w:x+w+100])
            # print(field)
            return field
    # print(i)
    return "NOTHING IS CHECKED"

def gender(contours, img):
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
        # cv2.putText(image_np,str(i_cb),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        roi_image = img[y:y+h, x:x+w]
        
        # # Display each detected contour region for debugging
        # cv2.imshow("roi", roi_image)
        # cv2.waitKey(0)  # Wait for a key press to proceed

        # Check if the region is ticked
        if is_ticked(roi_image):
            # If ticked, display the region next to the checkbox
            text_roi = img[y:y+h, x+w:x+w+67]
            field=recognize_text_trocr_2(text_roi)
            # print(field)
            # cv2.imshow("Field", text_roi)
            # cv2.waitKey(0)  # Wait for a key press to proceed

            # Return the recognized text
            return field
    
    return "NOTHING IS CHECKED"

def race(contours,img):
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
        # cv2.putText(image_np,str(i_cb),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        roi_image=img[y:y+h,x:x+w]

        if is_ticked(roi_image):
            field=recognize_text_trocr_2(img[y:y+h,x+w:x+w+80])
            # cv2.imshow("field",img[y:y+h,x+w:x+w+80])
            # cv2.waitKey(0)
            # print(field)
            if "others" in field.strip().lower():
                # cv2.imshow("Others",img[y:y+h,x+w+210:x+w+600])
                # cv2.waitKey(0)
                return recognize_text_trocr(img[y:y+h,x+w+210:x+w+600])
            else:
                return field
        
    return "NOTHING IS CHECKED"

def residency_status(contours,img,image_np):
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
        # cv2.putText(image_np,str(i_cb),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        roi_image = img[y:y+h, x:x+w]
    # print(roi_image.shape)
        if is_ticked(roi_image):
            # cv2.imshow("Feild",img[y:y+h,x+w:x+w+200])
            feild=recognize_text_trocr_2(image_np[y:y+h,x+w:x+w+250])
            # print(feild)
            if "non-resident" in feild.strip().lower():
                # cv2.imshow("non resident",image_np[y+h+10:y+h+30,x:x+w+620])
                # cv2.waitKey(0)
                return recognize_text_trocr(image_np[y+h+10:y+h+30,x:x+w+620])
            else:
                return feild
    return "NOTHING IS CHECKED"

def other_identification_no(contours,img):
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        roi_image=img[y:y+h,x:x+w]
        iden_no=recognize_text_trocr(img[y+h+10:y+h+40,x+w+130:x+w+550])   
        if is_ticked(roi_image):
            feild=recognize_text_trocr_2(img[y:y+h,x+w:x+w+130])
                # i,j,k,l=cv2.boundingRect(co)
            # print(feild)
            # cv2.imshow("field",img[y:y+h,x+w:x+w+130])
            # cv2.imshow("Iden no:",img[y+h+10:y+h+40,x+w+130:x+w+550])
            # cv2.waitKey(0)
        
            return feild,iden_no
                # return field
    return "NOTHING IS CHECKED",iden_no
# Define constraints for filtering checkboxes\

def salutation(contours,img):
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
        # cv2.putText(image_np,str(i_cb),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        roi_image=img[y:y+h,x:x+w]

        if is_ticked(roi_image):
            field=recognize_text_trocr_2(img[y:y+h,x+w:x+w+80])
            print(field)
            # cv2.imshow("Field",field)
            # cv2.waitKey(0)
            if "others" in field.strip().lower():
                # cv2.imshow("Others",img[y:y+h,x+w+220:x+w+550])
                # cv2.waitKey(0)
                return recognize_text_trocr(img[y:y+h,x+w+220:x+w+550])
            else:
                
                return field
        
    return "NOTHING IS CHECKED"

def existing(contours,img):
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
        # cv2.putText(image_np,str(i_cb),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        roi_image=img[y:y+h,x:x+w]
        if is_ticked(roi_image):
            # cv2.imshow("Field",img[y:y+h,x+w:x+w+100])
            # cv2.waitKey(0)
            field=recognize_text_trocr_2(img[y:y+h,x+w:x+w+100])
            # print(field)
            return field
    return "NOTHING IS CHECKED"
result=[]
def products(contours,img):
    for i,c in enumerate(contours):
        x,y,w,h=cv2.boundingRect(c)
        roi_image=img[y:y+h,x:x+w]
        if is_ticked(roi_image):
            # if i==0 or i==2 or i==3:
            # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
            # cv2.putText(image_np,str(i_cb),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            # cv2.imshow("Field",img[y:y+h,x+w:x+w+300])
            # cv2.waitKey(0)
            result.append(recognize_text_trocr_2(img[y:y+h,x+w:x+w+300]) )
            # print(result)
        # print(result)   
            # elif i==1 or i==4 :
    return result

cb_c_gender = []
cb_c_children = []
cb_c_highest_education = []
cb_c_marital = []
cb_c_race = []
cb_c_residency = []
cb_c_other_identification = []
cb_c_sal = []
cb_c_ec = []
cb_c_product = []

field_contours=[]
for i_cb,contour_cb in enumerate(contours_cb):
    x, y, w, h = cv2.boundingRect(contour_cb)
    aspect_ratio = w / float(h)
    if 0.8 <= aspect_ratio <= 1.5:  # Aspect ratio close to 1 for squares
        # Filter based on size to exclude large or very small boxes
        if 5<= w <=20  and 5<= h <= 20:  # Adjust these size thresholds as needed
            area = cv2.contourArea(contour_cb)
            if area >250 and area <= 450 and y > 250:
                # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
                # cv2.putText(image_np,str(i_cb),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
                if y >= 250 and y <= 450:
                    cb_c_product.append(contour_cb)
                elif y > 450 and y <= 550 and x <735:
                    cb_c_ec.append(contour_cb)
                elif y > 550 and y <= 670 and x <735:
                    cb_c_sal.append(contour_cb)
                elif y >= 875 and y <= 1030 and x <720:
                    cb_c_other_identification.append(contour_cb)
                elif y >= 470 and y <= 665 and x >710:
                    cb_c_residency.append(contour_cb)
                elif y >= 680 and y <= 800 and x >710:
                    cb_c_race.append(contour_cb)
                elif y >= 950 and y <= 1050 and x >710:
                    # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
                    # cv2.putText(image_np,str(i_cb),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
                    cb_c_marital.append(contour_cb)
                    # print(contour_cb)
                    # print(i_cb)
                elif y >= 1153 and x >710:
                    cb_c_highest_education.append(contour_cb)
                elif y >= 1066 and y <= 1155 and x >710:
                    cb_c_children.append(contour_cb)
                elif y >= 800 and y <= 845 and x >860:
                    cb_c_gender.append(contour_cb)





# result=products(cb_c_product,img)
# if not result:
#     text["Products Interested in"] = "NOTHING IS CHECKED"
# else:
#     text["Products Interested in"]= set(result)

# # print(text)

# text["Highest_education_level"] = highest_education_level(cb_c_highest_education,img)
# text["With Dependent Children"]= with_dependent_children(cb_c_children,img)
# # print(len(cb_c_marital))
# text["Marital Status"] = marital_status(cb_c_marital,img)
# # print(text)
# # plt.imshow(image_np,cmap = 'gray')
# # plt.show()
# text["Gender"] = gender(cb_c_gender,img)
# text["Race"] = race(cb_c_race,img)
# text["Residency Status"] = residency_status(cb_c_residency,img,image_np)
# text["Other Identification no"] = other_identification_no(cb_c_other_identification,img)
# text["Salutation"] = salutation(cb_c_sal,img)
# # # text[field_contours)

# text["Are you an existing customer"] = existing(cb_c_ec,img)
# result=products(checkbox_contours[42:48],img)
# if not result:
#     text["Field_Products Interested in"] = "NOTHING IS CHECKED"
# else:
#     text["Field_Products Interested in"]= ' ' .join(i for i in result)


# #                 checkbox_contours.append(i_cb)

# # # # Draw the detected checkboxes on the original image for visualization
# # output_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# # contours_needed = [c for i, c in enumerate(contours_cb) if i in checkbox_contours]
# # # contour=checkbox_contours[42:48]
# # for i,c in enumerate(contours_needed):
# #     x,y,w,h=cv2.boundingRect(c)
# #     cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
# #     cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
# #     print(recognize_text_trocr(image_np[y:y+h,x+w:x+w+300]))
# #     cv2.imshow("Feild",image_np[y:y+h,x+w:x+w+300])
# # output_image=cv2.resize(output_image,(550,850))
# # # cv2.imwrite("detected_fields_page_1.png",output_image)
# # cv2.imshow("Detected feilds",output_image)


contours=remove_nearby_contours(contours)
# l_i = []
# # print(contours)
# # text = {}
fields=[]
for i,c in enumerate(contours):
    x,y,w,h=cv2.boundingRect(c)
    area=cv2.contourArea(c)
    if area>16000 and area<20000 and y>500:
        # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
        # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        # roi_image = img[y:y+h, x:x+w]
        # generated_text = recognize_text_trocr(roi_image)
        # text[f'Field_{i}'] = generated_text
        fields.append(c)
extra=[]
for i,c in enumerate(contours):
    x,y,w,h=cv2.boundingRect(c)
    area=cv2.contourArea(c)
    if area>10000 and area<15000 and y>500:
        # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
        # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        # roi_image = img[y:y+h, x:x+w]
        # generated_text = recognize_text_trocr(roi_image)
        # text[f'Field_{i}'] = generated_text
        extra.append(c)

# for i,c in enumerate(fields):
#         x,y,w,h=cv2.boundingRect(c)
#         cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#         cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

# for i,c in enumerate(extra):
#         x,y,w,h=cv2.boundingRect(c)
#         cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#         cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

# image_np=cv2.resize(image_np,(500,800))
# cv2.imshow("Image",image_np)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cs={}
for i,c in enumerate(fields):
    x,y,w,h=cv2.boundingRect(c)
    roi=img[y:y+h,x:x+w]
    if i==0:
        # print(recognize_text_trocr(roi))
        g,c=recognize_text_trocr(roi)
        cs[g]=c
        text["Visa Type"]=g
        
      
    if i==1:
        continue
    if i==2:
        # print(recognize_text_trocr(roi))
        g,c=recognize_text_trocr(roi)
        cs[g]=c
        text["Passport Number"]=g
    if i==3:
        # print(recognize_text_trocr(roi))
        g,c=recognize_text_trocr(roi)
        cs[g]=c
        text[""]=g
    if i==4:
        # print(recognize_text_trocr(roi))
        g,c=recognize_text_trocr(roi)
        text["Name"]=g
        cs[g]=c

        

for i,c in enumerate(extra):
    x,y,w,h=cv2.boundingRect(c)
    roi=img[y:y+h,x:x+w]
    if i==0:
        g,c=recognize_text_trocr(roi)
        cs[g]=c
        text["Visa Expiry Date"]=g
        # text["Visa Expiry date"]=recognize_text_trocr(roi)
        # print(recognize_text_trocr(roi))
        
        
        
    if i==1:
        g,c=recognize_text_trocr(roi)
        cs[g]=c
        text["Date of birth"]=g
        # text["Date Of Birth"]=recognize_text_trocr(roi)
        # print(recognize_text_trocr(roi))
    if i==2:
        g,c=recognize_text_trocr(roi)
        cs[g]=c
        text["Passport Expiry date"]=g
        # text["Passport Expiry Date"]=recognize_text_trocr(roi)
        # print(recognize_text_trocr(roi))
        


# image_np=cv2.resize(image_np,(500,800))
# cv2.imshow("Image",image_np)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
        # l_i.append(i)
    # print(i,c)
# contours_needed = [c for i, c in enumerate(contours) if i in l_i]
# print([i for i,c in enumerate(contours_needed)])
# for i,c in enumerate(contours_needed):
#     x,y,w,h=cv2.boundingRect(c)
#     cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#     cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)



# text["Field_Highest_education_level"] = highest_education_level(checkbox_contours[0:8],img)
# text["Field_With Dependent CHildren"]= with_dependent_children(checkbox_contours[8:10],img)
# text["Field_Marital Status"] = marital_status(checkbox_contours[14:20],img)
# text["Field_Gender"] = gender(checkbox_contours[25:27],img)
# text["Field_Race"] = race(checkbox_contours[27:31],img)
# text["Field_Residency Status"] = residency_status(checkbox_contours[36:40],img)
# text["Field_Other Identification no"] = other_identification_no(checkbox_contours[20:25],img)
# text["Field_Salutation"] = salutation(checkbox_contours[31:36],img)
# # text[field_contours)

# text["Field_Are you an existing customer"] = existing(checkbox_contours[40:42],img)
# result=products(checkbox_contours[42:48],img)
# if not result:
#     text["Field_Products Interested in"] = "NOTHING IS CHECKED"
# else:
#     text["Field_Products Interested in"]= ' ' .join(i for i in result)



# print(text)

# # order_text = dict(sorted(text.items(),reverse = True))

df = pd.DataFrame(list(cs.items()),columns = ['Text','Probablities' ])
print(df)

