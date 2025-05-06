import cv2
import numpy as np
from pdf2image import convert_from_path
import pandas as pd
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

import time

processor = TrOCRProcessor.from_pretrained("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\AJK_CODE\\EXECUTABLE\\trocr_processor")
model = VisionEncoderDecoderModel.from_pretrained("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\AJK_CODE\\EXECUTABLE\\trocr_model")

start_time = time.time()
#IMAGE FOR CHECK BOXES
# images = convert_from_path("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\CIMB AI LABS.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:\\Program Files (x86)\\poppler-24.07.0\\Library\\bin")  # Set dpi for higher resolution
# images = convert_from_path("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\Different handwritings\\fine 1 - Copy.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:\\Program Files (x86)\\poppler-24.07.0\\Library\\bin")  # Set dpi for higher resolution
images = convert_from_path("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\Different handwritings\\File 5.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:\\Program Files (x86)\\poppler-24.07.0\\Library\\bin")  # Set dpi for higher resolution

image_np = np.array(images[10])
image_np=cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR)
img=cv2.cvtColor(image_np,cv2.COLOR_RGB2GRAY)
_, img_bin = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
img_bin=cv2.dilate(img_bin,np.ones((2,2),np.uint8),iterations=1)
line_min_width = 15  # Adjust based on your checkbox size
kernal_h = np.ones((1, line_min_width), np.uint8)
kernal_v = np.ones((line_min_width, 1), np.uint8)
img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)
img_bin_combined = cv2.add(img_bin_h, img_bin_v)

image_np = np.array(images[10])
image_np=cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR)
img=cv2.cvtColor(image_np,cv2.COLOR_RGB2GRAY)
_, img_bin = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
img_bin_c=cv2.dilate(img_bin,np.ones((1,1),np.uint8),iterations=1)
line_min_width = 15  # Adjust based on your checkbox size
kernal_h = np.ones((1, line_min_width), np.uint8)
kernal_v = np.ones((line_min_width, 1), np.uint8)
img_bin_h = cv2.morphologyEx(img_bin_c, cv2.MORPH_OPEN, kernal_h)
img_bin_v = cv2.morphologyEx(img_bin_c, cv2.MORPH_OPEN, kernal_v)
img_bin_combined_c = cv2.add(img_bin_h, img_bin_v)


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
    
    # Remove contours that are too close from the original list
    print(to_remove)
    contours = [c for i, c in enumerate(contours) if i not in to_remove]
    # print(len(contours))
    return contours

#IMAGE FOR FIELDS
def is_ticked(roi):
    roi=cv2.resize(roi,(50,73))
    _, binary = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)
    total_pixels = binary.size
    black_pixels = np.sum(binary == 255)
    black_pixel_ratio = black_pixels / total_pixels
    # print(black_pixel_ratio)
    return black_pixel_ratio > 0.229


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


def recognize_text_trocr(roi_image):
    if is_blank_space(remove_lines(roi_image)):
        return "",1 # Skip OCR for blank regions
    pil_image = Image.fromarray(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values

    decoder_input_ids = torch.zeros((1, 1), dtype=torch.long)

    # Run inference to get logits without labels
    with torch.no_grad():
        outputs = model(pixel_values,decoder_input_ids=decoder_input_ids)
    
    # Extract logits for the generated sequence
    logits = outputs.logits[0]  # Shape: (sequence_length, vocab_size)

    # Get predicted token IDs by taking the highest probability at each step
    predicted_token_ids = torch.argmax(logits, dim=-1)
    generated_text = processor.batch_decode(predicted_token_ids.unsqueeze(0), skip_special_tokens=True)[0]

    # Calculate confidence scores for each token
    confidence_scores = []
    for token_id, token_logits in zip(predicted_token_ids, logits):
        # Apply softmax to token logits to get probabilities
        probs = torch.nn.functional.softmax(token_logits, dim=-1)
        token_prob = probs[token_id].item()  # Probability of the selected token
        confidence_scores.append(token_prob)

    # Optionally map token scores to individual characters
    char_confidences = map_token_scores_to_characters(generated_text, confidence_scores)
    
    return generated_text, char_confidences

def recognize_text_trocr_2(roi_image):
    # Pre-process image for model input
    pil_image = Image.fromarray(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values

    decoder_input_ids = torch.zeros((1, 1), dtype=torch.long)

    # Run inference to get logits without labels
    with torch.no_grad():
        outputs = model(pixel_values,decoder_input_ids=decoder_input_ids)
    
    # Extract logits for the generated sequence
    logits = outputs.logits[0]  # Shape: (sequence_length, vocab_size)

    # Get predicted token IDs by taking the highest probability at each step
    predicted_token_ids = torch.argmax(logits, dim=-1)
    generated_text = processor.batch_decode(predicted_token_ids.unsqueeze(0), skip_special_tokens=True)[0]

    # Calculate confidence scores for each token
    confidence_scores = []
    for token_id, token_logits in zip(predicted_token_ids, logits):
        # Apply softmax to token logits to get probabilities
        probs = torch.nn.functional.softmax(token_logits, dim=-1)
        token_prob = probs[token_id].item()  # Probability of the selected token
        confidence_scores.append(token_prob)

    # Optionally map token scores to individual characters
    char_confidences = map_token_scores_to_characters(generated_text, confidence_scores)
    
    return generated_text, char_confidences

def map_token_scores_to_characters(text, token_scores):
    # Expand token scores across individual characters for finer confidence mapping
    char_confidences = []
    for token, score in zip(text.split(), token_scores):
        char_confidences.extend([score] * len(token))  # Apply token score to each character
    return char_confidences


def field_detection(contour,img,image_np):
    x,y,w,h=cv2.boundingRect(contour)
    field_name=img[y-30:y,x-50:x+w-100]
    # cv2.imshow("Field",field_name)
    # cv2.waitKey(0)
    text,c=recognize_text_trocr(image_np[y:y+h,x:x+w])
    print(text,c)
    field,c=recognize_text_trocr_2(field_name)
    print(field,c)
    return text,field


def field_detection_2(contour, img,image_np):
    x, y, w, h = cv2.boundingRect(contour)
    field_name = img[y:y+h, x-100:x]

    try:
        # Check if field_name is empty
        if field_name is None or field_name.size == 0:
            raise ValueError("Field name region is empty or not loaded")
        
        text,c = recognize_text_trocr(image_np[y:y+h, x:x+w])
        print(text,c)
        field,c= recognize_text_trocr_2(field_name)
        print(field,c)
    
    except ValueError as ve:
        # print(f"Error: {ve}")
        text = recognize_text_trocr(image_np[y:y+h, x:x+w])
        field = recognize_text_trocr_2(img[y-30:y,x-40:x+w-100])  # Handle the empty case with a default value or message

    return text, field


def salutation(contours,img):
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        roi_image=img[y:y+h,x:x+w]

        if is_ticked(roi_image):
            field=recognize_text_trocr_2(img[y:y+h,x+w:x+w+80])
            if "others" in field.strip().lower():
                return recognize_text_trocr_2(img[y:y+h,x+w+220:x+w+550])
            else:
                return field
        
    return "NOTHING IS CHECKED"

def race(contours,img):
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        roi_image=img[y:y+h,x:x+w]

        if is_ticked(roi_image):
            field=recognize_text_trocr_2(img[y:y+h,x+w:x+w+80])
            if "others" in field.strip().lower():
                return recognize_text_trocr_2(img[y:y+h,x+w+210:x+w+400])
            else:
                return field
        
    return "NOTHING IS CHECKED"

def other_identification_no(contours,img):
    # x,y,w,h=cv2.boundingRect(contours[4])
    # roi_image=img[y:y+h,x:x+w]
    # if is_ticked(roi_image):
    for c in range(0,4):
        x,y,w,h=cv2.boundingRect(contours[c])
        roi=img[y:y+h,x:x+w]
        if is_ticked(roi):
            feild=recognize_text_trocr_2(img[y:y+h,x+w:x+w+150])
            # i,j,k,l=cv2.boundingRect(contours[23])
            # iden_no=recognize_text_trocr(img[y+h+10:y+h+40,x+w+130:x+w+450])
            return feild
    return "NOTHING IS CHECKED"

def mobile_no_type(contours,img):
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        roi_img=img[y:y+h,x:x+w]
        if is_ticked(roi_img):
            return recognize_text_trocr_2(img[y:y+h,x+w:x+w+250])
    return "NOTHING IS CHECKED"

def credit_limit(contours,img):
    x,y,w,h=cv2.boundingRect(contours)
    roi_img=img[y:y+h,x:x+w]
    # if is_ticked(roi_img):
    return recognize_text_trocr_2(img[y+h+40:y+h+70,x:x+w+570])
    # return "NOTHING IS CHECKED"


# # Find contours of the CHECK BOXES
contours_c, _ = cv2.findContours(img_bin_combined_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours,_=cv2.findContours(img_bin_combined,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# # ENTIRE CODE FOR CHECK BOX



checkbox_contours = []
# field_contours=[]
for i,contour in enumerate(contours_c):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    if 0.8 <= aspect_ratio <= 1.5:  # Aspect ratio close to 1 for squares
        # Filter based on size to exclude large or very small boxes
        if 5<= w <=20  and 5<= h <= 20:  # Adjust these size thresholds as needed
            area = cv2.contourArea(contour)
            if area >250 and area <= 450:
                checkbox_contours.append(contour)
# print(checkbox_contours)
# checkbox_contours=remove_nearby_contours(checkbox_contours)


field_contours=[]
for i,c in enumerate(contours):
    area=cv2.contourArea(c)
    if area>14000 and area<20000:
        field_contours.append(c)
        # x,y,w,h=cv2.boundingRect(c)
        # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
    # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    # print(i,c)

field_contours=remove_nearby_contours(field_contours)
result={}

for i,c in enumerate(field_contours):
    if i==len(field_contours)-1:
        text,field=field_detection_2(c,img,image_np)
    else:

        text,field=field_detection(c,img,image_np)
    if field in result.keys():
        field=field+"2"
    result[field]=text


dates=[]
for i,c in enumerate(contours):
    area=cv2.contourArea(c)
    if area>5000 and area<11500:

        dates.append(c)

extra=[]
for i,c in enumerate(contours):
    area=cv2.contourArea(c)
    if area>11500 and area<14000:
        extra.append(c)
# # # # for i,c in enumerate(dates):
# # # #         x,y,w,h=cv2.boundingRect(c)
# # # #         cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
# # # #         cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
# # # image_np=cv2.resize(image_np,(550,700))
# # # cv2.imshow("Image",image_np)
# # # # cv2.imwrite("C",image_np)
# # # img_bin=cv2.resize(img_bin,(550,700))
# # # cv2.imshow("Binary Image",img_bin)
# # # cv2.waitKey(0)
# # # cv2.destroyAllWindows()

dates=remove_nearby_contours(dates)
for c in dates:
    text,field=field_detection_2(c,img,image_np)
    if field in result.keys():
        field=field+"2"
    result[field]=text
extra=remove_nearby_contours(extra)



for c in extra:
    text,field=field_detection(c,img,image_np)
    if field in result.keys():
        field=field+"2"
    result[field]=text


# # contour=checkbox_contours[4]
# # x,y,w,h=cv2.boundingRect(contour)
# # cv2.imshow("Image",img[y:y+h,x+w+400:x+w+1200])
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# # # output_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# # contour=checkbox_contours[0]
# # # for i,c in enumerate(contour):
# # x,y,w,h=cv2.boundingRect(contour)
# # # print(recognize_text_trocr(image_np[y:y+h, x+w:x+w+250]))
# # cv2.imshow("Feild",image_np[y+h+40:y+h+70, x:x+w+570])
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# # # # output_image=cv2.resize(output_image,(550,850))
# # # # cv2.imwrite("detected_fields_page_1.png",output_image)
# # # cv2.imshow("Detected feilds",output_image)
# # print("Salutaion:",salutation(checkbox_contours[16:21],img))
# # print("Race:",race(checkbox_contours[10:14],img))
# # print("Other identification number:",other_identification_no(checkbox_contours[1:6],img))
# # print("Mobile no type: ",mobile_no_type(checkbox_contours[14:16],img))
# # print("Credit Limit: ",credit_limit(checkbox_contours[0],img))

result["Salutation"]=salutation(checkbox_contours[16:21],img)
result["Race"]=race(checkbox_contours[10:14],img)
result["other identification number"]=other_identification_no(checkbox_contours[1:6],img)
result["mobile no type"]=mobile_no_type(checkbox_contours[14:16],img)
result["Credit Limit"]=credit_limit(checkbox_contours[0],img)

# print(result)




df = pd.DataFrame(list(result.items()), columns=['Field', 'Text Extracted'])

print(df)
# df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\pradyumn\\pradyumn_11.xlsx',index = False)

end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken for the process is {total_time}")