import cv2
import numpy as np
import time
from pdf2image import convert_from_path
import pandas as pd
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from final1 import highest_education_level,with_dependent_children,marital_status,gender,residency_status,existing,products
from final1 import salutation as salutation1
from final1 import race as race1
from final1 import other_identification_no as other_identification_no1\

from final2 import residence_type,type_of_company,preferred_mailing_address,size_of_current_employment,employee_status

from final3 import tax_number

from final4 import yes_no,pep,code

from final5 import declaration as declaration5

from eleventh_page import field_detection,field_detection_2,mobile_no_type,credit_limit
from eleventh_page import salutation as salutation11
from eleventh_page import race as race11
from eleventh_page import other_identification_no as other_identification_no11

from tenth_page import islamic_credit_card,credit_card,remove_lines,card_type,remove_lines_2

from ninth_page import regional_customer,takaful_plan
from ninth_page import declaration as declaration9

from eight_page import declaration as declaration8

from seventh_page import declaration as declaration7
from seventh_page import loan_tenure

from sixth_page import declaration as declaration6
from sixth_page import acted_as_guarantor

processor = TrOCRProcessor.from_pretrained("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\AJK_CODE\\EXECUTABLE\\trocr_processor")
model = VisionEncoderDecoderModel.from_pretrained("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\AJK_CODE\\EXECUTABLE\\trocr_model")

images = convert_from_path("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\Different handwritings\\File 7.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:\\Program Files (x86)\\poppler-24.07.0\\Library\\bin") 


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



def recognize_text_trocr(roi_image):
    # roi_image=remove_lines(roi_image)
    if is_blank_space(remove_lines(roi_image)):
        return ""  # Skip OCR for blank regions
    pil_image = Image.fromarray(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def recognize_text_trocr_2(roi_image):
    # if is_blank_space(roi_image):
    #     return ""  # Skip OCR for blank regions
    pil_image = Image.fromarray(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def page_1(img,contours,contours_c,image_np):
    # contours_c=remove_nearby_contours(contours_c)
    text={}
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
    for i_cb,contour_cb in enumerate(contours_c):
        x, y, w, h = cv2.boundingRect(contour_cb)
        aspect_ratio = w / float(h)
        if 0.8 <= aspect_ratio <= 1.5:  # Aspect ratio close to 1 for squares
            # Filter based on size to exclude large or very small boxes
            if 5<= w <=20  and 5<= h <= 20:  # Adjust these size thresholds as needed
                area = cv2.contourArea(contour_cb)
                if area >250 and area <= 450 and y > 250:
                    cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
                    cv2.putText(image_np,str(i_cb),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
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
    result=products(cb_c_product,img)
    if not result:
        text["Field_Products Interested in"] = "NOTHING IS CHECKED"
    else:
        text["Field_Products Interested in"]= set(result)

    # print(text)

    text["Field_Highest_education_level"] = highest_education_level(cb_c_highest_education,img)
    text["Field_With Dependent Children"]= with_dependent_children(cb_c_children,img)
    # print(len(cb_c_marital))
    text["Field_Marital Status"] = marital_status(cb_c_marital,img)
    # print(text)
    # plt.imshow(image_np,cmap = 'gray')
    # plt.show()
    text["Field_Gender"] = gender(cb_c_gender,img)
    text["Field_Race"] = race1(cb_c_race,img)
    text["Field_Residency Status"] = residency_status(cb_c_residency,img,image_np)
    text["Field_Other Identification no"] = other_identification_no1(cb_c_other_identification,img)
    text["Field_Salutation"] = salutation1(cb_c_sal,img)
    # text[field_contours)

    text["Field_Are you an existing customer"] = existing(cb_c_ec,img)
    contours=remove_nearby_contours(contours)
    l_i = []
    # print(contours)
    # text = {}
    for i,c in enumerate(contours):
        x,y,w,h=cv2.boundingRect(c)
        area=cv2.contourArea(c)
        if area>10000 and area<20000 and y> 500:
            # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
            # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            roi_image = img[y:y+h, x:x+w]
            generated_text = recognize_text_trocr(roi_image)
            text[f'Field_{i}'] = generated_text
    image_np=cv2.resize(image_np,(550,850))
    cv2.imshow("Image",image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    df = pd.DataFrame(list(text.items()),columns = ['Field','Extracted Text' ])
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\archer\\archer_1.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\harsha\\harsha_1.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\pradyumn\\pradyumn_1.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\devanshi\\devanshi_1.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\selva\\selva_1.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\thomas\\thomas_1.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\aditya\\aditya_1.xlsx',index = False)
    return df

def page_11(img,contours,contours_c,image_np):
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


    

    result["Salutation"]=salutation11(checkbox_contours[16:21],img)
    result["Race"]=race11(checkbox_contours[10:14],img)
    result["other identification number"]=other_identification_no11(checkbox_contours[1:6],img)
    result["mobile no type"]=mobile_no_type(checkbox_contours[14:16],img)
    result["Credit Limit"]=credit_limit(checkbox_contours[0],img)

    df = pd.DataFrame(list(result.items()), columns=['Field', 'Text Extracted'])

# print(df)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\archer\\archer_11.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\harsha\\harsha_11.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\pradyumn\\pradyumn_11.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\devanshi\\devanshi_11.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\selva\\selva_11.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\thomas\\thomas_11.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\aditya\\aditya_11.xlsx',index = False)
    return df

def page_10(img,contours,contours_c,image_np):
    checkbox_contours = []
    field_contours=[]
    for i,contour in enumerate(contours_c):
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 0.8 <= aspect_ratio <= 1.5:  # Aspect ratio close to 1 for squares
            # Filter based on size to exclude large or very small boxes
            if 5<= w <=20  and 5<= h <= 20:  # Adjust these size thresholds as needed
                area = cv2.contourArea(contour)
                if area >250 and area <= 450:
                    checkbox_contours.append(contour)
    # # print(checkbox_contours)
    checkbox_contours=remove_nearby_contours(checkbox_contours)

    field_contours=[]
    for i,c in enumerate(contours): 
        area=cv2.contourArea(c)
        if area>14000 and area<20000:

            field_contours.append(c)

    extra=[]

    for c in contours:
        area=cv2.contourArea(c)
        if area>5500 and area<7000:
        
            extra.append(c)

    field_contours=remove_nearby_contours(field_contours)

    result={}
    for i,c in enumerate(field_contours):
        if i==0:
            text,field=field_detection_2(c,img,image_np)
        else:

            text,field=field_detection(c,img,image_np)
        if field in result.keys():
            field=field+"2"
        result[field]=text

    for i,c in enumerate(extra):
        x,y,w,h=cv2.boundingRect(c)
        if i==2:
            r=remove_lines_2(image_np[y:y+h,x+w+100:x+w+230])
            # r=remove_lines(r)
            # cv2.imshow("image",r)
            result["CONVENTIONAL BRANCH CODE"]=recognize_text_trocr(r)

        elif i==1:
            r=remove_lines_2(image_np[y:y+h,x+w:x+w+240])
            # r=remove_lines(r)
            # cv2.imshow("Image",r)
            result["EMPLOYEE NO."]=recognize_text_trocr(r)
        else:
            r=remove_lines_2(image_np[y:y+h,x+w+130:x+w+240])
            # r=remove_lines(r)
            # cv2.imshow("Image",r)
            result["CAMPAIGN CODE"]=recognize_text_trocr(r)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    result["Islamic credit card:"]=islamic_credit_card(checkbox_contours[0:3],img)
    result["credit card:"]=credit_card(checkbox_contours[3:12],img)
    result["card type:"]=card_type(checkbox_contours[12:15],img)
    df=pd.DataFrame(list(result.items()),columns=["Field","Text Extracted"])
# print(df)
        # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\archer\\archer_10.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\harsha\\harsha_10.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\pradyumn\\pradyumn_10.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\devanshi\\devanshi_10.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\selva\\selva_10.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\thomas\\thomas_10.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\aditya\\aditya_10.xlsx',index = False)
    return df

def page_9(img,contours,contours_c,image_np):
    checkbox_contours = []
    field_contours=[]
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
    checkbox_contours=remove_nearby_contours(checkbox_contours)

    field_contours=[]
    for i,c in enumerate(contours):
        area=cv2.contourArea(c)
        if area>18000 and area<20000:
            field_contours.append(c)

    extra=[]
    for i,c in enumerate(contours):
        area=cv2.contourArea(c)
        if area>10000 and area<18000:
            extra.append(c)


    takaful=[]
    for c in contours:
        area=cv2.contourArea(c)
        if area>25000 and area<30000:                                                                                                                                                                                  
            takaful.append(c)

    result={}
    field_contours=remove_nearby_contours(field_contours)
    for i,c in enumerate(field_contours):
       
        x,y,w,h=cv2.boundingRect(c)
        text,field=field_detection(c,img,image_np)
        if field in result.keys():
            field=field+"2"
        result[field]=text

    extra=remove_nearby_contours(extra)
    for i,c in enumerate(extra):
        if i==0 or i==1:
            continue
        x,y,w,h=cv2.boundingRect(c)
        text,field=field_detection_2(c,img,image_np)
        if field in result.keys():
            field=field+"2"
        result[field]=text

        
    result["Regional customer"]=regional_customer(checkbox_contours[0:2],img)
    result["Differently abled"]=regional_customer(checkbox_contours[2:4],img)
    result["Takapul Plan"]=takaful_plan(contours[4],img,takaful)
    r=declaration9(checkbox_contours[5:8],img)
    if not r:
        result["Declaration"]="NONE is CHECKED"
    else:
        result["Declaration"]=r

    df=pd.DataFrame(list(result.items()),columns=["Field","Text Extracted"])
    # print(df)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\archer\\archer_9.xlsx',index = False)
    ## df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\harsha\\harsha_9.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\pradyumn\\pradyumn_9.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\devanshi\\devanshi_9.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\selva\\selva_9.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\thomas\\thomas_9.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\aditya\\aditya_9.xlsx',index = False)
    return df

def page_8(img,contours,contours_c,image_np):
    checkbox_contours = []
    field_contours=[]
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
    checkbox_contours=remove_nearby_contours(checkbox_contours)

    field_contours=[]
    for i,c in enumerate(contours):
        area=cv2.contourArea(c)
        if area>10000 and area<20000:
            field_contours.append(c)

    result={}
    # x,y,w,h=cv2.boundingRect(field_contours[0])
    text,field=field_detection_2(field_contours[0],img,image_np)
    result[field]=text

    result["declaration"]=declaration8(checkbox_contours[0],img)

    df=pd.DataFrame(list(result.items()),columns=["Field","Text Extracted"])
    # print(df)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\archer\\archer_8.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\harsha\\harsha_8.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\pradyumn\\pradyumn_8.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\devanshi\\devanshi_8.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\selva\\selva_8.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\thomas\\thomas_8.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\aditya\\aditya_8.xlsx',index = False)
    return df

def page_7(img,contours,contours_c,image_np):
    checkbox_contours = []
    field_contours=[]
    for i,contour in enumerate(contours_c):
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 0.8 <= aspect_ratio <= 1.5:  # Aspect ratio close to 1 for squares
            # Filter based on size to exclude large or very small boxes
            if 5<= w <=20  and 5<= h <= 20:  # Adjust these size thresholds as needed
                area = cv2.contourArea(contour)
                if area >240 and area <= 450:
                    checkbox_contours.append(contour)

    field_contours=[]
    for i,c in enumerate(contours):
        area=cv2.contourArea(c)
        if area>10000 and area<20000:

            # x,y,w,h=cv2.boundingRect(c)
            field_contours.append(c)
            # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
        # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        # print(i,c)


    extra=[]
    for i,c in enumerate(contours):
        area=cv2.contourArea(c)
        if area>5500 and area<7000:
            extra.append(c)
    field_contours=remove_nearby_contours(field_contours)
    extra=remove_nearby_contours(extra)
    checkbox_contours=remove_nearby_contours(checkbox_contours)

    # total_contours=field_contours+extra+checkbox_contours
    # for c in total_contours:
    #     x,y,w,h=cv2.boundingRect(c)
    #     cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),2)
    # cv2.imwrite("thomas_7.png",image_np)
    result={}
    for i,c in enumerate(field_contours):
        if i==0 or i==2:
            text,field=field_detection(c,img,image_np)
        else:
            text,field=field_detection_2(c,img,image_np)
        if field in result.keys():
                field=field+"2"
        result[field]=text

    for i,c in enumerate(extra):
        x,y,w,h=cv2.boundingRect(c)
        if i==1:
            r=remove_lines_2(image_np[y:y+h,x+w:x+w+240])
            # cv2.imshow("image",r)
            result["CAMPAGIN CODE"]=recognize_text_trocr(r)

        elif i==2:
            r=remove_lines_2(image_np[y:y+h,x+w+100:x+w+230])
            # cv2.imshow("Image",r)
            result["EMPLOYEE NO."]=recognize_text_trocr(r)
        else:
            r=remove_lines_2(image_np[y:y+h,x+w+130:x+w+240])
            # cv2.imshow("Image",r)
            result["CONVENTIONAL BRANCH CODE"]=recognize_text_trocr(r)
    
    result["Loan Tenure:"]=loan_tenure(checkbox_contours[0:8],img)
    # result["Applied for:"]=declaration(checkbox_contours[8:10],img)

    r=declaration7(checkbox_contours[8:10],img)
    if not r:
        result["Applied for"]="NONE"
    else:
        result["Applied for"]=r

    df=pd.DataFrame(list(result.items()),columns=["Fields","Text extracted"])
# print(df)

    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\archer\\archer_7.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\harsha\\harsha_7.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\pradyumn\\pradyumn_7.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\devanshi\\devanshi_7.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\selva\\selva_7.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\thomas\\thomas_7.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\aditya\\aditya_7.xlsx',index = False)
    return df

def page_6(img,contours,contours_c,image_np):
    checkbox_contours = []
    field_contours=[]
    extra=[]
    for i,contour in enumerate(contours_c):
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 0.8 <= aspect_ratio <= 1.5:  # Aspect ratio close to 1 for squares
            # Filter based on size to exclude large or very small boxes
            if 5<= w <=20  and 5<= h <= 20:  # Adjust these size thresholds as needed
                area = cv2.contourArea(contour)
                if area >250 and area <= 450:
                    checkbox_contours.append(contour)



    for i,c in enumerate(contours):
        area=cv2.contourArea(c)
        if area>5000 and area<9000:
            extra.append(c)
        # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        # print(i,c)
    for c in contours:
        area=cv2.contourArea(c)
        if area>10000 and area<16000:
            field_contours.append(c)

    field_contours=field_contours[::-1]

    # print(field_contours)
    result={}
    list1=field_contours[3:6]
    for i,c in enumerate(list1):
        x,y,w,h=cv2.boundingRect(c)
        if i==0:
            result["Main_applicant_name_1"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
        elif i==1:
            result["Main_applicant_debt_1"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
        else:
            result["Main_applicant_monthly_installment_1"]=recognize_text_trocr(image_np[y:y+h,x:x+w])

    list2=field_contours[6:9]

    for i,c in enumerate(list2):
        x,y,w,h=cv2.boundingRect(c)
        if i==0:
            result["Main_applicant_name_2"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
        elif i==1:
            result["Main_applicant_debt_2"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
        else:
            result["Main_applicant_monthly_installment_2"]=recognize_text_trocr(image_np[y:y+h,x:x+w])

    list3=field_contours[12:15]
    for i,c in enumerate(list3):
        x,y,w,h=cv2.boundingRect(c)
        if i==0:
            result["Joint_applicant_name_1"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
        elif i==1:
            result["Joint_applicant_debt_1"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
        else:
            result["Joint_applicant_monthly_installment_1"]=recognize_text_trocr(image_np[y:y+h,x:x+w])

    list4=field_contours[15:18]

    for i,c in enumerate(list4):
        x,y,w,h=cv2.boundingRect(c)
        if i==0:
            result["Joint_applicant_name_2"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
        elif i==1:
            result["Joint_applicant_debt_2"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
        else:
            result["Joint_applicant_monthly_installment_2"]=recognize_text_trocr(image_np[y:y+h,x:x+w])

    list5=[]
    # list5.append(field_contours[18])
    list5.append(field_contours[20])
    list5.append(field_contours[22])

    # list5=list5+extra[0:4]
    for i,c in enumerate(list5):
        x,y,w,h=cv2.boundingRect(c)
        if i==0:
            result["CIMB_group_relative_name_1"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
        else:
            result["CIMB_group_relative_name_2"]=recognize_text_trocr(image_np[y:y+h,x:x+w])

    list6=[]
    list6.append(extra[3])
    list6.append(extra[1])

    for i,c in enumerate(list6):
        x,y,w,h=cv2.boundingRect(c)
        if i==0:
            result["CIMB_group_relative_passport_no_1"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
        else:
            result["CIMB_group_relative_passport_no_2"]=recognize_text_trocr(image_np[y:y+h,x:x+w])

    list7=[]
    list7.append(extra[2])
    list7.append(extra[0])


    for i,c in enumerate(list7):
        x,y,w,h=cv2.boundingRect(c)
        if i==0:
            result["CIMB_group_relative_relationship_1"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
        else:
            result["CIMB_group_relative_relationship_2"]=recognize_text_trocr(image_np[y:y+h,x:x+w])

    result["acted_as_guarantor_1"]=acted_as_guarantor(checkbox_contours[2:4],image_np)
    result["acted_as_guarantor_2"]=acted_as_guarantor(checkbox_contours[0:2],image_np)

    r=declaration6(checkbox_contours[4:6],img)
    if not result:
        result["declaration"]="NOTHING IS CHECKED"
    else:
        result["declaration"]=r

    df=pd.DataFrame(list(result.items()),columns=["Field","Text Extracted"])
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\archer\\archer_6.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\harsha\\harsha_6.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\pradyumn\\pradyumn_6.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\devanshi\\devanshi_6.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\selva\\selva_6.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\thomas\\thomas_6.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\aditya\\aditya_6.xlsx',index = False)
    return df

def page_5(img,contours,contours_c,image_np):
    checkbox_contours = []
    field_contours=[]
    contours_c = remove_nearby_contours(contours_c)
    for i,contour in enumerate(contours_c):
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 0.8 <= aspect_ratio <= 1.5:  # Aspect ratio close to 1 for squares
            # Filter based on size to exclude large or very small boxes
            if 5<= w <=20  and 5<= h <= 20:  # Adjust these size thresholds as needed
                area = cv2.contourArea(contour)
                if area >250 and area <= 450 and y > 270:
                    cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
                    checkbox_contours.append(contour)
    text = {}
    f_c = []

    contours_keep = remove_nearby_contours(contours)
    for i,c in enumerate(contours):
        area=cv2.contourArea(c)
        if area>15000 and area<20000:
            roi_image = img[y:y+h, x:x+w]
            # generated_text = recognize_text_trocr(roi_image)
            # text[f'Field_{i}'] = generated_text
            x,y,w,h=cv2.boundingRect(c)
            f_c.append(c)
            cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)

    for c in f_c:
        res,field=field_detection(c,img,image_np)
        if field in text.keys():
            field=field+"2"
        text[field]=res

    dec = declaration5(checkbox_contours[:4],img)
    # if not dec:
        # text['declaration'] = 'No'
    # else:
    text['declaration'] = dec

    # print(text)

    # order_text = dict(sorted(text.items(),reverse = True))

    df = pd.DataFrame(list(text.items()),columns = ['Field','Extracted Text' ])

    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\archer\\archer_5.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\harsha\\harsha_5.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\pradyumn\\pradyumn_5.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\devanshi\\devanshi_5.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\selva\\selva_5.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\thomas\\thomas_5.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\aditya\\aditya_5.xlsx',index = False)
    return df

def page_2(img,contours,contours_c,image_np):
    contour_c = remove_nearby_contours(contours_c)

    cb_c_size = []
    cb_c_type = []
    cb_c_employement = []
    cb_c_mailing = []
    cb_c_residence = []
    field_contours=[]
    for i,c in enumerate(contours_c):
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        if 0.8 <= aspect_ratio <= 1.5:  # Aspect ratio close to 1 for squares
            # Filter based on size to exclude large or very small boxes
            if 5<= w <=20  and 5<= h <= 20:  # Adjust these size thresholds as needed
                area = cv2.contourArea(c)
                if area >250 and area <= 450 and y > 329 and y < 495:
                    cb_c_residence.append(c)
                if area >250 and area <= 450 and y > 157 and y < 226 and x > 700:
                    cb_c_mailing.append(c)
                if area >250 and area <= 450 and y > 780 and y < 870 and x>700:
                    cb_c_employement.append(c)
                if area >250 and area <= 450 and y > 935 and y < 1215 and x>700:
                    cb_c_type.append(c)
                if area >250 and area <= 450 and y > 1215 and x>700:
                    cb_c_size.append(c)

    field_contours = []
    field_contours_sl = []
    contours_keep = remove_nearby_contours(contours)
    for i,c in enumerate(contours_keep):
        area=cv2.contourArea(c)
        x,y,w,h=cv2.boundingRect(c)
        if area>10000 and area<20000:
            x,y,w,h=cv2.boundingRect(c)
            field_contours.append(c)
            cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
        elif area > 2500 and area < 5000 and y > 1000 and x > 460 and x < 730:
            field_contours_sl.append(c)
            cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
        
        elif area > 2500 and area < 5000 and y > 1000 and x >= 660:
            field_contours.append(c)
            cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)

        


        # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        # print(i,c)
    result = {}

    for c in field_contours:
        text,field=field_detection(c,img,image_np)
        if field in result.keys():
            field=field+"2"
        result[field]=text

    for c in field_contours_sl:
        text,field=field_detection_2(c,img,image_np)
        if field in result.keys():
            field=field+"2"
        result[field]=text

    result["Size of current employment"] = size_of_current_employment(cb_c_size, img)
    result["Type of company"] = type_of_company(cb_c_type ,img)
    result["Employee Status"] = employee_status(cb_c_employement,img)
    result["preferred_mailing_address"] = preferred_mailing_address(cb_c_mailing,img)
    result["Residence Type"] = residence_type(cb_c_residence,img)

    # print(result)
    order_text = dict(sorted(result.items(),reverse = True))

    df = pd.DataFrame(list(order_text.items()),columns = ['Field','Extracted Text' ])

    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\archer\\archer_2.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\harsha\\harsha_2.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\pradyumn\\pradyumn_2.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\devanshi\\devanshi_2.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\selva\\selva_2.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\thomas\\thomas_2.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\aditya\\aditya_2.xlsx',index = False)
    return df

def page_3(img,contours,contours_c,image_np):
    us_resident_pa = []
    us_resident_ja1 = []
    us_resident_ja2 = []
    us_permanent_resident_pa = []
    us_permanent_resident_ja1 = []
    us_permanent_resident_ja2 = []
    us_citizen_pa = []
    us_citizen_ja1 = []
    us_citizen_ja2 = []
    field_contours=[]

    contours_c = remove_nearby_contours(contours_c)
    for i,contour in enumerate(contours_c):
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 0.8 <= aspect_ratio <= 1.5:  # Aspect ratio close to 1 for squares
            # Filter based on size to exclude large or very small boxes
            if 5<= w <=20  and 5<= h <= 20:  # Adjust these size thresholds as needed
                area = cv2.contourArea(contour)
                if area >250 and area <= 450 and x>400 and x <666 and y>=1089 and y< 1190:
                    # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
                    # cv2.rectangle(image_np,(x,y+50),(x+260,y+80),(0,255,0),3)
                    # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
                    us_resident_pa.append(contour)
                elif area >250 and area <= 450 and x > 666 and x<978  and y>= 1089 and y < 1190:
                    # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
                    # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
                    us_resident_ja1.append(contour)
                elif area >250 and area <= 450 and x>1000 and y>=1089 and y< 1190:
                    # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
                    # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
                    us_resident_ja2.append(contour)
                elif area >250 and area <= 450 and x>400 and x <666 and y>=1200 and y< 1300:
                    # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
                    # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
                    us_citizen_pa.append(contour)
                elif area >250 and area <= 450 and x>666 and x <978 and y>=1200 and y< 1300:
                    # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
                    # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
                    us_citizen_ja1.append(contour)
                elif area >250 and area <= 450 and x>1000 and y>=1200 and y< 1300:
                    # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
                    # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
                    us_citizen_ja2.append(contour)
                elif area >250 and area <= 450 and x>400 and x <666 and y>=1300 and y< 1405:
                    # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
                    # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
                    us_permanent_resident_pa.append(contour)
                elif area >250 and area <= 450 and x>666 and x <978 and y>=1300 and y< 1405:
                    # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
                    # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
                    us_permanent_resident_ja1.append(contour)
                elif area >250 and area <= 450 and x>1000 and x< 1300 and y>=1300 and y< 1405:
                    # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
                    # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
                    us_permanent_resident_ja2.append(contour)
                
    text = {}
    f_c = []

    # print(len(us_permanent_resident_ja1),len(us_permanent_resident_ja2),len(us_permanent_resident_pa))
    # print(len(us_resident_pa))
    text['tax number if principal applicant is US Resident '] = tax_number(us_resident_pa,img)
    text['tax number if joint applicant 1 is US Resident '] = tax_number(us_resident_ja1,img)
    text['tax number if joint applicant 2 is US Resident '] = tax_number(us_resident_ja2,img)
    text['tax number if principal applicant is US Citizen '] = tax_number(us_citizen_pa,img)
    text['tax number if joint applicant 1 is US Citizen '] = tax_number(us_citizen_ja1,img)
    text['tax number if joint applicant 2 is US Citizen '] = tax_number(us_citizen_ja2,img)
    text['tax number if principal applicant is US Permanent Resident '] = tax_number(us_permanent_resident_pa,img)
    text['tax number if joint applicant 1 is US Resident '] = tax_number(us_permanent_resident_ja1,img)
    text['tax number if joint applicant 2 is US Resident '] = tax_number(us_permanent_resident_ja2,img)

    contours_keep = remove_nearby_contours(contours)
    for i,c in enumerate(contours_keep):
        x,y,w,h=cv2.boundingRect(c)
        area=cv2.contourArea(c)
        if area>15000 and area<30000 and x> 300 and y> 1390:
            roi_image = img[y:y+h, x:x+w]
            generated_text = recognize_text_trocr(roi_image)
            text[f'Field_{i}'] = generated_text
            
            f_c.append(c)
            # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
            # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
    df = pd.DataFrame(list(text.items()),columns = ['Field','Extracted Text' ])
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\archer\\archer_3.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\harsha\\harsha_3.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\pradyumn\\pradyumn_3.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\devanshi\\devanshi_3.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\selva\\selva_3.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\thomas\\thomas_3.xlsx',index = False)
    # df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\aditya\\aditya_3.xlsx',index = False)
    return df

def page_4(img,contours,contours_c,image_np):
    pa = []
    ja = []
    ja_2 = []
    pep_pa = []
    pep_ja1 = []
    pep_ja2 = []
    pep_fm_pa = []
    pep_fm_ja1 = []
    pep_fm_ja2 = []
    pep_ca_pa = []
    pep_ca_ja1 = []
    pep_ca_ja2 = []
    fm_pa_code = []
    fm_ja1_code = []
    fm_ja2_code = []
    ca_pa_code = []
    ca_ja1_code = []
    ca_ja2_code = []
    others_pa = []
    others_ja1 = []
    others_ja2 = []

    text = {}
    contours, _ = cv2.findContours(img_bin_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = remove_nearby_contours(contours)
    for i,contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 0.8 <= aspect_ratio <= 1.5:  # Aspect ratio close to 1 for squares
            # Filter based on size to exclude large or very small boxes
            if 5<= w <=20  and 5<= h <= 20:  # Adjust these size thresholds as needed
                area = cv2.contourArea(contour)
                if area >230 and area <= 450 and x< 485 and y >250 and y< 370:
                    pa.append(contour)
                    # checkbox_contours.append(contour)
                if area >230 and area <= 450 and x> 485 and x<897 and y>250 and y<370:
                    ja.append(contour)
                if area>230 and area<=450 and x>900 and y>250 and y<370:
                    ja_2.append(contour)

                
                if area > 230 and area<=450 and x >429 and x<688 and y >496 and y<562:
                    pep_pa.append(contour)
                
                if area > 230 and area<=450 and x >703 and x<999 and y >496 and y<562:
                    pep_ja1.append(contour)

                if area > 230 and area<=450 and x > 1025 and x< 1315 and y >496 and y<562:
                    pep_ja2.append(contour)
                
                if area > 230 and area<=450 and x >429 and x<688 and y >565 and y<626:
                    pep_fm_pa.append(contour)

                if area > 230 and area<=450 and x >703 and x<999 and y >565 and y<626:
                    pep_fm_ja1.append(contour)

                if area > 230 and area<=450 and x >1025 and x<1315 and y >565 and y<626:
                    pep_fm_ja2.append(contour)
                
                if area > 230 and area<=450 and x >429 and x<688 and y >725 and y<880:
                    pep_ca_pa.append(contour)

                if area > 230 and area<=450 and x >703 and x<999 and y >725 and y<880:
                    pep_ca_ja1.append(contour)

                if area > 230 and area<=450 and x >1025 and x<1315 and y >725 and y<880:
                    pep_ca_ja2.append(contour)

    for i,c in enumerate(contours):
        # print(i)
        x,y,w,h=cv2.boundingRect(c)
        area=cv2.contourArea(c)
        if area>5000 and area<10000 and x > 406 and x< 670 and y >590 and y<690:
            fm_pa_code.append(c)
            # print(recognize_text_trocr(c))

        if area>5000 and area<10000 and x > 680 and x< 980 and y >590 and y<690:
            fm_ja1_code.append(c)

        if area>5000 and area<10000 and x > 1000 and x< 1380 and y >590 and y<690:
            fm_ja2_code.append(c)
        
        if area>5000 and area<10000 and x > 406 and x< 670 and y >750 and y<870:
            ca_pa_code.append(c)

        if area>5000 and area<10000 and x > 680 and x< 980 and y >750 and y<870:
            ca_ja1_code.append(c)

        if area>5000 and area<10000 and x > 1000 and x< 1380 and y >750 and y<870:
            ca_ja2_code.append(c)
        
        if area>5000 and area<10000 and x > 406 and x< 670 and y >880 and y<970:
            others_pa.append(c)

        if area>5000 and area<10000 and x > 680 and x< 980 and y >880 and y< 970:
            others_ja1.append(c)

        if area>5000 and area<10000 and x > 1000 and x< 1380 and y >880 and y<970:
            others_ja2.append(c)

    text['Principal Applicant'] = yes_no(pa,img)
    text['Joint Applicant/ Supplementary Card/ Gurantor/ Registered Owner'] = yes_no(ja,img)
    text['Joint Applicant/ Supplementary Card/ Gurantor/ Registered Owner 2'] = yes_no(ja_2,img)
    text['Principal Applicant is PEP'] = pep(pep_pa,img)
    text['Joint Applicant/ Supplementary Card/ Gurantor/ Registered Owner is PEP'] = pep(pep_ja1,img)
    text['Joint Applicant/ Supplementary Card/ Gurantor/ Registered Owner 2 is PEP'] = pep(pep_ja2,img)
    text['Principal Applicant is a family member of the PEP'] = pep(pep_fm_pa,img)
    text['Code of pa'] = code(fm_pa_code,img)
    text['Joint Applicant/ Supplementary Card/ Gurantor/ Registered Owner  is a family member of the PEP'] = pep(pep_fm_ja1,img)
    text['Code of ja1'] = code(fm_ja1_code,img)
    text['Joint Applicant/ Supplementary Card/ Gurantor/ Registered Owner 2 is a family member of the PEP'] = pep(pep_fm_ja2,img)
    text['Code of ja2'] = code(fm_ja2_code,img)
    text['Principal Applicant is a close associate of the PEP'] = pep(pep_ca_pa,img)
    text['Code of pa as close associate'] = code(ca_pa_code,img)
    text['Joint Applicant/ Supplementary Card/ Gurantor/ Registered Owner  is a close associate of the PEP'] = pep(pep_ca_ja1,img)
    text['Code of ja as close associate'] = code(ca_ja1_code,img)
    text['Joint Applicant/ Supplementary Card/ Gurantor/ Registered Owner 2 is a close associate of the PEP'] = pep(pep_ca_ja2,img)
    text['Code of ja2 as close assocaite'] = code(ca_ja2_code,img)
    text['If the relationship between Personal Applicant and PEP is others mention'] = code(others_pa,img)
    text['If the relationship between Joint Applicant and PEP is others mention'] = code(others_ja1,img)
    text['If the relationship between Joint Applicant 2 and PEP is others mention'] = code(others_ja2,img)

    df=pd.DataFrame(list(text.items()),columns = ['Field','Extracted Text' ])
    return df

start_time = time.time()
dfs=[]
# image _np=cv2.resize(image_
for i in range(0,11):
    image_np = np.array(images[i])
    image_np=cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR)
    img=cv2.cvtColor(image_np,cv2.COLOR_RGB2GRAY)
    cv2.imshow("gray",img)
    cv2.waitKey(0)
    _, img_bin = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Binary",img_bin)
    cv2.waitKey(0)
    img_bin=cv2.dilate(img_bin,np.ones((2,2),np.uint8),iterations=1)
    cv2.imshow("image dilated",img_bin)
    cv2.waitKey(0)
    line_min_width = 15  # Adjust based on your checkbox size
    kernal_h = np.ones((1, line_min_width), np.uint8)
    kernal_v = np.ones((line_min_width, 1), np.uint8)
    img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
    cv2.imshow("Image Bin Horizontal",img_bin_h)
    img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)
    cv2.imshow("Image Binary Vertical",img_bin_v)
    cv2.waitKey(0)
    img_bin_combined = cv2.add(img_bin_h, img_bin_v)
    cv2.waitKey(0)
    cv2.imshow("Image binary combined",img_bin_combined)

    cv2.waitKey(0)


    image_np = np.array(images[i])
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


#     contours, _ = cv2.findContours(img_bin_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours_c, _ = cv2.findContours(img_bin_combined_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     if i==0:
#         df1=page_1(img,contours,contours_c,image_np)
#         print("PAGE 1")
# #     if i==1:
# #         df2=page_2(img,contours,contours_c,image_np)
# #         print("PAGE 2")
# #     if i==2:
# #         df3=page_3(img,contours,contours_c,image_np)
# #         print("PAGE 3")
# #     if i==3:
# #         df4=page_4(img,contours,contours_c,image_np)
# #         print("PAGE 4")
# #     if i==4:
# #         df5=page_5(img,contours,contours_c,image_np)
# #         print("PAGE 5")
# #     if i==5:
# #         df6=page_6(img,contours,contours_c,image_np)
# #         print("PAGE 6")

# #     if i==6:
# #         df7=page_7(img,contours,contours_c,image_np)
# #         print("PAGE 7")
# #     if i==7:
# #         df8=page_8(img,contours,contours_c,image_np)
# #         print("PAGE 8")
# #     if i==8:
# #         df9=page_9(img,contours,contours_c,image_np)
# #         print("PAGE 9")

# #     if i==9:
# #         df10=page_10(img,contours,contours_c,image_np)
# #         print("PAGE 10")
# #     if i==10:
# #         df11=page_11(img,contours,contours_c,image_np)
# #         print("PAGE 11")

# # df_dict = {
# #     'Sheet1': df1,
# #     'Sheet2': df2,
# #     'Sheet3': df3,
# #     'Sheet4': df4,
# #     'Sheet5': df5,
# #     'Sheet6': df6,
# #     'Sheet7': df7,
# #     'Sheet8': df8,
# #     'Sheet9': df9,
# #     'Sheet10': df10,
# #     'Sheet11': df11
# # }

# # # Create an Excel writer object
# # with pd.ExcelWriter('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\aditya\\aditya_all.xlsx', engine='xlsxwriter') as writer:
# #     for sheet_name, df in df_dict.items():
# #         df.to_excel(writer, sheet_name=sheet_name, index=False)




  
   



# # end_time = time.time()
# # total_time = end_time - start_time
# # print(f"Total time taken for the process is {total_time}")

