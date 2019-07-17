from yolo_edit_final import YOLO
from PIL import Image
import pandas as pd


def detect_img(yolo, img):
    while True:
        #img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_box = yolo.detect_image(image)
            #r_image.show()
            print(img[:-4]+" "+str(r_box))
            return r_box
    yolo.close_session()
     

    
    
    
if __name__ == '__main__':
    class_desc=pd.read_csv('../challenge-2019-classes-description-500.csv', header=None, names=['Class ID','Class Name'])
    class_desc['ClassCode'] = range(class_desc.shape[0])
    
    yolo=YOLO()
    
    #f = open('img_list', 'r')
    #img = f.readline()
    img = input('Input image filename:')
    r_box = detect_img(yolo, img)
    res=pd.DataFrame(columns=['ImageId','PredictionString'])
    temp=pd.DataFrame(r_box)
    temp=pd.merge(temp,class_desc, right_on='Class Name', left_on='Class').drop(['Class Name','ClassCode','Class'],axis=1)
    pred_str=''
    for row in temp.T:
        p_str=str(temp.loc[row,'Class ID'])+" "+str(temp.loc[row, 'Score'])+" "+str(temp.loc[row, 'Box'][0])+" "+str(temp.loc[row, 'Box'][1])+" "+str(temp.loc[row, 'Box'][2])+" "+str(temp.loc[row, 'Box'][3])+"\n"
        pred_str=pred_str+p_str
    res=res.append(pd.Series({'ImageId':img[:-4], 'PredictionString':pred_str}), ignore_index=True)
    res.to_csv('tester.csv', index=False)