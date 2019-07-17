import pandas as pd
import numpy as np

train_bbox=pd.read_csv('challenge-2019-train-detection-bbox.csv')
class_desc=pd.read_csv('challenge-2019-classes-description-500.csv', header=None, names=['Class ID','Class Name'])
class_desc['ClassCode'] = range(class_desc.shape[0])

f=open('cont.txt','r')
cont=f.readlines()
img_list=[]
for row in cont:
    img_list.append(row.split()[-1][:-4] )

train_annot=(train_bbox.set_index('ImageID')).loc[img_list, : ]


train_annot.dropna(inplace=True)
train_annot=train_annot.reset_index()
train_annot=pd.merge(train_annot,class_desc, right_on='Class ID', left_on='LabelName').drop(['Class ID','LabelName'],axis=1)
# Row format: image_file_path box1 box2 ... boxN;
# Box format: x_min,y_min,x_max,y_max,class_id (no space)


train_OP=pd.DataFrame(columns=['ImageID','Annot_Str'])
annot_str=""
op={}

for img in train_annot['ImageID'].unique():
    pred_str=''
    op['ImageID']= img
    obj = (train_annot[ train_annot['ImageID'] == img]).reindex()
    for i in range(obj.shape[0]):
        pred_str= pred_str+' '+str(obj.iloc[i,3])+','+str(obj.iloc[i,5])+','+str(obj.iloc[i,4])+','+str(obj.iloc[i,6])+','+str(obj.iloc[i,13])
    op['Annot_Str']=str.strip(pred_str)
    pred_str=pred_str+"\n"
    annot_str= annot_str+op['ImageID']+'.jpg'+pred_str
    train_OP=train_OP.append(op, ignore_index=True)

train_OP['ImageID']='../train_00/train_00/'+train_OP['ImageID']+'.jpg'

train_OP.to_csv('train_OP.txt', sep=' ', header=False,  index=False)