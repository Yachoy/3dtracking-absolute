import cv2,sys,os,numpy as np,pandas as pd
d,out=sys.argv[1],sys.argv[2]
def load(n):
    return pd.read_csv(os.path.join(d,n),sep=';',encoding='cp1251').iloc[:,:7]
T=load('object_transform_SM_AA_track.csv').iloc[:,:4]
foc={}
for i in range(1,6):
    foc[i]=float(open(os.path.join(d,f'object_transform_CineCameraActor{i}.csv'),encoding='cp1251').readline().strip().split(';')[-1])
def axes(r,p,y):
    r,p,y=np.radians([r,p,y]);SR,CR,SP,CP,SY,CY=np.sin(r),np.cos(r),np.sin(p),np.cos(p),np.sin(y),np.cos(y)
    F=np.array([CP*CY,CP*SY,SP]);R=np.array([SR*SP*CY-CR*SY,SR*SP*SY+CR*CY,-SR*CP]);U=np.array([-(CR*SP*CY+SR*SY),CY*SR-CR*SP*SY,CR*CP])
    return F,R,U
for i in range(1,6):
    C=load(f'object_transform_CineCameraActor{i}.csv')
    cap=cv2.VideoCapture(os.path.join(d,f'Way1.FinalImage.CineCameraActor{i}.mp4'))
    fx=foc[i]/23.76*1920; fy=foc[i]/13.365*1080
    for f in (100,500,900):
        c=C.iloc[f];F,R,U=axes(*map(float,(c.Rot_Roll,c.Rot_Pitch,c.Rot_Yaw)))
        dv=T.iloc[f][['Loc_X','Loc_Y','Loc_Z']].values.astype(float)-c[['Loc_X','Loc_Y','Loc_Z']].values.astype(float)
        z=dv@F;u=960+fx*(dv@R)/z;v=540-fy*(dv@U)/z
        print(f'cam{i} f{f} dist={z/100:.0f}m u={u:.0f} v={v:.0f}')
        cap.set(1,f);ok,img=cap.read()
        if z>0 and -2000<u<4000: cv2.circle(img,(int(u),int(v)),25,(0,0,255),4)
        cv2.imwrite(os.path.join(out,f'p{i}_{f}.jpg'),cv2.resize(img,(960,540)))
