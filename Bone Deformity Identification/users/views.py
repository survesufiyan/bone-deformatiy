from django.shortcuts import render
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel

# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})    

def Upload(request):
    from django.core.files.storage import FileSystemStorage
    import os
    from django.conf import settings
    if request.method == 'POST':
        myfile = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        # uploaded_file_url = fs.url(filename)
        img_path = os.path.join(settings.MEDIA_ROOT, filename)

        import cv2
        import numpy as np
        

        from .Algo.pre_process import _reshape_img, get_model


            
        model_name= "users/Algo/rfc_model"

        

        # img_file= 'users/images/resized/F79.JPG'
        # print('img_file:',img_file)
        # orig_img= 'users/images/Fractured Bone/F79.JPG'
        # print('orig_img:',orig_img)

        #for image read
        try:
            img_t=cv2.imread(img_path,cv2.IMREAD_COLOR)
            img=cv2.imread(img_path,cv2.IMREAD_COLOR)
            shape= img.shape
        except (AttributeError,FileNotFoundError):
            try:
                img_t=cv2.imread(img_path,cv2.IMREAD_COLOR)
                img=cv2.imread(img_path,cv2.IMREAD_COLOR)
                shape=img.shape
            except (AttributeError,FileNotFoundError):
                img_t=cv2.imread(img_path+".png",cv2.IMREAD_COLOR)
                img=cv2.imread(img_path+".png",cv2.IMREAD_COLOR)
                shape=img.shape

            #else: raise FileNotFoundError("No image file {img_file}.jpg or {img_file}.JPG".format(img_file=img_file))
        #else:
        #	raise FileNotFoundError("No image file {img_file}.jpg or {img_file}.JPG".format(img_file=img_file))


        #details of Imge
        print("\nShape: ",shape)
        print("\nSize: ",img.size)
        print("\nDType: ",img.dtype)

        #==============Manual edge ditect=====================
        def segment_img(_img,limit):
            for i in range(0,_img.shape[0]-1):
                for j in range(0,_img.shape[1]-1): 
                    if int(_img[i,j+1])-int(_img[i,j])>=limit:
                        _img[i,j]=0
                    elif(int(_img[i,j-1])-int(_img[i,j])>=limit):
                        _img[i,j]=0
            
            return _img
        #======================================================

        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #for i in range(0,gray.shape[0]):
        #	for j in range(0,gray.shape[1]): 
        #		if (int(gray[i,j]))<=100:
        #			gray[i,j]=100

        #gray=segment_img(gray,15)
        cv2.imshow("GrayEdited",gray)
        median = cv2.medianBlur(gray,5)

        model= get_model(model_name)
        pred_thresh= model.predict([_reshape_img(img_t)])
        # bool,threshold_img=cv2.threshold(median,pred_thresh,255,cv2.THRESH_BINARY)
        bool,threshold_img=cv2.threshold(median, 120, 255, cv2.THRESH_BINARY)

        #blur=cv2.GaussianBlur(threshold_img,(7,7),0)
        cv2.imshow("threshold",threshold_img)


        initial=[]
        final=[]
        line=[]
        #count=[]
        #for i in range(0,256):
        #	count.append(0)

        for i in range(0,gray.shape[0]):
            tmp_initial=[]
            tmp_final=[]
            for j in range(0,gray.shape[1]-1):
                #count[gray[i,j]]+=1
                if threshold_img[i,j]==0 and (threshold_img[i,j+1])==255:
                    tmp_initial.append((i,j))
                    #img[i,j]=[255,0,0]
                if threshold_img[i,j]==255 and (threshold_img[i,j+1])==0:
                    tmp_final.append((i,j))
                    #img[i,j]=[255,0,0]
            
            x= [each for each in zip(tmp_initial,tmp_final)]
            x.sort(key= lambda each: each[1][1]-each[0][1])
            try:
                line.append(x[len(x)-1])
            except IndexError: pass

        #print(count)


        err= 15
        danger_points=[]

        #store distances
        dist_list=[]

        for i in range(1,len(line)-1):
            dist_list.append(line[i][1][1]-line[i][0][1])
            try:
                prev_= line[i-3]
                next_= line[i+3]

                dist_prev= prev_[1][1]-prev_[0][1]
                dist_next= next_[1][1]-next_[0][1]
                diff= abs(dist_next-dist_prev)
                if diff>err:
                    #print("Dist: {}".format(abs(dist_next-dist_prev)))
                    #print(line[i])
                    data=(diff, line[i])
                    #print(data)
                    if len(danger_points):
                        prev_data=danger_points[len(danger_points)-1]
                        #print(prev_data)
                        #print("here1....")
                        if abs(prev_data[0]-data[0])>2 or data[1][0]-prev_data[1][0]!=1:
                            #print("here2....")
                            print(data)
                            danger_points.append(data)
                    else:
                        print(data)
                        danger_points.append(data)
            except Exception as e:
                print(e)
                pass

            #print(each)
            start,end= line[i]
            #raise ZeroDivisionError
            mid=int((start[0]+end[0])/2),int((start[1]+end[1])/2)
            #img[mid[0],mid[1]]=[0,0,255]

        for i in range(0,len(danger_points)-1,2):
            try:
                start_rect=danger_points[i][1][0][::-1]
                start_rect=(start_rect[0]-40, start_rect[1]-40)
        
                end_rect= danger_points[i+1][1][1][::-1]
                end_rect= (end_rect[0]+40, end_rect[1]+40)
        
                cv2.rectangle(img,start_rect,end_rect,(0,255,0),2)
            except:
                print("Pair not found")

        #blur= cv2.GaussianBlur(img,(5,5),0)

        import matplotlib.pyplot as plt
        import numpy as np

        fig, (ax1, ax2)= plt.subplots(2,1)

        fig2, ax3= plt.subplots(1,1)

        x= np.arange(1,gray.shape[0]-1)
        y= dist_list

        #print(len(x),len(y))

        cv2.calcHist(gray,[0],None,[256],[0,256])

        try:
            ax1.plot(x,y)
        except:
            print("Could not plot")
        img= np.rot90(img)
        ax2.imshow(img)

        #count= range(256)
        #ax3.hist(count, 255, weights=count, range=[0,256])
        ax3.hist(gray.ravel(),256,[0,256])

        plt.show()



        #wait for key pressing
        cv2.waitKey(0)

        #Distroy all the cv windows
        cv2.destroyAllWindows()
        return render(request,'users/upload.html')

    else:
        return render(request,'users/upload.html')


def training(request):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import keras.backend as k
    import cv2
    from keras.preprocessing.image import ImageDataGenerator
    from keras.applications.vgg16 import preprocess_input,VGG16
    from django.conf import settings
    import os

    print('trying to read dataset')
    train_dir = os.path.join(settings.MEDIA_ROOT, 'Dataset','train')
    test_dir = os.path.join(settings.MEDIA_ROOT, 'Dataset','valid')
    print('datasets loaded')


    train_imgs_path=pd.read_csv(os.path.join(settings.MEDIA_ROOT ,'Dataset/train_image_paths.csv'))
    train_labels=pd.read_csv(os.path.join(settings.MEDIA_ROOT, 'Dataset/train_labeled_studies.csv'))
    test_imgs_path=pd.read_csv(os.path.join(settings.MEDIA_ROOT, 'Dataset/valid_image_paths.csv'))
    test_labels=pd.read_csv(os.path.join(settings.MEDIA_ROOT, 'Dataset/valid_labeled_studies.csv'))
    print('csv loaded')

    train_labels['Body_Part']=train_labels['Img_Path'].apply(lambda x: str(x.split('/')[2])[3:])
    train_labels['Study_Type']=train_labels['Img_Path'].apply(lambda x: str(x.split('/')[4])[:6])
    test_labels['Body_Part']=test_labels['Img_Path'].apply(lambda x: str(x.split('/')[2])[3:])
    test_labels['Study_Type']=test_labels['Img_Path'].apply(lambda x: str(x.split('/')[4])[:6])


    def read_image(Path):
        img=cv2.imread(Path)
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(224,224))
        #print (img.shape)
        img=np.array(img)
        #img=np.resize(img,(224,224))
        #print (img.shape)
        img=img/255.
        return img

    IMG_SIZE=(224,224)
    train_datagen=ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=30,
        fill_mode='nearest',
        preprocessing_function=preprocess_input,
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_imgs=list(train_imgs_path.Img_Path.values)
    valid_imgs=list(test_imgs_path.Img_Path.values)

    batch_size=128

    train_generator = train_datagen.flow_from_directory(
        train_dir,  # this is the target directory
        target_size=IMG_SIZE,  # all images will be resized to 224*224
        batch_size=batch_size,
        class_mode='binary') 
    print('train_generator ............')

    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='binary')
    print('validation_generator ............')
    VGGmodel=VGG16(input_shape=(224,224,3),weights='imagenet',include_top=False)

    from keras.layers import  Dense,Flatten,Dropout
    from keras.models import  Sequential
    from tensorflow.keras.optimizers import Adam

    model=Sequential()
    model.add(VGGmodel)
    model.add(Flatten(input_shape=model.output_shape[1:]))
    model.add(Dropout(0.5))
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))

    for layer in model.layers:
        layer.trainable=False

    model.compile(loss='binary_crossentropy',
                optimizer=Adam(lr=1e-3),
                metrics=['accuracy'])

    batch_size=128

    model.fit(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)

    acc = model.history.history['accuracy']
    print(acc)
    return render(request,'users/training.html',{'acc':acc})
