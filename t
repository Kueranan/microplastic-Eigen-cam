'''
            model = YOLO('models/best_mic_L_edit.pt')
            #img = cv2.imread('images/puppies.jpg')
            plt.rcParams["figure.figsize"] = [3.0, 3.0]
            img = cv2.resize(uploaded_image, (640, 640))
            rgb_img = img.copy()
            img = np.float32(img) / 255
            
            target_layers =[model.model.model[-4]]
            
            cam = EigenCAM(model, target_layers,task='od')
            grayscale_cam = cam(rgb_img)[0, :, :]
            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            
            st.image(cam_image, caption='Eigen',
                     use_column_width=True)
            '''
            
            
            
            
            '''
            img = np.array(uploaded_image)
            img = cv2.resize(img, (640, 640))
            rgb_img = img.copy()
            img = np.float32(img) / 255
            
            #target_layers = [model.model.model[-4]]
            
            cam = EigenCAM(model, target_layers, task='od')
            grayscale_cam = cam(rgb_img)[0, :, :]
            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            st.image(cam_image, caption='Eigen', use_column_width=True)
            
            
    
            
            '''
            '''
            st.sidebar.button('Detect Objects')
                       
            res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
            
            boxes = res[0].boxes
            #boxes_np = boxes.xyxy.cpu().numpy()
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
            '''
            
            class_counts = Counter([model.names[int(cls)] for cls in boxes.cls])
   
            try:
                
                 with st.expander("Detection Results"):
                     
                     #for box in boxes:
                         #st.write(box.cls)
                     
                    for class_name, count in class_counts.items():     #box in boxes:
                        st.write(f"{count} {class_name}s")                                             
                        #st.write(box.data)
            except Exception as ex:
                              
                 st.write("No image is uploaded yet!")
'''