from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui import QPixmap
import cv2
import os
import glob
from pymodbus.client import ModbusSerialClient
import serial
import keyboard
import time
import random
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import PIL
from PIL import Image
 
class CamWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.cam = cv2.VideoCapture(0)
        self.swinecount = 0
        self.populate_ui()
        self.frame_count = 0
        self.measure_count = 0
        self.detect_seg = 0
        self.save_path = r"saved_frames"
        self.saveseg_path = r"saved_segment"
        self.savemeasure_path = r"save_measure"
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.saveseg_path, exist_ok=True)
        os.makedirs(self.savemeasure_path, exist_ok=True)
      
        # Load the YOLO model
        self.model = YOLO(r"C:\Users\66909\Downloads\model wieght\best weight for object detection.pt", "v8")
        self.segmentmodel = YOLO(r"C:\Users\66909\Downloads\model wieght\best latest.pt", "v8")
        self.save_timer = QtCore.QTimer(self)
        self.save_timer.timeout.connect(self.save_frame)
        self.save_timer.setInterval(1000)  # Interval between saving frames in milliseconds
        self.save_timer.setSingleShot(True)  # Only triggers once
        
        
        self.modbus_client = ModbusSerialClient(method='rtu', port='COM7', baudrate=115200, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE)
        
        self.modbus_timer = QtCore.QTimer(self)
        self.modbus_timer.timeout.connect(self.read_modbus)
        self.modbus_timer.start(100)
      
#     def find_highest_point(mask_tensor):
#     # Convert tensor to NumPy array
#         mask_np = mask_tensor.squeeze().cpu().numpy()  # Assuming the mask tensor is on the CPU

#         # Find the coordinates where the mask is 'True' (non-zero)
#         y_coordinates, _ = np.where(mask_np > 0)

#         # Get the maximum Y-coordinate (lowest point since the image origin is at the top-left)
#         highest_point = y_coordinates.min() if len(y_coordinates) > 0 else 0
#         return highest_point


# # Function to sort masks based on the highest point
#     def sort_masks_by_highest_point(masks):
#         mask_points = [(idx, find_highest_point(mask_tensor)) for idx, mask_tensor in enumerate(masks)]
#         sorted_masks = sorted(mask_points, key=lambda x: x[1], reverse=True)
#         return sorted_masks



    def resize_and_pad_images(input_folder, output_folder, target_size=(640, 640)):
        os.makedirs(output_folder, exist_ok=True)
        for filename in os.listdir(input_folder):
            if filename.endswith(".jpg"):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)

                # Open the image
                img = Image.open(input_path)

                # Resize to target height
                img_resized = img.resize((img.width * target_size[1] // img.height, target_size[1]), Image.ANTIALIAS)

                # Create a new image with the target size
                new_img = Image.new("RGB", target_size, (0, 0, 0))

                # Calculate left padding
                left_padding = (target_size[0] - img_resized.width) // 2

                # Paste the resized image in the center
                new_img.paste(img_resized, (left_padding, 0))

                # Save the padded image
                new_img.save(output_path)

    def populate_ui(self):
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setGeometry(10, 10, 640, 480)  # Adjust dimensions as needed
 
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // 30)  # Adjust frame rate (30 fps in this case)
 
    def update_frame(self):
        ret, frame = self.cam.read()
        if ret:
            # Preprocess the frame for YOLO
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Detect objects in the frame using YOLO
            results = self.model(rgb_frame)
            # Check if results is a list or an object
            # print(results[0].plot())
 
            frame = results[0].plot()
            #for r in results:
                #for box in boxes:
                    
         
            
            # Display the frame with detected objects
            
            image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.image_label.setPixmap(pixmap)
 
            #if isinstance(results, list):
                # No detections
                #print("No detections")
            # else:
            #     # Draw rectangles around the detected objects
            #     # for result in results[0].plot():
            #     #     x1, y1, x2, y2, _ = result
            #     #     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
 
            #     frame = results[0].plot()
 
            #     # Display the frame with detected objects
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #     image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888)
            #     pixmap = QPixmap.fromImage(image)
            #     self.image_label.setPixmap(pixmap)
 
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_S:
            self.pbottom = []
            self.ptop = []
            self.swinecount += 1
            print("---------------------------Swine No :" + str(self.swinecount) + "--------------------------")
            self.delete_existing_files()
            self.frame_count = 0
            self.measure_count = 0
            self.detect_seg = 0
            self.save_frames(5)
            
 
    def delete_existing_files(self):
        # Delete existing files in the folder
        files = glob.glob(os.path.join(self.save_path, '*'))
        for f in files:
            os.remove(f)
        files_Seg = glob.glob(os.path.join(self.saveseg_path, '*'))
        for s1 in files_Seg:
            os.remove(s1)
        files_measure = glob.glob(os.path.join(self.savemeasure_path, '*'))
        for m1 in files_measure:
            os.remove(m1)
 
    def save_frames(self, num_frames):
        for _ in range(num_frames):
            self.save_timer.start()
            QtWidgets.QApplication.processEvents()  # Process events to allow GUI updates
            #QtCore.QThread.msleep(1000)  # Sleep for 1 second
            self.save_timer.timeout.emit()  # Manually emit the timer timeout signal
 
    def save_frame(self):
        
        def find_highest_point(mask_tensor):
    # Convert tensor to NumPy array
            mask_np = mask_tensor.squeeze().cpu().numpy()  # Assuming the mask tensor is on the CPU

                            # Find the coordinates where the mask is 'True' (non-zero)
            y_coordinates, _ = np.where(mask_np > 0)

                            # Get the maximum Y-coordinate (lowest point since the image origin is at the top-left)
            highest_point = y_coordinates.min() if len(y_coordinates) > 0 else 0
            return highest_point


                    # Function to sort masks based on the highest point
        def sort_masks_by_highest_point(masks):
            mask_points = [(idx, find_highest_point(mask_tensor)) for idx, mask_tensor in enumerate(masks)]
            sorted_masks = sorted(mask_points, key=lambda x: x[1], reverse=True)
            return sorted_masks
                
        ret, frame = self.cam.read()
        if ret:
            # Preprocess the frame for YOLO
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Detect objects in the frame using YOLO
            results = self.model(rgb_frame)
            
            
            #iframe = results[0].plot()
            if results[0].boxes.cls.numel() != 0 :
                self.frame_count += 1
                rgb2_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
                filename = os.path.join(self.save_path, f"frame_{self.frame_count}.png")
                cv2.imwrite(filename, rgb2_frame)
                rgb3_frame = rgb2_frame
                result = self.segmentmodel(rgb3_frame)
                if result[0].boxes.cls.numel() >= 3 :
                    self.detect_seg += 1
                    seg_frame = result[0].plot()
                    filename2 = os.path.join(self.saveseg_path, f"frame_seg_{self.detect_seg}.png")
                    cv2.imwrite(filename2, seg_frame)
                    
                # else :
                #     if self.detect_seg < 5:
                #         self.save_frames(5 - self.detect_seg)
                
            
           
                    

                    
                    pos = 0


                    # Path to the folder containing images
                    folder_path = r'C:\Users\66909\Downloads\model wieght\saved_segment'
                


                    # Open the CSV file in write mode


                    # Iterate through each image in the folder
                        # Get a sorted list of image filenames in numerical order
                    # image_filenames = sorted([filename for filename in os.listdir(folder_path) if filename.endswith('.jpg') or filename.endswith('.png')],
                    #                         key=lambda x: (x.split('_png')[0])if x.split('_png')[0].isdigit() else (x.split('_PNG')[0]))

                    

                        # Iterate through each image in the sorted list
                    #for filename in image_filenames:
                            # print("aha:" + filename)
                            # if filename.endswith('.jpg') or filename.endswith('.png'):
                
                    filename_m = os.path.join(folder_path, f"frame_seg_{self.detect_seg}.png")
                            # Load the image
               
                            # original_image = cv2.imread(folder_path)
                    original_image = cv2.imread(filename_m)
                        #     # Assuming you have results and want to get masks for each image
                        # # Extract the number from the filename
                            # print('this is result')
                            # print(result[pos])
                            #print("this is len")
                            #print(len(result[pos].masks.item))
                # print(result[0].boxes.cls.numel())
                # if (result[0].boxes.cls.numel() < 3):
                #         self.save_frames(5 - self.frame_count)
                #         print(f"Image {filename} has missing masks.")
                #         pos += 1
                #         chk += 1
                        
                           
                            # if result[pos].masks == None :
                            #     print("not found mask")
                            #     break
                            # else :
                
                            # print("mask found")
                    #             print(masks[0])
                    #             print(masks[1])
                    #             print(masks[2])
               
                    masks = result[0].masks.data   
                    image_masks = masks  # Assuming 'masks' is the mask data for each image

                        # # Sort masks based on area (Modify this sorting method as per your requirement)
                    sorted_masks = sort_masks_by_highest_point(image_masks)

                        # # # Rearrange masks based on the sorted order
                    sorted_mask_tensors = [image_masks[idx] for idx, _ in sorted_masks]

                        # # # Replace the original masks with the sorted ones
                    masks = sorted_mask_tensors

                                # Assuming masks[0] is used as reference mask for top and bottom positions
                    mask_tensor_1 = masks[0]

                                # Convert tensor to NumPy array
                    mask_np_1 = mask_tensor_1.squeeze().cpu().numpy()
                    mask_uint8_1 = (mask_np_1 * 255).astype(np.uint8)

                                # Find contours in mask 1
                    contours_1, _ = cv2.findContours(mask_uint8_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                # Find the largest contour in mask 1
                    largest_contour_1 = max(contours_1, key=cv2.contourArea)
                                # Find topmost and bottommost points of the largest contour in mask 1
                    topmost_1 = tuple(largest_contour_1[largest_contour_1[:, :, 1].argmin()][0])
                    bottommost_1 = tuple(largest_contour_1[largest_contour_1[:, :, 1].argmax()][0])

                                # Convert tensor 2 to NumPy array
                    mask_tensor_2 = masks[1]
                    mask_np_2 = mask_tensor_2.squeeze().cpu().numpy()

                                # Get the Y-axis positions from mask 1
                    y_top = topmost_1[1]
                    y_bottom = bottommost_1[1]
                    x_top = topmost_1[0]
                    x_bottom = bottommost_1[0]

                                # Find edge points of mask 2 at the same Y-axis positions as mask 1's topmost and bottommost points
                    edge_points_top = []
                    edge_points_bottom = []

                    for x in range(mask_np_2.shape[1]):
                        column = mask_np_2[:, x]
                        if column[y_top] > 0:
                            edge_points_top.append((x, y_top))
                        if column[y_bottom] > 0:
                            edge_points_bottom.append((x, y_bottom))
                                # print("hey you")
                                # line_length_top = abs(edge_points_top[0][0] - edge_points_top[-1][0])
                                # line_length_bottom = abs(edge_points_bottom[0][0] - edge_points_bottom[-1][0])
                                # print(f"File: {filename}")

                                # line_length_top_cm = round(line_length_top /4 * 2) / 2  # Round to nearest 0.5 cm
                                # line_length_bottom_cm = round(line_length_bottom / 4 * 2) / 2
                                # if line_length_top_cm <= 3 and line_length_bottom_cm <= 4:
                                #     r = "pass"
                                # else:
                                #     r = "not pass"




                                # print(f"Length of the horizontal line at topmost (P2) position: {line_length_top} pixels")
                                # print(f"Length of the horizontal line at bottommost (P1) position: {line_length_bottom} pixels")
                                # print(f"Length of the horizontal line at topmost (P2) position: {line_length_top_cm} cm  ")
                                # print(f"Length of the horizontal line at bottommost (P1) position: {line_length_bottom_cm} cm")
                                # print(f"Criteria: {r}")
                    pos = pos + 1
                                # # Mark edge points on the original image
                    marked_image = cv2.cvtColor(original_image.copy(), cv2.COLOR_BGR2RGB)
                            
                    cv2.circle(marked_image, [x_top,y_top], 2, (0, 255, 0), -1)
                    cv2.circle(marked_image, [x_bottom,y_bottom], 2, (0, 255, 0), -1)
                    for point in edge_points_top:
                            cv2.circle(marked_image, point, 1, (0, 255, 0), -1)

                    for point in edge_points_bottom:
                            cv2.circle(marked_image, point, 1, (0, 255, 0), -1)

                    max_display_size = 600
                    scale_factor = min(max_display_size / marked_image.shape[1], max_display_size / marked_image.shape[0])
                    resized_image = cv2.resize(marked_image, (0, 0), fx=scale_factor, fy=scale_factor)

                            #Display the resized image with marked edge points
                                #cv2.imshow(cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
                    self.measure_count += 1
                    bgr_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
                    filename_m = os.path.join(self.savemeasure_path, f"frame_measure_{self.measure_count}.png")
                    # cv2.imshow("Image", bgr_image)
                    # cv2.waitKey(0)
                    cv2.imwrite(filename_m, bgr_image)
                    # QtCore.QThread.msleep(1000)
                    # cv2.destroyAllWindows()
                                # Calculate the length of the horizontal lines
                    if len(edge_points_top) >= 2 and len(edge_points_bottom) >= 2:
                        
                        line_length_top = abs(edge_points_top[0][0] - edge_points_top[-1][0])
                        line_length_bottom = abs(edge_points_bottom[0][0] - edge_points_bottom[-1][0])
                        print(f"File detect: {filename}---- File segment {filename_m}")

                        line_length_top_cm = round(line_length_top /4 * 2) / 2  # Round to nearest 0.5 cm
                        line_length_bottom_cm = round(line_length_bottom / 4 * 2) / 2
                        self.ptop.append(line_length_top_cm)
                        self.pbottom.append(line_length_bottom_cm)
                        maxtop = max(self.ptop)
                        maxbuttom = max(self.pbottom)
                        
                        
                        if maxtop <= 3 and maxbuttom <= 4:
                            r = "pass"
                        else:
                            r = "not pass"



                        print(f"----------------------------------- the cm Length for frame : {self.measure_count} ----------------------------------- ")
                        print(f"Length of the horizontal line at topmost (P2) position: {line_length_top} pixels")
                        print(f"Length of the horizontal line at bottommost (P1) position: {line_length_bottom} pixels")
                        print(f"Length of the horizontal line at topmost (P2) position: {line_length_top_cm} cm  ")
                        print(f"Length of the horizontal line at bottommost (P1) position: {line_length_bottom_cm} cm")
                        
                        
                        print(f"----------------------------------- the cm Length for this Swine : {self.swinecount} ---------------------------------")
                        print(f"Length of the horizontal line at topmost (P2) position: {maxtop} cm  ")
                        print(f"Length of the horizontal line at bottommost (P1) position: {maxbuttom} cm")
                        print(f"Criteria: {r}")
           
            

                
            
     
            
            # # Save the frame if any objects are detected
            # if not isinstance(results, list) and results.xyxy[0] is not None:
            #     filename = os.path.join(self.save_path, f"frame_{self.frame_count}.png")
            #     cv2.imwrite(filename, frame)
            #     self.frame_count += 1
 
    def closeEvent(self, event):
        self.cam.release()
        event.accept()
 
   
    def read_modbus(self):
        if self.modbus_client.connect():
            try:
                result = self.modbus_client.read_holding_registers(0x9C7D, 1, unit=128)
                if not result.isError():
                    print("Holding register value:", result.registers[0])
                    if 100 <= result.registers[0] <= 1000:
                        self.delete_existing_files()
                        self.pbottom = []
                        self.ptop = []
                        self.swinecount += 1
                        print("----------------------Swine No :" + str(self.swinecount) + "----------------------")
                        self.frame_count = 0
                        self.measure_count = 0
                        self.detect_seg = 0
                        self.save_frames(5)
            except Exception as e:
                print("Modbus Error:", e)
            finally:
                self.modbus_client.close()  
 
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    widget = CamWidget()
    widget.show()
    app.exec()