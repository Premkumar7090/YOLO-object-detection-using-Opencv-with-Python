open("link.html", "w").write('<a href="https://www.linkedin.com/in/prem-kumar-4159271b6/"> Link </a>')

"""

Copyright 2021 Google LLC


Connect to my Linkedin
     https://www.linkedin.com/in/prem-kumar-4159271b6/

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
                                        LETS START✔
"""

import cv2
import numpy as np

#Load YOLO Algorithms
net=cv2.dnn.readNet(r"C:\Users\premk\Desktop\volume control using hand tracking\volume\yolov3.weights",r"C:\Users\premk\Desktop\volume control using hand tracking\yolov3.cfg.txt")


#To load all objects that have to be detected
classes=[]
with open(r"C:\Users\premk\Desktop\volume control using hand tracking\coco.names.txt","r") as f:
    read=f.readlines()
for i in range(len(read)):
    classes.append(read[i].strip("\n"))

#print(classes)


#Defining layer names
layer_names=net.getLayerNames()
output_layers=[]
for i in net.getUnconnectedOutLayers():
    output_layers.append(layer_names[i[0]-1])

#print(output_layers)    

#Loading the Image
img=cv2.imread(r'C:\Users\premk\Desktop\volume control using hand tracking\volume\r424_0_2978_1429_w1200_h678_fmax.jpg')
height,width,channels=img.shape


#Extracting features to detect objects
blob=cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)
                                       #Standard         #Inverting blue with red
                                       #ImageSize        #bgr->rgb


#We need to pass the img_blob to the algorithm
net.setInput(blob)
outs=net.forward(output_layers)
#print(outs)

#Displaying information on the screen
class_ids=[]
confidences=[]
boxes=[]
for output in outs:
    for detection in output:
        #Detecting confidence in 3 steps
        scores=detection[5:]                #1
        class_id=np.argmax(scores)          #2
        confidence =scores[class_id]        #3

        if confidence >0.5: #Means if the object is detected
            center_x=int(detection[0]*width)
            center_y=int(detection[1]*height)
            w=int(detection[2]*width)
            h=int(detection[3]*height)

            #Drawing a rectangle
            x=int(center_x-w/2) # top left value
            y=int(center_y-h/2) # top left value

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

print(class_ids)
print(confidences)
print(boxes)


#Removing Double Boxes
indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.3,0.4)

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]  # name of the objects
       
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)


cv2.imshow("Output",img)
cv2.waitKey(0)
cv2.destroyAllWindows()        

