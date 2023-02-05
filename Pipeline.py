from object_detector import detector
import os
from pathlib import Path
import PyQt5


filename='WithCUP.5eada684-83b6-11ed-bac8-98fa9bfba7d0.jpg'
#dire, filename= os.path.split(filename)
#os.chdir(dire)
#path= os.path.join(dire, filename)
model = detector(filepath=filename)

model.image_detection()