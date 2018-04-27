# face-recognition-drone
[![built with Python3](https://img.shields.io/badge/built%20with-Python3-red.svg)](https://www.python.org/)<br><br>
Recognizes Peoples From Drone Cameras<br>
### Installation:
First Open Command Shell / Terminal :

```bash
1. pip install -r requirements.txt
```
#### Execute It:
```bash
1. python webcam.py
OR
2. python drone.py
```
#### Add People's Database Of Face:
  - In Folder "training-data" make new folder names 's*', where * is number of this face. Means if there is database of 3 faces, folder name should be s1,s2 and s3
  - Then in repective folder add images of that person starting from 1.jpg to n.jpg . suppose you add 5 images of a person in s3 then the names of images should be 1.jpg, 2.jpg, 3.jpg, 4.jpg and 5.jpg.
  - Add **person's name** in **name.txt**.
  - Add **person's status** in **status.txt** (vip or blacklisted) . Make sure you give correct spelling of blacklisted or vip.
  - Thats it ! Now Execute The Code from webcam or from drone.

#### Edit Config.py:
  - Paste the IP of your webcam streaming in config.py
  - And Edit Scale According to your need . (Scale : It is the number from 1.01 to 1.7 , lower the scale more is the scanning of face in image , greater the scale more it works smooth)
