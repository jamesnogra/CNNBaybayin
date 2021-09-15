# CNNBaybayin
This repository already contains the trained CNN model for Baybayin. For testing, refer to the "Testing" section of this readme.

### Prerequisites/Installation
* Install python 3.6 if you are in Windows. For Mac or Ubuntu, make sure your python version is 3.x
* If you are using Python 2.X, then use `pip2 xxx xxx` for the installations that follow
* Install OpenCV for image manipulation ```pip install opencv-python```
* Install numpy for arrays ```pip install numpy```
* Install tensorflow ```pip install tensorflow```
* Install TFLearn ```pip install tflearn```
* Install tqdm for progress bars ```pip install tqdm```
* Install matplotlib for displaying graphed results ```pip install matplotlib```
* Install other libraries ```apt update && apt install -y libsm6 libxext6```
* Install optional libraries ```pip3 install flask```
* Install optional libraries ```pip3 install flask-cors```
* For Ubuntu or Mac users, you might be required to install Tkinter Package, to do this, just execute ```apt-get install -y python3-tk``` for Python3.X or ```apt-get install python2.7-tk``` for Python 2.7

### Testing the Model
To test the model, just run the command ```python test.py "C:\Users\lenovo\Documents\Images\ba-test-image.jpg"```. The output would be something like this:
```
RESULTS:
be/bi 0.1196 %
bo/bu fruit 0.2781 %
ba 99.0874 %
la 0.5148 %
Image is class ba
```
