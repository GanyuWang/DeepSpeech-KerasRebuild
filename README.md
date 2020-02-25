# DeepSpeech
 This project is about using keras to rebuilding DeepSpeech from 
 
https://arxiv.org/abs/1412.5567
https://github.com/mozilla/DeepSpeech 

To build the project

## Download common_voice_EN
https://voice.mozilla.org/en/datasets

the default achieve for the download is 'data/CV_EN/'
To check if  you made it correctly, after you decompress the file there should be 'data/CV_EN/clips', in which save all of the audio file.

## Run the program
using the following graph will run the training process with the default paremeter. 
```linux
python DeepSpeech.py
```
If you want to change the parameter, use the help -h. 
```linux
python DeepSpeech.py -h
```
