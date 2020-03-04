## DeepSpeech
 This project is about using keras to rebuilding DeepSpeech from 
 
https://arxiv.org/abs/1412.5567
https://github.com/mozilla/DeepSpeech 


## project structure
main training procedure is in DeepSpeech.py
model is Model1.py
dealing with data preprocessing and data generator util/Feeding.py
all of the parameter of the project util/Flags.py


## Easy implement step. 

clone the project
```
git clone https://github.com/WalterJohnson0/DeepSpeech-KerasRebuild.git
```
swith to common voice EN data folder
```
cd DeepSpeech-KerasRebuild/data/CV_EN
```
Download common_voice_EN
```
wget https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/en.tar.gz
```

unpack the data en.tar.gz
```
tar xvzf file.tar.gz
```

go back to the main folder and run the program

```linux
cd ../../
python DeepSpeech.py
```

If you want to change the parameter, use the help -helpfull. 
```
python DeepSpeech.py -helpfull
```
