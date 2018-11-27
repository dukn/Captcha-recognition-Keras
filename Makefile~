clean:
	rm -f Captcha_Generator/Captcha-php/demo/Data/*
	rm -f  Data/*
	rm -f *.pkl.gz
	rm -f *.pyc

captcha:	
	php Captcha_Generator/Captcha-php/demo/demo.php

preprocessing:
	python Preprocessing.py

train:
	python Train.py
recognition:
	python Recognition.py
