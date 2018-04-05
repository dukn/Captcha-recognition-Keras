<?php

include(__DIR__.'/../CaptchaBuilderInterface.php');
include(__DIR__.'/../PhraseBuilderInterface.php');
include(__DIR__.'/../CaptchaBuilder.php');
include(__DIR__.'/../PhraseBuilder.php');

use Gregwar\Captcha\CaptchaBuilder;

$myfile = fopen('Data/pass.txt','w'); 
for ($i=0; $i <20000; $i++){
    $captname  = sprintf('Data/%d.jpg',$i);                                                
    $captcha = new CaptchaBuilder;
    $captcha->build()->save($captname); 
    $xxx = $captcha->getPhrase();
    fwrite($myfile,$xxx);
    fwrite($myfile,' ');
    if (($i) % 100 == 0) echo ".";
}      
fclose($myfile);
echo "\n";