#!/usr/bin/env bash 

file=03-mlp.pdf

pushd ~/Downloads 
pdftotext -layout $file out1.md 
iconv -c -f utf-8 -t ascii//TRANSLIT out1.md > out2.md 
sed 's/^o /- /g' out2.md > out3.md
sed 's/?//g' out3.md > out4.md
sed 's/.*CSE 5526: .*//g' out4.md > out5.md
sed 's/^\s\{9,\}/### /g' out5.md > out6.md
sed 's/^\s\{3,\}o /    - /g' out6.md > out7.md
popd 

cp ~/Downloads/out*.md . 