#!/usr/bin/env bash 

file=03-mlp.pdf

pushd ~/Downloads 
pdftotext -layout $file out1.txt 
iconv -c -f utf-8 -t ascii out1.txt > out.txt 
popd 

cp ~/Downloads/out.txt . 