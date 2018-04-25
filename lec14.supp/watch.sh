#!/bin/bash 

FILE=main.tex
DIR=lec14.supp
FROM=/home/xinglu/Desktop/opt
TO=prj/

if [[ $(hostname) == "luzai-PC" ]]; then
    echo "ON luazai PC"
    inotifywait -m -e close_write $FILE | 
    while read -r filename events; do
        echo ">> GUARD: running rsync" ${events} ${filename}
        rsync -avzP ${FROM} amax:${TO}
    done
else
    echo "ON host" $(hostname)
    inotifywait -m -e close_write,modify,close,move,create $FILE | 
    while read -r filename events; do
        echo ">> GUARD: running make" ${events} ${filename}
        make 
        rsync ./main.pdf luzai:/tmp/
    done
fi
