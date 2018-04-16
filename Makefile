SHELL = /bin/bash
DIRS:=$(wildcard ./*)

.PHONY:all clean
ALL:
	echo $(DIRS)	
COMPILE:$(DIRS)
	$(foreach i, $(DIRS), $(TEX) $(CONFIG) $i;)
	$(foreach i, $(DIRS), $(TEX) $(CONFIG) $i;)
CLEAR:
	mv  $(OBJFILE) $(wildcard ./bin/*.log ./bin/*.aux ./bin/*.nav \
./bin/*.toc ./bin/*.snm ./bin/*.out ./bin/*.vrb) ./obj
