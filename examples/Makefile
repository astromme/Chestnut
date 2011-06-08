#
# make: builds minimal
# make TARGET=filename:  builds another example file
# make clean
# make clean TARGET=filename
#

CC=nvcc

INCLUDES=-I.

# default TARGET set to compile minimal.chestnut
TARGET ?= minimal

all: $(TARGET) 

$(TARGET): $(TARGET).chestnut
	../parser/parser.py $(TARGET).chestnut $(TARGET)
	$(CC) $(INCLUDES) -o $(TARGET) $(TARGET).cu

clean: 
	$(RM) $(TARGET)


