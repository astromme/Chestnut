# make: builds minimal
# make TARGET=filename  builds another example file
#
CC=nvcc

INCLUDES=-I.

# default TARGET set to build minimal.chestnut
TARGET ?= minimal

all: $(TARGET) 

$(TARGET): $(TARGET).chestnut
	../parser/parser.py $(TARGET).chestnut $(TARGET)
	$(CC) $(INCLUDES) -o $(TARGET) $(TARGET).cu

clean: 
	$(RM) $(TARGET)


