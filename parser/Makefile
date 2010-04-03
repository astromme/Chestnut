#
# Sample lex/yacc Makefile
# Created: Shawn Ostermann - Sun Feb  1, 1998
# Edited: Brett Humphreys - Sun Sept 17, 2000
#	handles the lex and flex flags by setting the LEXCCFLAGS VARIABLE
# Edited: Brett Humphreys - Thurs Oct 12, 2000
#	handles yacc and bison by setting the ANIMAL variable to what parser you want to use
#       but you need to comment out the -y option for bison flags, if you are using yacc.
#
# Edited: Brett Humphreys - Thurs Oct 12, 2000
#	okay there seems to be comflicting sources as to how to do if statements in make, but 
#	below seems to work.  just simply select flex/lex or bison/yacc and it will automatically
#	choose your flags for you.

# Credits:  The if else structure was taken from this reference:
#		http://www.gnu.org/manual/make/html_mono/make.html#SEC73




#######################################
#
# These are just constants, that should be modified
# for your program
#
#######################################
#The name of the binary make will create
PROGRAM = testmap

#The actual name of your lex/flex file	
LEXFILE = testmap.l

#The actual name of the yacc/bison file
YACCFILE = testmap.y

#Some source files that should be compiled in
CFILES = SymbolTable.cpp Utilities.cpp

#General .h files used in your program 
#that everything should be compiled against
HFILE = SymbolTable.h SymbolEntry.h Utilities.h

#flags to your Compiler
CFLAGS = -g -Wall

#The compiler
CC = g++

#Determines what lexer you are going to use, either flex or lex.
LEXER = flex

#Defines which beast you will parse with, bison or yacc
PARSER = bison


##################################################
# Most of the following don't need to be messed with 
##################################################
LFLAGS = #these are flags that are options actually for lex/flex
PARSEFLAGS= -dvt #Defines flags that either yacc or bison will want/need for you
FLEXFLAGS = --c++ #Flags specific to flex
LEXFLAGS = --c++ #Flags specific to lex
BISONFLAGS = -y -v#Flags specific to bison
YACCFLAGS = #Flags specific to Yacc

##################################################
#
# You shouldn't need to change anything else
#
##################################################

ifeq ($(LEXER), flex) 
	#use the flex flags for GCC
	LEXCCFLAGS = -lfl
	LEXERFLAGS = ${LFLAGS} ${FLEXFLAGS}
else				#else lex is used
	#lex flags for GCC
	LEXCCFLAGS = -ll
	LEXERFLAGS = ${LFLAGS} ${LEXFLAGS}
endif

ifeq ($(PARSER), bison)				 #if bison chosen, need to specify yacc output files
	#the -y outputs y.tab.c file.
	IPARSEFLAGS = ${PARSEFLAGS}  ${BISONFLAGS}
else						 #yacc was chosen for the parser
	#default flags for yacc
	IPARSEFLAGS = ${PARSEFLAGS} ${YACCFLAGS}
endif

# compute the OFILES
#OFILES = ${CFILES:.c=.o}

# all of the .o files that the program needs
OBJECTS = y.tab.o lex.yy.o ${CFILES}

# How to make the whole program
${PROGRAM} : ${OBJECTS}
	${CC} ${CFLAGS} ${OBJECTS} -o ${PROGRAM} ${LEXCCFLAGS}


# 
# Turn the parser.y file into y.tab.c using "yacc"
# 
# Also, yacc generates a header file called "y.tab.h" which lex needs
# It's almost always the same, so we'll have lex use a different
# file and just update it when y.tab.h changes (to save compiles)
#
y.tab.c : ${YACCFILE} ${HFILES}
	${PARSER} ${IPARSEFLAGS} ${YFLAGS} ${YACCFILE}
y.tab.o: y.tab.c
	${CC} -g -c y.tab.c
y.tab.h: y.tab.c
parser.h: y.tab.h
	cmp -s y.tab.h parser.h || cp y.tab.h parser.h

# 
#  Turn the scanner.l file into lex.yy.c using "lex"
# 
lex.yy.c : ${LEXFILE} ${HFILE}
	${LEXER} ${LFLAGS} ${LEXFILE}
lex.yy.o: lex.yy.c
	${CC} -g -c lex.yy.c

#
# File dependencies
#
${OFILES}: ${HFILE} 
#parser.h

#
# Compile actual CUDA code
#
cudalang : cudalang.cpp
	${CC} -g -Wall -o cudalang cudalang.cpp

clean:
	/bin/rm -f *.o lex.yy.c y.tab.c c.tab.h c.output c.tab.c ${PROGRAM} parser.h y.output y.tab.h core *.*~ *~ #*#

