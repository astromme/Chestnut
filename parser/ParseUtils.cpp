#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <boost/algorithm/string/replace.hpp>
#include <boost/regex.hpp>
#include "ParseUtils.h"

using namespace std;
using boost::algorithm::replace_all;

/****************************************
 * NOTE: This file only constains the constructor and destructor.
 * The ParseUtils class is in charge of outputting all the Thrust code, and
 * as such is very large.
 * 
 * To make the ParseUtils class more readable and manageable, sections of 
 * the class are broken up into different files that have the 
 * following naming scheme:
 *      ParseUtils_<description>.cpp
 * 
 * Currently the extra files are:
 *    ParseUtils_Functions.cpp
 *      -- writes basic functions like map, sort, reduce
 *      
 *    ParseUtils_Foreach.cpp
 *      -- writes the foreach loop
 *
 *    ParseUtils_HelperFunctions.cpp
 *      -- helper functions that other functions in ParseUtils uses. Does NOT write any code!
 *
 *    ParseUtils_InputOutput.cpp
 *      -- writes I/O like read, write, print
 *
 *    ParseUtils_Timer.cpp
 *      -- writes timer code
 *
 *    ParseUtils_Variables.cpp
 *      -- writes variable declarations
 */

// constructor
ParseUtils::ParseUtils(string outfname) {
  string cudafname = outfname + ".cu";
  string cppfname = outfname + ".cpp";
  string headerfname = outfname + ".h";

  // initialize files so that it's blank (with ios::trunc)
  cudafile = new FileUtils(cudafname);
  cppfile = new FileUtils(cppfname);
  headerfile = new FileUtils(headerfname);

  parselibs_included = false;
  // cudalang_included = false; // TODO Don't need cudalang?
  rand_included = false;

  indent = 0; // initially no indent offset 
  fcncount = 0; // no functions yet
} 

// destructor
ParseUtils::~ParseUtils(){
  // delete FileUtils
  delete cudafile;
  delete cppfile;
  delete headerfile;
}