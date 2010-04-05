#include "FileUtils.h"

using namespace std;

// constructor
FileUtils::FileUtils(string fname){
  // initialize files so that it's blank (with ios::trunc)
  outfile.open(fname.c_str(), ios::trunc);
}

// destructor
FileUtils::~FileUtils(){
  outfile.close();
}

/********************************
 * Function: writeAll
 * ------------------
 * Writes contents of vectors containing C++ code to the outfile
 */
void FileUtils::writeAll(){
  // add newlines to each vector so we have some readable code
  includeStrs.push_back("\n");
  fcnDecStrs.push_back("\n");
  mainStrs.push_back("\n");
  fcnDefStrs.push_back("\n");

  // write #includes
  for (unsigned int i=0; i<includeStrs.size(); i++)
    outfile << includeStrs[i];

  // write function declarations
  for (unsigned int i=0; i<fcnDecStrs.size(); i++)
    outfile << fcnDecStrs[i];

  // write main()
  for (unsigned int i=0; i<mainStrs.size(); i++)
    outfile << mainStrs[i];

  // write function definitions
  for (unsigned int i=0; i<fcnDefStrs.size(); i++)
    outfile << fcnDefStrs[i];
}

// push back strings to various vectors
void FileUtils::pushInclude(string include){ includeStrs.push_back(include); }
void FileUtils::pushMain(string main){ mainStrs.push_back(main); }
void FileUtils::pushFcnDec(string fcnDec){ fcnDecStrs.push_back(fcnDec); }
void FileUtils::pushFcnDef(string fcnDef){ fcnDefStrs.push_back(fcnDef); }
