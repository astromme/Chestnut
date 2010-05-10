// Currently not used, but may be expanded in the future

#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <boost/algorithm/string/replace.hpp>
#include <boost/regex.hpp>
#include "ParseUtils.h"

using namespace std;
using boost::algorithm::replace_all;

/********************************
 * Function: makeTimer
 * -------------------
 * Associated with syntax in Chestnut
 *      timer <name>
 *
 * Generates code to declare timers
 *
 * Inputs:
 *    timer: name of timer
 */
void ParseUtils::makeTimer(string timer)
{
  if (symtab.addEntry(timer, VARIABLE_TIMER, "")){
    obj_names objnames = get_obj_names(timer);
    string start = objnames.timerstart;
    string stop = objnames.timerstop;
  
    // need to include sys/time.h and ParseLibs
    string timerinclude;
    timerinclude += add_include("<sys/time.h>");
    timerinclude += add_include("\"ParseLibs.h\"");
    cudafile->pushInclude(timerinclude);
  
    string cuda_outstr;
    cuda_outstr += prep_str("struct timeval " +
      start + ", " + stop + "; // instantiate timer");
    cuda_outstr += "\n";
    cudafile->pushMain(cuda_outstr);
  }
}

/********************************
 * Function: makeTimerStart
 * ------------------------
 * Associated with syntax in Chestnut
 *      <name> start
 *
 * Generates code to start a timer
 *
 * Inputs:
 *    timer: name of timer
 */
void ParseUtils::makeTimerStart(string timer)
{
  obj_names objnames = get_obj_names(timer);
  string start = objnames.timerstart;
 
  string cuda_outstr;
  cuda_outstr += prep_str("gettimeofday(&" + start + ", 0); // start timer");
  cuda_outstr += "\n";
  cudafile->pushMain(cuda_outstr);
}

/********************************
 * Function: makeTimerStop
 * -----------------------
 * Associated with syntax in Chestnut
 *      <name> stop
 *
 * Generates code to stop a timer
 *
 * Inputs:
 *    timer: name of timer
 */
void ParseUtils::makeTimerStop(string timer)
{
  obj_names objnames = get_obj_names(timer);
  string stop = objnames.timerstop;
 
  string cuda_outstr;
  cuda_outstr += prep_str("gettimeofday(&" + stop + ", 0); // stop timer");
  cuda_outstr += "\n";
  cudafile->pushMain(cuda_outstr);
}