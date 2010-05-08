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
 * Function: makeForeach
 * ---------------------
 * Associated with the syntax in Chestnut
 *      <type> <name> <rows> <cols> foreach ( <expr> );
 *
 * Generates code to fill data with given expression.
 *
 * FIXME: more specific
 * Expression can consist of a bunch of stuff like rows, cols, maxrows, etc
 * that we have to deal with
 *
 * Inputs:
 *    object: name of variable to be initialized
 *    type: type of var (int, float, etc)
 *    datarows: number of rows to initialize
 *    datacols: number of columns to initialize
 *    expr: expression used to populate variable
 */
void ParseUtils::makeForeach(string object, string type, string datarows, string datacols, string expr){
  // define names of vars in prog
  obj_names objnames = get_obj_names(object);
  string host = objnames.host;
  string dev = objnames.dev;
  string rows = objnames.rows;
  string cols = objnames.cols;
  expr = processForeachExpr(expr, objnames);

  /* We want to check if the keywords "row" or "col" are in the given
   * expression. If they are, we need to use the expanded for-loop syntax to
   * populate the data on the CPU and then copy over to GPU. This is slow.
   *
   * If those keywords are not in the expression, then we know all of the
   * variable values without the need of a for-loop. We can thus optimize this
   * by using thrust::fill.
   */
  
  // right hand side of expression
  string expr_rhs = expr.substr(expr.find("=")+1,expr.length()); 
  boost::regex row_or_col("\\<row\\>|\\<col\\>");

  bool optimize_fill = true;
  if (boost::regex_search (expr_rhs, row_or_col)){
    optimize_fill = false;
  }

  // check if rand exists, and, if so, whether we have seeded the random
  // number generator
  boost::regex rand("\\<rand\\>");
  if (boost::regex_search (expr_rhs, rand)){
    optimize_fill = false;

    if (!rand_included){
      string randincl;
      randincl += add_include("<cstdlib>");
      randincl += add_include("<ctime>");
      cudafile->pushInclude(randincl);

      string randstr = prep_str("srand( time(NULL) ); // seed random number generator");
      randstr += "\n";
      cudafile->pushMain(randstr);

      rand_included = true;
    }
  }
  
  string cuda_outstr;
  if (symtab.addEntry(object, VARIABLE_VECTOR, type)){
    cuda_outstr += prep_str("int " + rows + " = " + datarows + ";");
    cuda_outstr += prep_str("int " + cols + " = " + datacols + ";"); 

    // if we decided to optimize, we don't want to waste time allocating space
    // on the host_vector
    string host_declaration = "host_vector<" + type + "> " + host;
    if (!optimize_fill){
      host_declaration += "(" + rows + "*" + cols + ")";
    }
    host_declaration += "; // Memory on host (cpu) side";
    cuda_outstr += prep_str(host_declaration);

    // need to allocate space on device regardless of optimization
    cuda_outstr += prep_str("device_vector<" + type + "> " + dev + 
        "(" + rows + "*" + cols + "); // Memory on device (gpu) side");
    cuda_outstr += "\n";
  } // TODO: else we need to delete some memory? i.e. host has already been allocated

  cuda_outstr += prep_str("// populate data");
  // if we can optimize using fill, do so
  if (optimize_fill){
    cuda_outstr += prep_str("thrust::fill(" + dev + ".begin(), " +
        dev + ".end(), " + expr_rhs + ");");
  } else {
    cuda_outstr += prep_str("for (int row=0; row<" + rows + "; row++){"); indent++;
    cuda_outstr += prep_str("for (int col=0; col<" + cols + "; col++){"); indent++;
    cuda_outstr += prep_str(expr + ";"); indent--;
    cuda_outstr += prep_str("}"); indent--;
    cuda_outstr += prep_str("}");
    cuda_outstr += prep_str(dev + " = " + host + "; // copy host data to GPU");
  }
  cuda_outstr += "\n";
  cudafile->pushMain(cuda_outstr);
}

/********************************
 * Function: processForeachExpr
 * ----------------------------
 * a `foreach` expression can consist of more than just simple arithmetic
 * expressions. We have a set of reserved words which can be used at a high
 * level to indicate lower-level changes. They are as follows.
 *
 * ==> maxrows: total number of rows in data block
 * ==> maxcols: total number of columns in data block
 * ==> row: current row index
 * ==> col: current column index
 * ==> value: current index of array, 
 *    i.e. arr[row][col], represented as arr[row*maxcols+col]
 * ==> rand: a random real number between 0 and RAND_MAX
 *
 * Thus, as we see these reserved words, we must replace them in the code with
 * the correct names, as given by the object name with which the foreach
 * statement is associated. Also, row and col will be replaced by 'r' and 'c',
 * respectively, which is object-name independent.
 */
string ParseUtils::processForeachExpr(string expr, const obj_names &objnames){
  string decimalcast = "(float)";
  
  // since the right hand side may have something like "row/maxrows" we want
  // that to be a float, not an int, so let's cast variables as floats
  // TODO: make sure this doesn't break anything
  replace_all(expr, "row", decimalcast + "row");
  replace_all(expr, "max" + decimalcast + "rows", decimalcast + objnames.rows);

  replace_all(expr, "col", decimalcast + "col");
  replace_all(expr, "max" + decimalcast + "cols", decimalcast + objnames.cols);

  replace_all(expr, "rand", decimalcast + "rand()");

  replace_all(expr, "value", objnames.host + "[row*" + objnames.cols + "+col]");
  replace_all(expr, "=", " = ");
  //replace_all(expr, "=", ".push_back(");
  // TODO: implement rand

  return expr;
}