#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <boost/program_options.hpp>
#include "GameOfLife.h"

//#define DEBUG // uncomment this line to print board after every generation

using namespace std;
namespace po = boost::program_options;

/*******************************
 * Function: main
 * --------------
 * reads in arguments and runs generations of those boards according to said
 * arguments.
 */
int main(int argc, char* argv[]){
  GameOfLife *g;
  bool usingGameboard = false;
  bool printGameboard = false;
  string gameboard = "";
  int iterations = 100000;
  int threads = 2;
  int boardsize = 64;

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "produce help message")
      ("iterations,i", po::value<int>(), "set number of iterations (default: 100,002) ")
      ("threads,t", po::value<int>(), "set number of threads (default: 2)")
      ("boardsize,b", po::value<int>(), "set board dimensions to N, results in a NxN array (default: 64x64)")
      ("gameboard,g", po::value<string>(), "set the gameboard file (default: random board)")
      ("printboard,p", "print the game board? (default: false)")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
      cout << desc << "\n";
      return 1;
  }

  if (vm.count("iterations")) {
    iterations = vm["iterations"].as<int>();
  }
  if (vm.count("threads")) {
    threads = vm["threads"].as<int>();
  }
  if (vm.count("boardsize")) {
    boardsize = vm["boardsize"].as<int>();
  }
  if (vm.count("gameboard")) {
    usingGameboard = true;
    gameboard = vm["gameboard"].as<string>();
  }
  if (vm.count("printboard")) {
    printGameboard = true;
  }

  if (usingGameboard) {
    g = new GameOfLife(gameboard);
  } else {
    g = new GameOfLife(boardsize);
  }

  if (printGameboard)
      g->printBoard();

  //while (true){
  struct timeval start_t, stop_t;
  double total_time;

  gettimeofday(&start_t, 0);  // start timer for measured portion of code

  cout << "Board Size is " << g->getRows() << " x " << g->getCols() << endl;
  cout << "Running " << iterations << " iterations with " << threads <<
    " thread" << (threads != 1 ? "s" : "") << "..." << endl;

  for (int i=0; i<iterations; i++){
    g->runGeneration();
  }

  gettimeofday(&stop_t, 0);  // stop timer for measured portion of code

  total_time = (stop_t.tv_sec + stop_t.tv_usec*0.000001)
    - (start_t.tv_sec + start_t.tv_usec*0.000001);

  // don't want to include printf in timed code
  if (printGameboard)
      g->printBoard();

  printf("total time is: %g\n", total_time);

  delete g;
  return 0;
}
