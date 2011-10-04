#include <cstdio>
#include "GameOfLife.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>

using namespace std;

/*******************************
 * Function: Constructor1
 * ----------------------
 * initializes game board
 *
 * Inputs:
 *    boardsize: size of board
 *    percentLiving: percent of cells on board that will be "alive"
 */
GameOfLife::GameOfLife(int boardsize, int percentLiving) {
  rows = boardsize;
  columns = boardsize;

  srand(time(NULL));

  currentBoard = new int*[rows];
  nextBoard = new int*[rows];

  for(int row=0; row < rows; row++) {
    currentBoard[row] = new int[columns];
    nextBoard[row] = new int[columns];
    for (int col=0; col < columns; col++) {
      int randVal = rand() % 100;
      currentBoard[row][col] = (randVal < percentLiving) ? 1 : 0;
    }
  }
}


/*******************************
 * Function: Constructor2
 * ----------------------
 * initializes game board by reading in board from file
 *
 * Inputs:
 *    boardFileName: path to board file
 */
GameOfLife::GameOfLife(string boardFileName) {
  string line;
  ifstream boardFile(boardFileName.c_str());

  string garbage;
  rows = 0;
  columns = 0;

  if (boardFile.is_open())
  {
    while (!boardFile.eof())
    {
      getline(boardFile,line);
      if (line.find('#') == 0) { // Comment, ignore
        continue;
      } else if (line.find("rows:") == 0) {
        stringstream s(line);
        s >> garbage >> rows;
      } else if (line.find("columns:") == 0) {        
        stringstream s(line);
        s >> garbage >> columns;
      } else if (line.find('\n') == 0) {
        continue;
      }

      if ((rows > 0) && (columns > 0)) {
        break;
      }
    }
    currentBoard = new int*[rows];
    nextBoard = new int*[rows];
    for(int row=0; row < rows; row++) {
      currentBoard[row] = new int[columns];
      nextBoard[row] = new int[columns];
    }
    for(int row=0; row < rows; row++) {
      for(int col=0; col < columns; col++) {
        boardFile >> currentBoard[row][col];
      }
    }
    boardFile.close();
  } else {
    cout << "Unable to open file";
  }
}

/*******************************
 * Function: Destructor
 * --------------------
 * deletes all allocated memory
 */
GameOfLife::~GameOfLife(){
  for (int r=0; r<rows; r++){
    delete [] currentBoard[r];
    delete [] nextBoard[r];
  }
  delete [] currentBoard;
  delete [] nextBoard;
}

/*******************************
 * Function: runGeneration
 * -----------------------
 * runs through one 'generation' of the board. Given each cell and its
 * neighbors, we can decide whether that cell 'lives' or 'dies'.
 * Parallelization (row- or col-wise) occurs here
 */
void GameOfLife::runGeneration(){
  int **localCurrentBoard = currentBoard;
  int **localNextBoard = nextBoard;
  int r,c;
  int neighborcount;
  {
    for (r=0; r<rows; r++){
      for (c=0; c<columns; c++){
        neighborcount = numNeighbors(r,c);

        // if cell is live
        if (localCurrentBoard[r][c]){
          // if <= 1 neighbor, cell dies from loneliness
          if (neighborcount <= 1){
            localNextBoard[r][c] = 0;
          }
          // if >= 4 neighbors, cell dies from overpopulation
          else if (neighborcount >= 4){
            localNextBoard[r][c] = 0;
          } else {
            localNextBoard[r][c] = 1;
          }
        // if cell is dead
        } else {
          if (neighborcount == 3){
            localNextBoard[r][c] = 1;
          } else {
            localNextBoard[r][c] = 0;
          }
        }
      }
    }
  }
  int **tmpCurrent = localCurrentBoard;
  currentBoard = localNextBoard;
  nextBoard = tmpCurrent;
}

/*******************************
 * Function: numNeighbors
 * ----------------------
 * Gets the number of neighbors for the cell at (row,col)
 *
 * Inputs:
 *    row: current row
 *    col: current column
 *
 * Returns:
 *    number of neighbors
 */
int GameOfLife::numNeighbors(int row, int col){
  int neighborcount = 0;
  int myrow, mycol;
  for (int r=row-1; r<=row+1; r++){
    for (int c=col-1; c<=col+1; c++){
      myrow = myMod(r,rows);
      mycol = myMod(c,columns);
      if (currentBoard[myrow][mycol]){
        neighborcount++;
      }
    }
  }
  // we counted the cell at row,col above, so now we account for it
  if (currentBoard[row][col]){
    neighborcount--;
  }
  return neighborcount;
}


/*******************************
 * Function: myMod
 * ---------------
 * modulus (%) in C++ does not work for negative numbers (like it does in
 * python), so we need to make a pseudo-mod function that is adapted
 * specifically for our purposes
 *
 * For example, myMod(-1, modVal) = modVal-1, as expected. This gives us the torus
 * shape that we need.
 *
 * inputs:
 *    val: value that we'll be modding
 *    modVal: value that we mod against
 *
 * Returns: correctly modded value
 */
int GameOfLife::myMod(int val, int modVal){
  if (val < 0) { val += modVal; }
  return val % modVal;
}

// prints out board in a nice, human readable format
void GameOfLife::printBoard(){
  for (int r=0; r<rows; r++){
    for (int c=0; c<columns; c++){
      printf("%c ", currentBoard[r][c] ? '1' : '.');
    }
    printf("\n");
  }
  printf("\n");
}

// accessor functions
int GameOfLife::getRows(){ return rows; }
int GameOfLife::getCols(){ return columns; }

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <boost/program_options.hpp>

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
