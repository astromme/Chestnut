#ifndef _GAME_OF_LIFE_H #define _GAME_OF_LIFE_H

#include <string>

using std::string;

class GameOfLife {
  public:
    GameOfLife(int boardsize=64, int percentLiving=30); // Initializes random board
    GameOfLife(string fname);
    ~GameOfLife();
    void runGeneration();
    void printBoard();

    // accessor functions
    int getRows();
    int getCols();

  private:
    int **currentBoard;
    int **nextBoard;
    int rows, columns;

    // functions
    int numNeighbors(int row, int col);
    int myMod(int val, int modVal);
};

#endif
