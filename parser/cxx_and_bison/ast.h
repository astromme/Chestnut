#ifndef AST_H_
#define AST_H_

class Expression {
public:
  virtual ~Expression () {}

  //  It's necessary because we need to clone objects without
  // knowing the exact type.
  virtual Expression *clone () = 0;

  // The value represented by the expression
  virtual int value () = 0;
};

// For addictive expressions
class Plus : public Expression {
  Expression *m_left, *m_right;

public:
   
  Plus (Expression *left, Expression *right): m_left (left), m_right (right) {}

  // Copy constructor
  Plus (const Plus &other) {
    m_left = other.m_left->clone ();
    m_right = other.m_right->clone ();
  }
  
  virtual ~Plus () 
  {
    delete m_left;
    delete m_right;
  }

  Plus &operator = (const Plus &other) {
    if (&other != this) {
      delete m_left;
      delete m_right;
      
      m_left = other.m_left->clone ();
      m_right = other.m_right->clone ();
    }
  }
  

  virtual Expression *clone () { return new Plus (*this); }
  
  virtual int value () { return m_left->value () + m_right->value (); }
  
};

// For multiplicative expressions
class Times : public Expression {
  Expression *m_left, *m_right;

public:
   
  Times (Expression *left, Expression *right): m_left (left), m_right (right) {}

  // Copy constructor
  Times (const Times &other) {
    m_left = other.m_left->clone ();
    m_right = other.m_right->clone ();
  }

  virtual ~Times () 
  {
    delete m_left;
    delete m_right;
  }

  Times &operator = (const Times &other) {
    if (&other != this) {
      delete m_left;
      delete m_right;
      
      m_left = other.m_left->clone ();
      m_right = other.m_right->clone ();
    }
  }
  

  virtual Expression *clone () { return new Times (*this); }
  
  virtual int value () { return m_left->value () * m_right->value (); }
  
};

// For numbers
class Number : public Expression {
  int m_val;

public:
   
  Number (int val): m_val (val) {}

  // Copy constructor
  Number (const Number &other) { m_val = other.m_val; }

  Number &operator = (const Number &other) {
    if (&other != this)
      m_val = other.m_val;
  }

  virtual Expression *clone () { return new Number (*this); }

  virtual int value () { return m_val; }
};

// For identifiers
class Ident : public Expression {
  int *m_val;

public:
   
  Ident (int *val): m_val (val) {}

  // Copy constructor
  Ident (const Ident &other) { m_val = other.m_val; }

  Ident &operator = (const Ident &other) {
    if (&other != this)
      m_val = other.m_val;
  }

  virtual Expression *clone () { return new Ident (*this); }
  

  virtual int value () { return *m_val; }
};

#endif






