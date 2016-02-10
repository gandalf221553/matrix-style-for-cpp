/*
A simple matrix class
c++ code
Author: Jos de Jong, Nov 2007. Updated March 2010
Author2: Biancardi Francesco, Feb 2016
With this class you can:
  - create a 2D matrix with custom size
  - get/set the cell values
  - use operators +, -, *, /
  - use functions Ones(), Zeros(), Diag(), Det(), Inv(), Size()
  - print the content of the matrix

Usage:
  write in your code #include "matrix.cpp"
  you can create a matrix by:
    Matrix A;
    Matrix A = Matrix(rows, cols);
    Matrix A = B;

  you can get and set matrix elements by:
    A(2,3) = 5.6;    // set an element of Matix A
    value = A(3,1);   // get an element of Matrix A
    value = A.get(3,1); // get an element of a constant Matrix A
    A = B;        // copy content of Matrix B to Matrix A

  you can apply operations with matrices and doubles:
    A = B + C;
    A = B - C;
    A = -B;
    A = B * C;
    A = B / C;

  the following functions are available:
    A = Ones(rows, cols);
    A = Zeros(rows, cols);
    A = Diag(n);
    A = Diag(B);
    d = Det(A);
    A = Inv(B);
    cols = A.GetCols();
    rows = A.GetRows();
    cols = Size(A, 1);
    rows = Size(A, 2);

  you can quick-print the content of a matrix in the console with:
    A.Print();
	
	
	
	Function Added by Biancardi:
	A=Trans(B);					//transpose the B matrix
	A=EraseARow(B,5); 			// erase row number 5 in B matrix
	A=EraseACol(B,3); 			// erase column number 3 in B matrix
	C=UnionCols(A,B); 			//put matrix A and B in column (A above B)
	C=UnionRows(A,B); 			//put matrix A next to matrix B
	V=ExtractARowOrACol(A,4,0);             //extract a row or a col (the other terms nedd to be zero) by A matrix ExtractARowOrACol(A,whatrow,woatcol)
	d=ConfrontaMatrix(A,B);		//if A is equal (equal in size and in single terms) to B write 1 otherwise 0
	A=EraseTwinRows(B);			//if two rows are equal, function delete one of them
	d=Trace(A);					//calculate the trace of the matrix A
	d=Min(A);					//minimum factor in matrix A
	d=Max(A);					//maximum factor in matric A
	C=UnionRowsAtipiche(A,B);	//put in column A and B, but if rows are different, create a 
	A=Linspace(a,b,c);			//A is a vector long 'c' factor, every factor is equidistant from a to b
	B=Linspace(a,b);			//B is a vector long 100 factor, every factor is equidistant from a to b
	B=find(A,a);				//B is a matrix (n,2) composed by the position where the factor of A matrix is equal to 'a'
*/

#include <cstdlib>
#include <cstdio>
#include <math.h>
#define PAUSE {printf("Press \"Enter\" to continue\n"); fflush(stdin); getchar(); fflush(stdin);}

// Declarations
class Matrix;
double Det(const Matrix& a);
Matrix Diag(const int n);
Matrix Diag(const Matrix& v);
Matrix Inv(const Matrix& a);
Matrix Ones(const int rows, const int cols);
int Size(const Matrix& a, const int i);
Matrix Zeros(const int rows, const int cols);
Matrix Trans(const Matrix& a);
Matrix EraseARow(const Matrix& a,const int n);
Matrix EraseACol(const Matrix& a,const int n);
Matrix UnionCols(const Matrix& a,const Matrix& b);
Matrix UnionRows(const Matrix& a,const Matrix& b);
Matrix ExtractARowOrACol(const Matrix& a, const int riganum, const int colonnanum);
int ConfrontaMatrix(const Matrix& a, const Matrix& b);
Matrix EraseTwinRows(const Matrix& a);
int Trace(const Matrix& a);
double Min(const Matrix& a);
double Max(const Matrix& a);
double Min(const double a, const double b);
double Max(const double a, const double b);
Matrix UnionRowsAtipiche(const Matrix& a,const Matrix& b);
Matrix Linspace(const double a,const double b, const double c);
Matrix Linspace(const double a,const double b);
Matrix find(const Matrix& a,const double b);


/*
 * a simple exception class
 * you can create an exeption by entering
 *   throw Exception("...Error description...");
 * and get the error message from the data msg for displaying:
 *   err.msg
 */
class Exception
{
public:
  const char* msg;
  Exception(const char* arg)
   : msg(arg)
  {
  }
};



class Matrix
{
public:
  // constructor
  Matrix()
  {
    //printf("Executing constructor Matrix() ...\n");
    // create a Matrix object without content
    p = NULL;
    rows = 0;
    cols = 0;
  }

  // constructor
  Matrix(const int row_count, const int column_count)
  {
    // create a Matrix object with given number of rows and columns
    p = NULL;

    if (row_count > 0 && column_count > 0)
    {
      rows = row_count;
      cols = column_count;

      p = new double*[rows];
      for (int r = 0; r < rows; r++)
      {
        p[r] = new double[cols];

        // initially fill in zeros for all values in the matrix;
        for (int c = 0; c < cols; c++)
        {
          p[r][c] = 0;
        }
      }
    }
  }

  // assignment operator
  Matrix(const Matrix& a)
  {
    rows = a.rows;
    cols = a.cols;
    p = new double*[a.rows];
    for (int r = 0; r < a.rows; r++)
    {
      p[r] = new double[a.cols];

      // copy the values from the matrix a
      for (int c = 0; c < a.cols; c++)
      {
        p[r][c] = a.p[r][c];
      }
    }
  }

  // index operator. You can use this class like myMatrix(col, row)
  // the indexes are one-based, not zero based.
  double& operator()(const int r, const int c)
  {
    if (p != NULL && r > 0 && r <= rows && c > 0 && c <= cols)
    {
      return p[r-1][c-1];
    }
    else
    {
      throw Exception("Subscript out of range");
    }
  }

  // index operator. You can use this class like myMatrix.get(col, row)
  // the indexes are one-based, not zero based.
  // use this function get if you want to read from a const Matrix
  double get(const int r, const int c) const
  {
    if (p != NULL && r > 0 && r <= rows && c > 0 && c <= cols)
    {
      return p[r-1][c-1];
    }
    else
    {
      throw Exception("Subscript out of range");
    }
  }

  // assignment operator
  Matrix& operator= (const Matrix& a)
  {
    rows = a.rows;
    cols = a.cols;
    p = new double*[a.rows];
    for (int r = 0; r < a.rows; r++)
    {
      p[r] = new double[a.cols];

      // copy the values from the matrix a
      for (int c = 0; c < a.cols; c++)
      {
        p[r][c] = a.p[r][c];
      }
    }
    return *this;
  }

  // add a double value (elements wise)
  Matrix& Add(const double v)
  {
    for (int r = 0; r < rows; r++)
    {
      for (int c = 0; c < cols; c++)
      {
        p[r][c] += v;
      }
    }
     return *this;
  }

  // subtract a double value (elements wise)
  Matrix& Subtract(const double v)
  {
    return Add(-v);
  }

  // multiply a double value (elements wise)
  Matrix& Multiply(const double v)
  {
    for (int r = 0; r < rows; r++)
    {
      for (int c = 0; c < cols; c++)
      {
        p[r][c] *= v;
      }
    }
     return *this;
  }

  // divide a double value (elements wise)
  Matrix& Divide(const double v)
  {
     return Multiply(1/v);
  }

  // addition of Matrix with Matrix
  friend Matrix operator+(const Matrix& a, const Matrix& b)
  {
    // check if the dimensions match
    if (a.rows == b.rows && a.cols == b.cols)
    {
      Matrix res(a.rows, a.cols);

      for (int r = 0; r < a.rows; r++)
      {
        for (int c = 0; c < a.cols; c++)
        {
          res.p[r][c] = a.p[r][c] + b.p[r][c];
        }
      }
      return res;
    }
    else
    {
      // give an error
      throw Exception("Dimensions does not match");
    }

    // return an empty matrix (this never happens but just for safety)
    return Matrix();
  }

  // addition of Matrix with double
  friend Matrix operator+ (const Matrix& a, const double b)
  {
    Matrix res = a;
    res.Add(b);
    return res;
  }
  // addition of double with Matrix
  friend Matrix operator+ (const double b, const Matrix& a)
  {
    Matrix res = a;
    res.Add(b);
    return res;
  }

  // subtraction of Matrix with Matrix
  friend Matrix operator- (const Matrix& a, const Matrix& b)
  {
    // check if the dimensions match
    if (a.rows == b.rows && a.cols == b.cols)
    {
      Matrix res(a.rows, a.cols);

      for (int r = 0; r < a.rows; r++)
      {
        for (int c = 0; c < a.cols; c++)
        {
          res.p[r][c] = a.p[r][c] - b.p[r][c];
        }
      }
      return res;
    }
    else
    {
      // give an error
      throw Exception("Dimensions does not match");
    }

    // return an empty matrix (this never happens but just for safety)
    return Matrix();
  }

  // subtraction of Matrix with double
  friend Matrix operator- (const Matrix& a, const double b)
  {
    Matrix res = a;
    res.Subtract(b);
    return res;
  }
  // subtraction of double with Matrix
  friend Matrix operator- (const double b, const Matrix& a)
  {
    Matrix res = -a;
    res.Add(b);
    return res;
  }

  // operator unary minus
  friend Matrix operator- (const Matrix& a)
  {
    Matrix res(a.rows, a.cols);

    for (int r = 0; r < a.rows; r++)
    {
      for (int c = 0; c < a.cols; c++)
      {
        res.p[r][c] = -a.p[r][c];
      }
    }

    return res;
  }

  // operator multiplication
  friend Matrix operator* (const Matrix& a, const Matrix& b)
  {
    // check if the dimensions match
    if (a.cols == b.rows)
    {
      Matrix res(a.rows, b.cols);

      for (int r = 0; r < a.rows; r++)
      {
        for (int c_res = 0; c_res < b.cols; c_res++)
        {
          for (int c = 0; c < a.cols; c++)
          {
            res.p[r][c_res] += a.p[r][c] * b.p[c][c_res];
          }
        }
      }
      return res;
    }
    else
    {
      // give an error
      throw Exception("Dimensions does not match");
    }

    // return an empty matrix (this never happens but just for safety)
    return Matrix();
  }

  // multiplication of Matrix with double
  friend Matrix operator* (const Matrix& a, const double b)
  {
    Matrix res = a;
    res.Multiply(b);
    return res;
  }
  // multiplication of double with Matrix
  friend Matrix operator* (const double b, const Matrix& a)
  {
    Matrix res = a;
    res.Multiply(b);
    return res;
  }

  // division of Matrix with Matrix
  friend Matrix operator/ (const Matrix& a, const Matrix& b)
  {
    // check if the dimensions match: must be square and equal sizes
    if (a.rows == a.cols && a.rows == a.cols && b.rows == b.cols)
    {
      Matrix res(a.rows, a.cols);

      res = a * Inv(b);

      return res;
    }
    else
    {
      // give an error
      throw Exception("Dimensions does not match");
    }

    // return an empty matrix (this never happens but just for safety)
    return Matrix();
  }

  // division of Matrix with double
  friend Matrix operator/ (const Matrix& a, const double b)
  {
    Matrix res = a;
    res.Divide(b);
    return res;
  }

  // division of double with Matrix
  friend Matrix operator/ (const double b, const Matrix& a)
  {
    Matrix b_matrix(1, 1);
    b_matrix(1,1) = b;

    Matrix res = b_matrix / a;
    return res;
  }

  /**
   * returns the minor from the given matrix where
   * the selected row and column are removed
   */
  Matrix Minor(const int row, const int col) const
  {
    Matrix res;
    if (row > 0 && row <= rows && col > 0 && col <= cols)
    {
      res = Matrix(rows - 1, cols - 1);

      // copy the content of the matrix to the minor, except the selected
      for (int r = 1; r <= (rows - (row >= rows)); r++)
      {
        for (int c = 1; c <= (cols - (col >= cols)); c++)
        {
          res(r - (r > row), c - (c > col)) = p[r-1][c-1];
        }
      }
    }
    else
    {
      throw Exception("Index for minor out of range");
    }

    return res;
  }

  /*
   * returns the size of the i-th dimension of the matrix.
   * i.e. for i=1 the function returns the number of rows,
   * and for i=2 the function returns the number of columns
   * else the function returns 0
   */
  int Size(const int i) const
  {
    if (i == 1)
    {
      return rows;
    }
    else if (i == 2)
    {
      return cols;
    }
    return 0;
  }

  // returns the number of rows
  int GetRows() const
  {
    return rows;
  }

  // returns the number of columns
  int GetCols() const
  {
    return cols;
  }

  // print the contents of the matrix
  void Print() const
  {
    if (p != NULL)
    {
      //printf("[");
      for (int r = 0; r < rows; r++)
      {
        if (r > 0)
        {
          printf(" ");
        }
        for (int c = 0; c < cols-1; c++)
        {
          printf("%.2f", p[r][c]);
  		  printf(", ");
        }
        if (r < rows-1)
        {
          printf("%.2f", p[r][cols-1]);
		  //printf(";");
		  printf("\n");
        }
        else
        {
          printf("%.2f", p[r][cols-1]);
		  //printf("]");
		  printf("\n");
        }
      }
    }
    else
    {
      // matrix is empty
      printf("[ ]\n");
    }
  }

public:
  // destructor
  ~Matrix()
  {
    // clean up allocated memory
    for (int r = 0; r < rows; r++)
    {
      delete p[r];
    }
    delete p;
    p = NULL;
  }

private:
  int rows;
  int cols;
  double** p;     // pointer to a matrix with doubles
};

/*
 * i.e. for i=1 the function returns the number of rows,
 * and for i=2 the function returns the number of columns
 * else the function returns 0
 */
int Size(const Matrix& a, const int i)
{
  return a.Size(i);
}


/**
 * returns a matrix with size cols x rows with ones as values
 */
Matrix Ones(const int rows, const int cols)
{
  Matrix res = Matrix(rows, cols);

  for (int r = 1; r <= rows; r++)
  {
    for (int c = 1; c <= cols; c++)
    {
      res(r, c) = 1;
    }
  }
  return res;
}

/**
 * returns a matrix with size cols x rows with zeros as values
 */
Matrix Zeros(const int rows, const int cols)
{
  return Matrix(rows, cols);
}


/**
 * returns a diagonal matrix with size n x n with ones at the diagonal
 * @param  v a vector
 * @return a diagonal matrix with ones on the diagonal
 */
Matrix Diag(const int n)
{
  Matrix res = Matrix(n, n);
  for (int i = 1; i <= n; i++)
  {
    res(i, i) = 1;
  }
  return res;
}

/**
 * returns a diagonal matrix
 * @param  v a vector
 * @return a diagonal matrix with the given vector v on the diagonal
 */
Matrix Diag(const Matrix& v)
{
  Matrix res;
  if (v.GetCols() == 1)
  {
    // the given matrix is a vector n x 1
    int rows = v.GetRows();
    res = Matrix(rows, rows);

    // copy the values of the vector to the matrix
    for (int r=1; r <= rows; r++)
    {
      res(r, r) = v.get(r, 1);
    }
  }
  else if (v.GetRows() == 1)
  {
    // the given matrix is a vector 1 x n
    int cols = v.GetCols();
    res = Matrix(cols, cols);

    // copy the values of the vector to the matrix
    for (int c=1; c <= cols; c++)
    {
      res(c, c) = v.get(1, c);
    }
  }
  else
  {
    throw Exception("Parameter for diag must be a vector");
  }
  return res;
}

/*
 * returns the determinant of Matrix a
 */
double Det(const Matrix& a)
{
  double d = 0;    // value of the determinant
  int rows = a.GetRows();
  int cols = a.GetRows();

  if (rows == cols)
  {
    // this is a square matrix
    if (rows == 1)
    {
      // this is a 1 x 1 matrix
      d = a.get(1, 1);
    }
    else if (rows == 2)
    {
      // this is a 2 x 2 matrix
      // the determinant of [a11,a12;a21,a22] is det = a11*a22-a21*a12
      d = a.get(1, 1) * a.get(2, 2) - a.get(2, 1) * a.get(1, 2);
    }
    else
    {
      // this is a matrix of 3 x 3 or larger
      for (int c = 1; c <= cols; c++)
      {
        Matrix M = a.Minor(1, c);
        //d += pow(-1, 1+c) * a(1, c) * Det(M);
        d += (c%2 + c%2 - 1) * a.get(1, c) * Det(M); // faster than with pow()
      }
    }
  }
  else
  {
    throw Exception("Matrix must be square");
  }
  return d;
}

// swap two values
void Swap(double& a, double& b)
{
  double temp = a;
  a = b;
  b = temp;
}

/*
 * returns the inverse of Matrix a
 */
Matrix Inv(const Matrix& a)
{
  Matrix res;
  double d = 0;    // value of the determinant
  int rows = a.GetRows();
  int cols = a.GetRows();

  d = Det(a);
  if (rows == cols && d != 0)
  {
    // this is a square matrix
    if (rows == 1)
    {
      // this is a 1 x 1 matrix
      res = Matrix(rows, cols);
      res(1, 1) = 1 / a.get(1, 1);
    }
    else if (rows == 2)
    {
      // this is a 2 x 2 matrix
      res = Matrix(rows, cols);
      res(1, 1) = a.get(2, 2);
      res(1, 2) = -a.get(1, 2);
      res(2, 1) = -a.get(2, 1);
      res(2, 2) = a.get(1, 1);
      res = (1/d) * res;
    }
    else
    {
      // this is a matrix of 3 x 3 or larger
      // calculate inverse using gauss-jordan elimination
      //   http://mathworld.wolfram.com/MatrixInverse.html
      //   http://math.uww.edu/~mcfarlat/inverse.htm
      res = Diag(rows);   // a diagonal matrix with ones at the diagonal
      Matrix ai = a;    // make a copy of Matrix a

      for (int c = 1; c <= cols; c++)
      {
        // element (c, c) should be non zero. if not, swap content
        // of lower rows
        int r;
        for (r = c; r <= rows && ai(r, c) == 0; r++)
        {
        }
        if (r != c)
        {
          // swap rows
          for (int s = 1; s <= cols; s++)
          {
            Swap(ai(c, s), ai(r, s));
            Swap(res(c, s), res(r, s));
          }
        }

        // eliminate non-zero values on the other rows at column c
        for (int r = 1; r <= rows; r++)
        {
          if(r != c)
          {
            // eleminate value at column c and row r
            if (ai(r, c) != 0)
            {
              double f = - ai(r, c) / ai(c, c);

              // add (f * row c) to row r to eleminate the value
              // at column c
              for (int s = 1; s <= cols; s++)
              {
                ai(r, s) += f * ai(c, s);
                res(r, s) += f * res(c, s);
              }
            }
          }
          else
          {
            // make value at (c, c) one,
            // divide each value on row r with the value at ai(c,c)
            double f = ai(c, c);
            for (int s = 1; s <= cols; s++)
            {
              ai(r, s) /= f;
              res(r, s) /= f;
            }
          }
        }
      }
    }
  }
  else
  {
    if (rows == cols)
    {
      throw Exception("Matrix must be square");
    }
    else
    {
      throw Exception("Determinant of matrix is zero");
    }
  }
  return res;
}

/* chiusa da 793 a 950
int main(int argc, char *argv[])
{
  // below some demonstration of the usage of the Matrix class
  try
  {
    // create an empty matrix of 3x3 (will initially contain zeros)
    int cols = 3;
    int rows = 3;
    Matrix A = Matrix(cols, rows);

    // fill in some values in matrix a
    int count = 0;
    for (int r = 1; r <= rows; r++)
    {
      for (int c = 1; c <= cols; c++)
      {
        count ++;
        A(r, c) = count;
      }
    }

    // adjust a value in the matrix (indexes are one-based)
    A(2,1) = 1.23;

    // read a value from the matrix (indexes are one-based)
    double centervalue = A(2,2);
    printf("centervalue = %f \n", centervalue);
    printf("\n");

    // print the whole matrix
    printf("A = \n");
    A.Print();
    printf("\n");

    Matrix B = Ones(rows, cols) + Diag(rows);
    printf("B = \n");
    B.Print();
    printf("\n");

    Matrix B2 = Matrix(rows, 1);    // a vector
    count = 1;
    for (int r = 1; r <= rows; r++)
    {
      count ++;
      B2(r, 1) = count * 2;
    }
    printf("B2 = \n");
    B2.Print();
    printf("\n");

    Matrix C;
    C = A + B;
    printf("A + B = \n");
    C.Print();
    printf("\n");

    C = A - B;
    printf("A - B = \n");
    C.Print();
    printf("\n");

    C = A * B2;
    printf("A * B2 = \n");
    C.Print();
    printf("\n");

    // create a diagonal matrix
    Matrix E = Diag(B2);
    printf("E = \n");
    E.Print();
    printf("\n");

    // calculate determinant
    Matrix D = Matrix(2, 2);
    D(1,1) = 2;
    D(1,2) = 4;
    D(2,1) = 1;
    D(2,2) = -2;
    printf("D = \n");
    D.Print();
    printf("Det(D) = %f\n\n", Det(D));

    printf("A = \n");
    A.Print();
    printf("\n");
    printf("Det(A) = %f\n\n", Det(A));

    Matrix F;
    F = 3 - A ;
    printf("3 - A = \n");
    F.Print();
    printf("\n");

    // test inverse
    Matrix G = Matrix(2, 2);
    G(1, 1) = 1;
    G(1, 2) = 2;
    G(2, 1) = 3;
    G(2, 2) = 4;
    printf("G = \n");
    G.Print();
    printf("\n");
    Matrix G_inv = Inv(G);
    printf("Inv(G) = \n");
    G_inv.Print();
    printf("\n");

    Matrix A_inv = Inv(A);
    printf("Inv(A) = \n");
    A_inv.Print();
    printf("\n");

    Matrix A_A_inv = A * Inv(A);
    printf("A * Inv(A) = \n");
    A_A_inv.Print();
    printf("\n");

    Matrix B_A = B / A;
    printf("B / A = \n");
    B_A.Print();
    printf("\n");

    Matrix A_3 = A / 3;
    printf("A / 3 = \n");
    A_3.Print();
    printf("\n");

    rows = 2;
    cols = 5;
    Matrix H = Matrix(rows, cols);
    for (int r = 1; r <= rows; r++)
    {
      for (int c = 1; c <= cols; c++)
      {
        count ++;
        H(r, c) = count;
      }
    }
    printf("H = \n");
    H.Print();
    printf("\n");
  }
  catch (Exception err)
  {
    printf("Error: %s\n", err.msg);
  }
  catch (...)
  {
    printf("An error occured...\n");
  }

  printf("\n");
  PAUSE;

  return EXIT_SUCCESS;
}
/*chiusa da 793 a 950

//////////////////////////////////////////////////////////////////////////
/**
 * returns a matrix senza le righe doppie (se le righe sono doppie le cancella)
 */ 
Matrix Trans(const Matrix& a)
{
	int rows = a.GetRows();
	int cols = a.GetCols();

	Matrix Trasp = Zeros(cols, rows);
if (rows>1 && cols>1)
	{
	for (int r = 1; r <= rows; r++)
	  {
	for (int c = 1; c <= cols; c++)
		{
			Trasp(c,r)=a.get(r,c);
		}
	  }
	}
  return Trasp;
}
//////////////////////////////////////////////////////////////////////////
/**
 * returns a matrix senza le righe doppie (se le righe sono doppie le cancella)
 */ 
Matrix EraseARow(const Matrix& a,const int n)
{
	int rows = a.GetRows();
	int cols = a.GetCols();
	Matrix Prima = Zeros(rows-1, cols);

if (rows>1 && cols>1)
	{
	for (int r = 1; r <= rows; r++)
	  {
		if (r!=n)
		{
			for (int c = 1; c <= cols; c++)
			{
			if (r<n){
				Prima(r, c) = a.get(r,c);
					}
			else{
				Prima(r-1, c) = a.get(r-1,c);
				}	
			}
		}
	  }
	}
  return Prima;
}
//////////////////////////////////////////////////////////////////////////
/**
 * returns a matrix senza le righe doppie (se le righe sono doppie le cancella)
 */ 
Matrix EraseACol(const Matrix& a,const int n)
{
	int rows = a.GetRows();
	int cols = a.GetCols();
	Matrix Terza= Zeros(rows, cols-1);
	
if (rows>1 && cols>1)
	{
		Matrix Prima= Matrix(cols, rows);
		Matrix Seconda= Matrix(cols-1, rows);
		Prima=Trans(a);
		Seconda=EraseARow(Prima,n);
		Terza=Trans(Seconda);
	}
  return Terza;
}
//////////////////////////////////////////////////////////////////////////
/**
 * returns a matrix union of two matrix of equal dimension
 */ 
Matrix UnionCols(const Matrix& a,const Matrix& b)
{
	int rowsa = a.GetRows();
	int colsa = a.GetCols();
	int rowsb = b.GetRows();
	int colsb = b.GetCols();
	int newrows=rowsa+rowsb;
	Matrix United= Zeros(newrows, colsa);

if (colsa==colsb)
	{
	for (int r = 1; r <= rowsa; r++)
	  {
	for (int c = 1; c <= colsa; c++)
		{
			United(r,c)=a.get(r,c);
		}
	  }
	for (int r = rowsa+1; r <= newrows; r++)
	  {
	for (int c = 1; c <= colsa; c++)
		{
			United(r,c)=b.get(r,c);
		}
	  }
	}
  return United;
}

/**
 * returns a matrix union of two matrix of equal dimension
 */ 
Matrix UnionRows(const Matrix& a,const Matrix& b)
{
	int rowsa = a.GetRows();
	int colsa = a.GetCols();
	int rowsb = b.GetRows();
	int colsb = b.GetCols();
	int newcols=colsa+colsb;
	Matrix United= Zeros(rowsa, newcols);

if (rowsa==rowsb)
	{
	for (int r = 1; r <= rowsa; r++)
	  {
	for (int c = 1; c <= colsa; c++)
		{
			United(r,c)=a.get(r,c);
		}
	  }
	for (int r = 1; r <= rowsb; r++)
	  {
	for (int c = 1+colsa; c <= newcols; c++)
		{
			United(r,c)=b.get(r,c);
		}
	  }
	}
  return United;
}

/**
 * estrai una riga dalla matrice, metti la riga o la colonna che vuoi, e nell'altra metti zero
 */ 
Matrix ExtractARowOrACol(const Matrix& a, const int riganum, const int colonnanum)
{
	int rows = a.GetRows();
	int cols = a.GetCols();
	Matrix Extracted;
if (riganum==0)
{
	Extracted=Zeros(rows,1);
	for (int r = 1; r <= rows; r++)
	{
		Extracted(r,1)=a.get(r,colonnanum);
	}

	
}
else if (colonnanum==0)
{
	Extracted=Zeros(1,cols);
	for (int c = 1; c <= cols; c++)
	{
		Extracted(1,c)=a.get(riganum,c);
	}
}
else{
		Extracted=Matrix(rows,cols);
}

  return Extracted;
}

int ConfrontaMatrix(const Matrix& a, const Matrix& b)
{
	int yorn=0;
	int rowsa = a.GetRows();
	int colsa = a.GetCols();
	int rowsb = b.GetRows();
	int colsb = b.GetCols();
	double grado=0;
	if (rowsa==rowsb && colsa==colsb)
	{
			for (int r = 1; r <= rowsa; r++)
			{
			for (int c = 1; c <= colsa; c++)
			{
				if (b.get(r,c)==a.get(r,c))
				{
					grado=grado+1;
				}
			}
			}
			if (grado==rowsa*colsa)
			{
				yorn=1;
			}
	}
	
		
	return yorn;
}
/**
 * returns a matrix without twin rows
 */ 
Matrix EraseTwinRows(const Matrix& a)
{
	int rows = a.GetRows();
	int cols = a.GetCols();
	Matrix Erased;
	Erased=ExtractARowOrACol(a,1,0);
	Matrix prova;
	int godo=0;
	for (int c = 1; c <= cols; c++)
	{
		prova=ExtractARowOrACol(a,c,0);
		for (int d=1; d<=cols;d++)
		{
			if (ConfrontaMatrix(prova,ExtractARowOrACol(Erased,d,0)))
			{
				godo=godo+1;
			}
		}
	if (godo==0)
	{
		Erased=UnionCols(Erased,prova);
	}
	}	
  return Erased;
}

int Trace(const Matrix& a)
{
	int rows = a.GetRows();
	int cols = a.GetCols();
	double Traccia=0;
		for (int r = 1; r <= rows; r++)
	  {
	for (int c = 1; c <= cols; c++)
		{
			if (c==r)
			{
				Traccia=Traccia+a.get(r,c);
			}
		}
	  }
	return Traccia;
}

double Max(const Matrix& a)
{
	int rows=a.GetRows();
	int cols=a.GetCols();
	double Massimo=a.get(1,1);
		for (int r = 1; r <= rows; r++)
	  {
	for (int c = 1; c <= cols; c++)
		{
			if (a.get(r,c)>Massimo)
			{
				Massimo=a.get(r,c);
			}
		}
	  }
	return Massimo;
}

double Min(const Matrix& a)
{
	int rows=a.GetRows();
	int cols=a.GetCols();
	double Minimo=a.get(1,1);
		for (int r = 1; r <= rows; r++)
	  {
	for (int c = 1; c <= cols; c++)
		{
			if (a.get(r,c)<Minimo)
			{
				Minimo=a.get(r,c);
			}
		}
	  }
	return Minimo;
}
double Max(const double a, const double b)
{
	double Massimo=a;
	if (b>a)
	{Massimo=b;}
return Massimo;
}
double Min(const double a, const double b)
{
	double Minimo=a;
	if (b<a)
	{Minimo=b;}
return Minimo;
}

Matrix UnionRowsAtipiche(const Matrix& a,const Matrix& b)
{
	int rowsa = a.GetRows();
	int colsa = a.GetCols();
	int rowsb = b.GetRows();
	int colsb = b.GetCols();
	int newrows=rowsa+rowsb;
	int massimo=Max(colsa,colsb);
	Matrix United= Zeros(newrows, massimo);

	for (int r = 1; r <= rowsa; r++)
	  {
	for (int c = 1; c <= colsa; c++)
		{
			United(r,c)=a.get(r,c);
		}
	  }
	for (int r = 1; r <= rowsb; r++)
	  {
	for (int c = 1+colsa; c <= newrows; c++)
		{
			United(r,c)=b.get(r,c);
		}
	  }
  return United;
}

Matrix Linspace(const double a,const double b, const double c)
{
	Matrix Finale=Zeros(1,c);
	double passo=(b-a)/c;
	for (int col=1; col<=c;col++)
	{
		Finale(1,col)=a+passo*(col-1);
	}

	Finale(1,c)=b;
	return Finale;
}

Matrix Linspace(const double a,const double b)
{
	int c=100;
	Matrix Finale=Zeros(1,c);
	double passo=(b-a)/c;
	for (int col=1; col<=c;col++)
	{
		Finale(1,col)=a+passo*(col-1);
	}
	Finale(1,100)=b;
	return Finale;
}


Matrix find(const Matrix& a,const double b)
{
	int rows = a.GetRows();
	int cols = a.GetCols();
	Matrix Posizioni;
	int i=1;
	for (int r=1; r<=rows;rows++)
	{
		for (int c=1; c<=cols;cols++)
		{
			if (b==a.get(r,c))
			{
				if (i==1)
				{
					Matrix Vector=Zeros(1,2);
					Vector(1,1)=r;
					Vector(1,2)=c;
					Posizioni=Vector;
					i++;
				}
				else
				{
					Vector(1,1)=r;
					Vector(1,2)=c;
					Posizioni=UnionRows(Posizioni,Vector);
				}
			}
		}
	}
	return Posizioni;
}

