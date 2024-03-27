#include <iostream>
#include<Eigen/Eigen>
#include<vector>
#include<limits>

using namespace std;
using namespace Eigen;

VectorXd solPALU(MatrixXd A, VectorXd b)
{
    // Solution with PALU
    PartialPivLU<MatrixXd> lu(A);
    VectorXd x = lu.solve(b);
    return x;
}

VectorXd solQR(MatrixXd A, VectorXd b)
{
    // Solution with QR
    HouseholderQR<MatrixXd> qr(A);
    VectorXd x = qr.solve(b);
    return x;
}


int main()
{

    VectorXd sol = VectorXd::Constant(2, -1.0);


    // System 1

    double err1_PALU = 0, err1_QR = 0;

    Matrix2d A1 {
        {5.547001962252291e-01, -3.770900990025203e-02},
        {8.320502943378437e-01, -9.992887623566787e-01}
    };

    if(fabs(A1.determinant()) <= numeric_limits<double>::epsilon())
        cerr << "The matrix A1 is singular" << endl;
    else
    {
        Vector2d b1 = {-5.169911863249772e-01, 1.672384680188350e-01};

        VectorXd x1_PALU = solPALU(A1, b1);
        err1_PALU = (x1_PALU - sol).norm() / (sol.norm());
        cout << "Relative error of x1_PALU: " << err1_PALU << endl;

        VectorXd x1_QR = solQR(A1,b1);
        err1_QR = (x1_QR - sol).norm() / (sol.norm());
        cout << "Relative error of x1_QR: " << err1_QR << endl;

        if(err1_PALU > err1_QR)
        {
            cout << "Best solution with QR " << endl;
        }
        else
        {
            cout << "Best solution with PALU " << endl;
        }
    }


    // System 2

    double err2_PALU = 0, err2_QR = 0;

    Matrix2d A2 {
        {5.547001962252291e-01, -5.540607316466765e-01},
        {8.320502943378437e-01, -8.324762492991313e-01}
    };

    if(fabs(A2.determinant()) <= numeric_limits<double>::epsilon())
        cerr << "The matrix A2 is singular" << endl;
    else
    {
        Vector2d b2 = {-6.394645785530173e-04, 4.259549612877223e-04};

        VectorXd x2_PALU = solPALU(A2, b2);
        err2_PALU = (x2_PALU - sol).norm() / (sol.norm());
        cout << endl << "Relative error of x2_PALU: " << err2_PALU << endl;

        VectorXd x2_QR = solQR(A2,b2);
        err2_QR = (x2_QR - sol).norm() / (sol.norm());
        cout << "Relative error of x2_QR: " << err2_QR << endl;

        if(err2_PALU > err2_QR)
        {
            cout << "Best solution with QR " << endl;
        }
        else
        {
            cout << "Best solution with PALU " << endl;
        }
    }


    // System 3
    double err3_PALU = 0, err3_QR = 0;

    Matrix2d A3 {
        {5.547001962252291e-01, -5.547001955851905e-01},
        {8.320502943378437e-01, -8.320502947645361e-01}
    };

    if(fabs(A3.determinant()) <= numeric_limits<double>::epsilon())
        cerr << "The matrix A3 is singular" << endl;
    else
    {
        Vector2d b3 = {-6.400391328043042e-10, 4.266924591433963e-10};

        VectorXd x3_PALU = solPALU(A3, b3);
        err3_PALU = (x3_PALU - sol).norm() / (sol.norm());
        cout << endl << "Relative error of x3_PALU: " << err3_PALU << endl;

        VectorXd x3_QR = solQR(A3,b3);
        err3_QR = (x3_QR - sol).norm() / (sol.norm());
        cout << "Relative error of x3_QR: " << err3_QR << endl;

        if(err3_PALU > err3_QR)
        {
            cout << "Best solution with QR " << endl;
        }
        else
        {
            cout << "Best solution with PALU " << endl;
        }
    }

    return 0;
}
