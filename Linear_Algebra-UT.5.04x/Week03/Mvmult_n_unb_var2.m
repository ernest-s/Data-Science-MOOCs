
% Copyright 2016 The University of Texas at Austin
%
% For licensing information see
%                http://www.cs.utexas.edu/users/flame/license.html 
%                                                                                 
% Programmed by: Ernest Kirubakaran Selvaraj
%                ernest.kirubakaran@gmail.com

function [ Y_out ] = Mvmult_n_unb_var2_unb( A, X, Y )

  [ AL, AR ] = FLA_Part_1x2( A, ...
                               0, 'FLA_LEFT' );

  [ XT, ...
    XB ] = FLA_Part_2x1( X, ...
                         0, 'FLA_TOP' );

  while ( size( AL, 2 ) < size( A, 2 ) )

    [ A0, a1, A2 ]= FLA_Repart_1x2_to_1x3( AL, AR, ...
                                         1, 'FLA_RIGHT' );

    [ X0, ...
      x1t, ...
      X2 ] = FLA_Repart_2x1_to_3x1( XT, ...
                                    XB, ...
                                    1, 'FLA_BOTTOM' );

    %------------------------------------------------------------%

    Y = laff_axpy ( x1t, a1, Y);
    %------------------------------------------------------------%

    [ AL, AR ] = FLA_Cont_with_1x3_to_1x2( A0, a1, A2, ...
                                           'FLA_LEFT' );

    [ XT, ...
      XB ] = FLA_Cont_with_3x1_to_2x1( X0, ...
                                       x1t, ...
                                       X2, ...
                                       'FLA_TOP' );

  end

  Y_out = Y;


return
