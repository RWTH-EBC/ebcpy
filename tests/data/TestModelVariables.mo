within ;
model TestModelVariables
  parameter Real test_par(min=0, max=100) = 50;
  parameter Real test_par_2 = 100 annotation(Evaluate=true);
  Modelica.Blocks.Interfaces.RealInput test_inp;
  Modelica.Blocks.Interfaces.RealOutput test_out;
  Real test_local;
equation
  test_local = test_par_2;
  test_out = test_inp * test_local - test_par;
  annotation (uses(Modelica(version="3.2.3")));
end TestModelVariables;
