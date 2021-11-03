within ;
model TestModelVariables
  parameter Real test_real(min=0, max=100) = 50;
  parameter Real test_real_eval = 100 annotation(Evaluate=true);
  parameter Integer test_int(min=-100, max=100) = 1;
  parameter Boolean test_bool = true annotation(Evaluate=false);
  parameter Modelica.Blocks.Types.Smoothness test_enum = Modelica.Blocks.Types.Smoothness.ConstantSegments annotation(Evaluate=false);
  Modelica.Blocks.Interfaces.RealInput test_inp;

  Modelica.Blocks.Interfaces.RealOutput test_out;
  Real test_local;
equation
  if test_bool then
    test_out = test_inp * test_local - test_real;
  else
    test_out = test_int;
  end if;
  if test_enum == Modelica.Blocks.Types.Smoothness.ConstantSegments then
    test_local = test_real_eval;
  else
    test_local = 0;
  end if;
  annotation (uses(Modelica(version="3.2.3")));
end TestModelVariables;
