within AixCalTest.BaseClasses;
model HeatExchanger

  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor heatCapacitor(C=C)
    annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={-30,0})));
  Modelica.Thermal.HeatTransfer.Components.Convection convection_a annotation (
      Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=90,
        origin={0,68})));
  Modelica.Thermal.HeatTransfer.Components.Convection convection_b annotation (
      Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=270,
        origin={0,-70})));
  Modelica.Thermal.HeatTransfer.Interfaces.HeatPort_b port_a
    annotation (Placement(transformation(extent={{-10,90},{10,110}})));
  Modelica.Thermal.HeatTransfer.Interfaces.HeatPort_b port_b
    annotation (Placement(transformation(extent={{-10,-110},{10,-90}})));
  parameter Modelica.SIunits.HeatCapacity C "Heat capacity of element (= cp*m)";
  Modelica.Blocks.Interfaces.RealInput Gc_a
    annotation (Placement(transformation(extent={{-140,48},{-100,88}})));
  Modelica.Blocks.Interfaces.RealInput Gc_b annotation (Placement(
        transformation(
        extent={{-20,-20},{20,20}},
        rotation=180,
        origin={120,-70})));
equation
  connect(convection_a.solid, heatCapacitor.port) annotation (Line(points={{
          -4.44089e-16,58},{0,58},{0,0},{-20,0}}, color={191,0,0}));
  connect(heatCapacitor.port, convection_b.solid)
    annotation (Line(points={{-20,0},{0,0},{0,-60}}, color={191,0,0}));
  connect(convection_a.fluid, port_a)
    annotation (Line(points={{0,78},{0,100}}, color={191,0,0}));
  connect(convection_b.fluid, port_b)
    annotation (Line(points={{0,-80},{0,-100}}, color={191,0,0}));
  connect(Gc_b, convection_b.Gc)
    annotation (Line(points={{120,-70},{10,-70}}, color={0,0,127}));
  connect(Gc_a, convection_a.Gc)
    annotation (Line(points={{-120,68},{-10,68}}, color={0,0,127}));
  annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
        coordinateSystem(preserveAspectRatio=false)));
end HeatExchanger;
