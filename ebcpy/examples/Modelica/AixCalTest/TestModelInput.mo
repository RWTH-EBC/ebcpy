within AixCalTest;
model TestModelInput
  "Basic model for testing of calibration and sensitivity analysis"
  import ModelicaServices;

    extends Modelica.Icons.Example;
   replaceable package Medium =
      Modelica.Media.Water.StandardWater
     constrainedby Modelica.Media.Interfaces.PartialMedium;
  Modelica.Fluid.Sources.MassFlowSource_T source_1(nPorts=1,
    redeclare package Medium = Medium,
    use_T_in=false,
    m_flow=0.5,
    use_m_flow_in=true,
    final T=313.15)
    annotation (Placement(transformation(extent={{-84,30},{-64,50}})));
  Modelica.Fluid.Sources.MassFlowSource_T source_2(
    nPorts=1,
    redeclare final package Medium = Medium,
    m_flow=m_flow_2,
    use_m_flow_in=true,
    final T=293.15)
              annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=180,
        origin={74,-38})));
  Modelica.Fluid.Sources.FixedBoundary sink_1(nPorts=1, redeclare package
      Medium = Medium,
    final p=Medium.p_default,
    final T=Medium.T_default,
    each final X=Medium.X_default)                      annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=180,
        origin={74,40})));
  Modelica.Fluid.Sources.FixedBoundary sink_2(nPorts=1, redeclare final package
      Medium = Medium,
    final p=Medium.p_default,
    final T=Medium.T_default,
    each final X=Medium.X_default)                      annotation (Placement(
        transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={-78,-38})));
  Modelica.Fluid.Pipes.DynamicPipe
                    heater(
    redeclare final package Medium = Medium,
    final use_T_start=true,
    redeclare model HeatTransfer =
        Modelica.Fluid.Pipes.BaseClasses.HeatTransfer.IdealFlowHeatTransfer,
    final nNodes=1,
    final use_HeatTransfer=true,
    final modelStructure=Modelica.Fluid.Types.ModelStructure.a_v_b,
    final length=2,
    final diameter=0.01,
    redeclare final model FlowModel =
        Modelica.Fluid.Pipes.BaseClasses.FlowModels.DetailedPipeFlow,
    final nParallel=1,
    final isCircular=true,
    final roughness=2.5e-5,
    final height_ab=0,
    final allowFlowReversal=heater.system.allowFlowReversal,
    each final X_start=Medium.X_default,
    final p_a_start=130000,
    final T_start=313.15)
    annotation (Placement(transformation(extent={{10,-10},{-10,10}},
        rotation=180,
        origin={0,40})));
  Modelica.Fluid.Pipes.DynamicPipe
                    heater1(
    redeclare final package Medium = Medium,
    final use_T_start=true,
    redeclare model HeatTransfer =
        Modelica.Fluid.Pipes.BaseClasses.HeatTransfer.IdealFlowHeatTransfer,
    final nNodes=1,
    final use_HeatTransfer=true,
    final modelStructure=Modelica.Fluid.Types.ModelStructure.a_v_b,
    final length=2,
    final diameter=0.01,
    redeclare final model FlowModel =
        Modelica.Fluid.Pipes.BaseClasses.FlowModels.DetailedPipeFlow,
    final nParallel=1,
    final isCircular=true,
    final roughness=2.5e-5,
    final height_ab=0,
    final allowFlowReversal=heater.system.allowFlowReversal,
    each final X_start=Medium.X_default,
    final T_start=Modelica.SIunits.Conversions.from_degC(20),
    final p_a_start=130000)
    annotation (Placement(transformation(extent={{10,-10},{-10,10}},
        rotation=0,
        origin={0,-38})));
  BaseClasses.HeatExchanger heatExchanger(final C=C)
    annotation (Placement(transformation(extent={{-18,-6},{18,14}})));
  Modelica.Blocks.Sources.Constant Gc_a(final k=heatConv_a)
    annotation (Placement(transformation(extent={{-62,-2},{-42,18}})));
  Modelica.Blocks.Sources.Constant Gc_b(final k=heatConv_b) annotation (
      Placement(transformation(
        extent={{-9,9},{9,-9}},
        rotation=180,
        origin={55,-5})));
  parameter Real heatConv_b=500
                            "Constant output value" annotation (Evaluate=false);
  parameter Real heatConv_a=300
                            "Constant output value" annotation (Evaluate=false);
  parameter Modelica.SIunits.HeatCapacity C=8000
                                            "Heat capacity of element (= cp*m)"
    annotation (Evaluate=false);
  inner Modelica.Fluid.System system(
    final g=Modelica.Constants.g_n,
    use_eps_Re=false,
    final m_flow_small=1e-2,
    final p_ambient=101325,
    final T_ambient=293.15,
    final dp_small=1)
    annotation (Placement(transformation(extent={{-100,80},{-80,100}})));
  parameter Modelica.Media.Interfaces.PartialMedium.MassFlowRate m_flow_2=0.03
    "Fixed mass flow rate going out of the fluid port" annotation (Evaluate=false);
  Modelica.Blocks.Sources.Constant m_flow_sink(final k=m_flow_2) annotation (
      Placement(transformation(
        extent={{-13,-13},{13,13}},
        rotation=180,
        origin={145,-5})));
  Modelica.Blocks.Logical.Switch switch1 annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={-102,48})));
  Modelica.Blocks.Logical.Switch switch2 annotation (Placement(transformation(
        extent={{-10,10},{10,-10}},
        rotation=180,
        origin={100,-46})));
  Modelica.Blocks.Sources.Constant const_zero(final k=0)
    annotation (Placement(transformation(extent={{-8,-8},{8,8}},
        rotation=180,
        origin={-96,-6})));
  Modelica.Blocks.Sources.Constant m_flow_source(final k=0.5)
    annotation (Placement(transformation(extent={{-154,28},{-130,52}})));
  Modelica.Blocks.Logical.GreaterThreshold greaterThreshold
    annotation (Placement(transformation(extent={{-148,-64},{-128,-44}})));
  Modelica.Blocks.Sources.CombiTimeTable combiTimeTable(
    tableOnFile=true,
    tableName="Simulation_Input",
    fileName=ModelicaServices.ExternalReferences.loadResource(
        "modelica://AixCalTest/Input/model_input.txt"),
    columns={2})
    annotation (Placement(transformation(extent={{-182,-64},{-162,-44}})));

equation
  connect(source_1.ports[1], heater.port_a)
    annotation (Line(points={{-64,40},{-10,40}}, color={0,127,255}));
  connect(heater.port_b, sink_1.ports[1])
    annotation (Line(points={{10,40},{64,40}}, color={0,127,255}));
  connect(source_2.ports[1], heater1.port_a) annotation (Line(points={{64,-38},
          {10,-38}},                  color={0,127,255}));
  connect(heater1.port_b, sink_2.ports[1])
    annotation (Line(points={{-10,-38},{-68,-38}}, color={0,127,255}));
  connect(heater.heatPorts[1], heatExchanger.port_a) annotation (Line(points={{0.1,
          35.6},{0.1,23.8},{0,23.8},{0,14}}, color={127,0,0}));
  connect(heater1.heatPorts[1], heatExchanger.port_b) annotation (Line(points={{-0.1,
          -33.6},{-0.1,-20.8},{0,-20.8},{0,-6}},      color={127,0,0}));
  connect(Gc_a.y, heatExchanger.Gc_a) annotation (Line(points={{-41,8},{-28,8},
          {-28,10.8},{-21.6,10.8}},
                                color={0,0,127}));
  connect(Gc_b.y, heatExchanger.Gc_b) annotation (Line(points={{45.1,-5},{30,-5},
          {30,-3},{21.6,-3}},
                            color={0,0,127}));
  connect(switch1.y, source_1.m_flow_in)
    annotation (Line(points={{-91,48},{-84,48}}, color={0,0,127}));
  connect(switch2.y, source_2.m_flow_in)
    annotation (Line(points={{89,-46},{84,-46}}, color={0,0,127}));
  connect(greaterThreshold.y, switch1.u2) annotation (Line(points={{-127,-54},{
          -120,-54},{-120,48},{-114,48}}, color={255,0,255}));
  connect(greaterThreshold.y, switch2.u2) annotation (Line(points={{-127,-54},{
          -84,-54},{-84,-82},{148,-82},{148,-46},{112,-46}}, color={255,0,255}));
  connect(combiTimeTable.y[1], greaterThreshold.u)
    annotation (Line(points={{-161,-54},{-150,-54}}, color={0,0,127}));
  connect(m_flow_source.y, switch1.u3)
    annotation (Line(points={{-128.8,40},{-114,40}}, color={0,0,127}));
  connect(const_zero.y, switch1.u1) annotation (Line(points={{-104.8,-6},{-118,
          -6},{-118,56},{-114,56}}, color={0,0,127}));
  connect(const_zero.y, switch2.u1) annotation (Line(points={{-104.8,-6},{-118,
          -6},{-118,-90},{126,-90},{126,-38},{112,-38}}, color={0,0,127}));
  connect(m_flow_sink.y, switch2.u3) annotation (Line(points={{130.7,-5},{130.7,
          -6},{122,-6},{122,-54},{112,-54}}, color={0,0,127}));
  annotation (Icon(coordinateSystem(preserveAspectRatio=false, extent={{-160,
            -100},{160,100}})),                                  Diagram(
        coordinateSystem(preserveAspectRatio=false, extent={{-160,-100},{160,
            100}})),
    experiment(StopTime=3600, Interval=1));
end TestModelInput;
