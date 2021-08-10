within ;
model HeatPumpSystemForTMO "Example for a heat pump system"
  package Medium_sin = AixLib.Media.Water;
  package Medium_sou = AixLib.Media.Water;
  AixLib.Fluid.MixingVolumes.MixingVolume vol(
    energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
    redeclare package Medium = Modelica.Media.Air.SimpleAir,
    V=40,
    m_flow_nominal=40*6/3600,
    T_start=293.15)
    annotation (Placement(transformation(extent={{86,34},{106,54}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalConductor theCon(G=10000/40)
    "Thermal conductance with the ambient"
    annotation (Placement(transformation(extent={{38,54},{58,74}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow preHea
    "Prescribed heat flow"
    annotation (Placement(transformation(extent={{38,84},{58,104}})));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor heaCap(C=2*40*1.2*1006)
    "Heat capacity for furniture and walls"
    annotation (Placement(transformation(extent={{80,64},{100,84}})));
  Modelica.Blocks.Sources.CombiTimeTable timTab(
      extrapolation=Modelica.Blocks.Types.Extrapolation.Periodic,
      smoothness=Modelica.Blocks.Types.Smoothness.ConstantSegments,
    table=[-6*3600,0; 8*3600,4000; 18*3600,0])
                          "Time table for internal heat gain"
    annotation (Placement(transformation(extent={{-2,84},{18,104}})));
  AixLib.Fluid.HeatExchangers.Radiators.RadiatorEN442_2 rad(
    energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
    T_a_nominal(displayUnit="degC") = 323.15,
    dp_nominal=0,
    m_flow_nominal=20000/4180/5,
    Q_flow_nominal=20000,
    redeclare package Medium = Medium_sin,
    T_start=313.15,
    T_b_nominal=318.15,
    TAir_nominal=293.15,
    TRad_nominal=293.15)         "Radiator"
    annotation (Placement(transformation(extent={{40,2},{20,22}})));

  AixLib.Fluid.Sources.Boundary_pT preSou(
    redeclare package Medium = Medium_sin,
    nPorts=1,
    T=313.15)
    "Source for pressure and to account for thermal expansion of water"
    annotation (Placement(transformation(extent={{-48,2},{-28,22}})));

  Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature TOut
    "Outside temperature"
    annotation (Placement(transformation(extent={{-2,54},{18,74}})));
  AixLib.Fluid.Sources.Boundary_pT sou(
    nPorts=1,
    redeclare package Medium = Medium_sou,
    p=200000,
    T=283.15) "Fluid source on source side"
    annotation (Placement(transformation(extent={{102,-100},{82,-80}})));

  AixLib.Fluid.Sources.Boundary_pT sin(
    nPorts=1,
    redeclare package Medium = Medium_sou,
    p=200000,
    T=281.15) "Fluid sink on source side"
    annotation (Placement(transformation(extent={{-48,-100},{-28,-80}})));
  AixLib.Systems.HeatPumpSystems.HeatPumpSystem heatPumpSystem(
    redeclare package Medium_con = Medium_sin,
    redeclare package Medium_eva = Medium_sou,
    dataTable=AixLib.DataBase.HeatPump.EN255.Vitocal350BWH113(),
    use_deFro=false,
    massDynamics=Modelica.Fluid.Types.Dynamics.DynamicFreeInitial,
    energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
    refIneFre_constant=0.01,
    dpEva_nominal=0,
    deltaM_con=0.1,
    use_opeEnvFroRec=true,
    tableUpp=[-100,100; 100,100],
    minIceFac=0,
    use_chiller=true,
    calcPel_deFro=100,
    minRunTime(displayUnit="min"),
    minLocTime(displayUnit="min"),
    use_antLeg=false,
    use_refIne=true,
    initType=Modelica.Blocks.Types.Init.InitialOutput,
    minTimeAntLeg(displayUnit="min") = 900,
    scalingFactor=1,
    use_tableData=false,
    dpCon_nominal=0,
    use_conCap=true,
    CCon=3000,
    use_evaCap=true,
    CEva=3000,
    Q_flow_nominal=5000,
    cpEva=4180,
    cpCon=4180,
    use_secHeaGen=false,
    redeclare model TSetToNSet = AixLib.Controls.HeatPump.BaseClasses.OnOffHP (
                                                                        hys=5),
    use_sec=true,
    QCon_nominal=10000,
    P_el_nominal=2500,
    redeclare model PerDataHea =
        AixLib.DataBase.HeatPump.PerformanceData.LookUpTable2D (
        smoothness=Modelica.Blocks.Types.Smoothness.LinearSegments,
        dataTable=
            AixLib.DataBase.HeatPump.EN255.Vitocal350BWH113(
            tableP_ele=[0,-5.0,0.0,5.0,10.0,15.0; 35,3750,3750,3750,3750,3833;
            45,4833,4917,4958,5042,5125; 55,5583,5667,5750,5833,5958; 65,7000,
            7125,7250,7417,7583]),
        printAsserts=false,
        extrapolation=false),
    redeclare function HeatingCurveFunction =
        AixLib.Controls.SetPoints.Functions.HeatingCurveFunction (
                                                           TDesign=328.15),
    use_minRunTime=true,
    use_minLocTime=true,
    use_runPerHou=true,
    pre_n_start=true,
    redeclare AixLib.Fluid.Movers.Data.Pumps.Wilo.Stratos80slash1to12
                                                               perEva,
    redeclare AixLib.Fluid.Movers.Data.Pumps.Wilo.Stratos25slash1to4
                                                              perCon,
    TCon_nominal=313.15,
    TCon_start=313.15,
    TEva_start=283.15,
    use_revHP=false,
    VCon=0.004,
    VEva=0.004)
    annotation (Placement(transformation(extent={{10,-92},{64,-32}})));

  AixLib.Fluid.Sensors.TemperatureTwoPort
                             senT_a1(
    redeclare final package Medium = Medium_sin,
    final m_flow_nominal=1,
    final transferHeat=true,
    final allowFlowReversal=true)
                                "Temperature at sink inlet" annotation (
      Placement(transformation(
        extent={{10,10},{-10,-10}},
        rotation=270,
        origin={88,-20})));
  Modelica.Blocks.Interfaces.RealInput TDryBul(start=273.15)
    annotation (Placement(transformation(extent={{-162,-20},{-122,20}})));
  Modelica.Blocks.Interfaces.RealOutput Pel
    annotation (Placement(transformation(extent={{118,-18},{158,22}})));
  Modelica.Blocks.Sources.RealExpression realExpression(y=heatPumpSystem.heatPump.sigBus.PelMea)
    annotation (Placement(transformation(extent={{56,12},{76,32}})));
equation
  connect(theCon.port_b,vol. heatPort) annotation (Line(
      points={{58,64},{68,64},{68,44},{86,44}},
      color={191,0,0},
      smooth=Smooth.None));
  connect(preHea.port,vol. heatPort) annotation (Line(
      points={{58,94},{68,94},{68,44},{86,44}},
      color={191,0,0},
      smooth=Smooth.None));
  connect(heaCap.port,vol. heatPort) annotation (Line(
      points={{90,64},{68,64},{68,44},{86,44}},
      color={191,0,0},
      smooth=Smooth.None));
  connect(timTab.y[1],preHea. Q_flow) annotation (Line(
      points={{19,94},{38,94}},
      color={0,0,127},
      smooth=Smooth.None));
  connect(rad.heatPortCon,vol. heatPort) annotation (Line(
      points={{32,19.2},{32,44},{86,44}},
      color={191,0,0},
      smooth=Smooth.None));
  connect(rad.heatPortRad,vol. heatPort) annotation (Line(
      points={{28,19.2},{28,44},{86,44}},
      color={191,0,0},
      smooth=Smooth.None));
  connect(TOut.port,theCon. port_a) annotation (Line(
      points={{18,64},{38,64}},
      color={191,0,0},
      smooth=Smooth.None));
  connect(sou.ports[1], heatPumpSystem.port_a2)
    annotation (Line(points={{82,-90},{64,-90},{64,-83.4286}},
                                                          color={0,127,255}));
  connect(sin.ports[1], heatPumpSystem.port_b2) annotation (Line(points={{-28,-90},
          {14,-90},{14,-83.4286},{10,-83.4286}},
                                       color={0,127,255}));
  connect(senT_a1.port_a, heatPumpSystem.port_b1) annotation (Line(points={{88,-30},
          {88,-57.7143},{64,-57.7143}},      color={0,127,255}));
  connect(senT_a1.port_b, rad.port_a) annotation (Line(points={{88,-10},{90,-10},
          {90,12},{40,12}}, color={0,127,255}));
  connect(senT_a1.T, heatPumpSystem.TAct) annotation (Line(points={{77,-20},{54,
          -20},{54,-16},{-2,-16},{-2,-24},{5.95,-24},{5.95,-36.0714}},
                                                     color={0,0,127}));
  connect(rad.port_b, heatPumpSystem.port_a1) annotation (Line(points={{20,12},
          {-12,12},{-12,-57.7143},{10,-57.7143}},color={0,127,255}));
  connect(rad.port_b, preSou.ports[1])
    annotation (Line(points={{20,12},{-28,12}}, color={0,127,255}));
  connect(heatPumpSystem.T_oda, TDryBul) annotation (Line(points={{5.95,
          -45.0714},{-108,-45.0714},{-108,0},{-142,0}}, color={0,0,127}));
  connect(TOut.T, TDryBul) annotation (Line(points={{-4,64},{-72,64},{-72,0},{
          -142,0}}, color={0,0,127}));
  connect(realExpression.y, Pel) annotation (Line(points={{77,22},{96,22},{96,
          26},{114,26},{114,2},{138,2}}, color={0,0,127}));
  annotation (Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-120,
            -120},{120,120}})),
    experiment(Tolerance=1e-6, StopTime=86400),
__Dymola_Commands(file="modelica://AixLib/Resources/Scripts/Dymola/Systems/HeatPumpSystems/Examples/HeatPumpSystem.mos"
        "Simulate and plot"),
    Documentation(info="<html><p>
  Model for testing the model <a href=
  \"modelica://AixLib.Systems.HeatPumpSystems.HeatPumpSystem\">AixLib.Systems.HeatPumpSystems.HeatPumpSystem</a>.
</p>
<p>
  A simple radiator is used to heat a room. This example is based on
  the example in <a href=
  \"modelica://AixLib.Fluid.HeatPumps.Examples.ScrollWaterToWater_OneRoomRadiator\">
  AixLib.Fluid.HeatPumps.Examples.ScrollWaterToWater_OneRoomRadiator</a>.
</p>
</html>",
   revisions="<html><ul>
  <li>
    <i>May 22, 2019</i> by Julian Matthes:<br/>
    Rebuild due to the introducion of the thermal machine partial model
    (see issue <a href=
    \"https://github.com/RWTH-EBC/AixLib/issues/715\">#715</a>)
  </li>
  <li>
    <i>November 26, 2018&#160;</i> by Fabian Wüllhorst:<br/>
    First implementation (see issue <a href=
    \"https://github.com/RWTH-EBC/AixLib/issues/577\">#577</a>)
  </li>
</ul>
</html>"),
    Icon(coordinateSystem(extent={{-120,-120},{120,120}}), graphics={
        Ellipse(lineColor = {75,138,73},
                fillColor={255,255,255},
                fillPattern = FillPattern.Solid,
                extent={{-120,-120},{120,120}}),
        Polygon(lineColor = {0,0,255},
                fillColor = {75,138,73},
                pattern = LinePattern.None,
                fillPattern = FillPattern.Solid,
                points={{-38,64},{68,-2},{-38,-64},{-38,64}})}),
    uses(AixLib(version="1.0.0"), Modelica(version="4.0.0")),
    version="1",
    conversion(noneFromVersion=""));
end HeatPumpSystemForTMO;
