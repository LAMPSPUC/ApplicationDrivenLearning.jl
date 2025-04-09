function mpc = pglib_opf_case24_ieee_rts
mpc.version = '2';
mpc.baseMVA = 100.0;

%% area data
%	area	refbus
mpc.areas = [
	1	 1;
];

%% bus data
%	bus_i	type	Pd	     Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [
	1	 2	  6.0	 0.0	 0.0	 0.0	 1	    1.0	   -0.0	 138.0	 1	    1.05000	    0.95000;
];

%% generator data
%	bus	 Pg	     Qg      Qmax	Qmin	Vg	mBase	status	Pmax	Pmin
mpc.gen = [
	1	  5.0	 0.0	 0.0	 0.0	 1.0	 100.0	 1	  5.0	 0.0;
	1	  5.0	 0.0	 0.0	 0.0	 1.0	 100.0	 1	  5.0	 0.0;
	1	  2.5	 0.0	 0.0	 0.0	 1.0	 100.0	 1	  2.5	 0.0;
	1	  2.5	 0.0	 0.0	 0.0	 1.0	 100.0	 1	  2.5	 0.0;
];

%% generator cost data
%	2	startup	shutdown	n	c(n-1)	...	c0
mpc.gencost = [
	2	 0.0	 0.0	 3	   0.000000	 1.000000	 0.0;
	2	 0.0	 0.0	 3	   0.000000	 2.000000	 0.0;
	2	 0.0	 0.0	 3	   0.000000	 4.000000	 0.0;
	2	 0.0	 0.0	 3	   0.000000	 8.000000	 0.0;
];

%% branch data
%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
mpc.branch = [
];
