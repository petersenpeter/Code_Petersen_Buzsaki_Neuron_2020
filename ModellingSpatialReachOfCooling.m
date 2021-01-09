% Modelling the reach of the temperature in the brain and its effect on theta and error trials
%
% Equation
% deltaT(r) = deltaTemp_probe * K_0(r/lambda) / K_0(r_p/lambda)
%
% Parameters
% deltaTemp_probe:  steady state temperature at the probe 
% K_0(r):           zero-order modified bessel function of the second kind
% r:                radial distance 
% r_p:              radius of probe
% lambda: thermal length constant (1.59mm)

r_p = 0.1;
r = r_p:0.1:3;
lambda = 1.59;
deltaTemp_probe = -20;
brainTemp = 37;

% Steady state temperature distribution
deltaT_steady = deltaTemp_probe * besselk(0,r/lambda) / besselk(0,r_p/lambda);
% deltaT_steady = deltaTemp_probe * exp(-r*lambda) / exp(-r_p*lambda);
figure, subplot(3,1,1)
plot(r,deltaT_steady,'linewidth',2), hold on
plot([r_p,r_p],[0,deltaTemp_probe],'--k'), plot([r(1),r(end)],[0,0],'-k')
title('Steady state temperature'), xlabel('Distance (µm)'), ylabel('Temperature (°C)')

% Time dependent temperature distribution
clear temperature
sr = 100;
duration = 30*60;
onsets = 1/sr;
time_points = [20,66,200,1000];

temperature.temp = brainTemp * ones(1,sr*duration);
temperature.time = [1/sr:1/sr:duration];
[t,gde] = alphafunction2(0);
gde = interp1(t,gde,[t(1):(1/sr):t(end)]);
injector = zeros(1,sr*duration);
injector(round(onsets*sr)) = 1;
injector = conv(injector,gde);
injector = injector(1:length(temperature.temp));
temperature.temp = temperature.temp + injector*deltaTemp_probe;


subplot(3,1,2)
plot(temperature.time,temperature.temp,'linewidth',2), hold on
plot([0,duration],[brainTemp,brainTemp],'-k')
title('Temperature across time'), xlabel('Time (s)'), ylabel('Temperature (°C)')
subplot(3,1,3)
for i = 1:length(time_points)
    idx = find(temperature.time>time_points(i),1);
    plot(r,-(37-temperature.temp(idx)) * besselk(0,r/lambda) / besselk(0,r_p/lambda)), hold on
%     plot(r,-(37-temperature.temp(idx)) * exp(-r*lambda) / exp(-r_p*lambda)), hold on
end
legend({'t=20s','t=66s','t=200s','t=2000s'}),
title('Changing temperature'), xlabel('Distance (µm)'), ylabel('Temperature (°C)')
