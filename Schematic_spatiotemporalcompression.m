% Place field schematic figure
x = 1:0.01:300;
c1 = 20;    % size of placefield1
mu1 = 100;   % center of placefield1
a1 = 10;    % period length control theta
a2 = 14;     % period length cooling theta
mu2 = 130;   % center of placefield1

% Temporal coding
c2 = 20;    % Place field size increased
tau1 = -1; % Preserved tau

% Phase coding
c3 = 30;    % Place field size preserved
tau2 = 1.7; % Greater tau

% Control condition
envelope1 = exp(-(x-mu1).^2/(2*c1^2)).*(cos(x*(2*pi/a1))+1);
envelope2 = exp(-(x-mu2).^2/(2*c2^2)).*(cos(x*(2*pi/a1)+tau1)+1);

% Temporal coding: 
envelope3 = exp(-(x-mu1).^2/(2*c3^2)).*(cos(x*(2*pi/a2))+1);
envelope4 = exp(-(x-mu2).^2/(2*c3^2)).*(cos(x*(2*pi/a2)+tau1)+1);

% Phase coding
envelope5 = exp(-(x-mu1).^2/(2*c1^2)).*(cos(x*(2*pi/a2))+1);
envelope6 = exp(-(x-mu2).^2/(2*c2^2)).*(cos(x*(2*pi/a2)+tau2)+1);

figure, 
subplot(3,1,1)
plot(x,envelope1), hold on, plot(x,envelope2), title('Control')
subplot(3,1,2)
plot(x,envelope3), hold on, plot(x,envelope4), title('Temporal coding')
subplot(3,1,3)
plot(x,envelope5), hold on, plot(x,envelope6), title('Phase coding')