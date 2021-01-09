function [t,gde] = alphafunction2(plots)
% By Peter Petersen
% petersen.peter@gmail.com
% Last edited: 13-09-2019

% Alpha
tmax=3000;
dt=1;
t=0:dt:tmax;
tau= 18;
tau2= 550;
ts=1;
tr=t(round(ts/dt):length(t))-(ts-dt);

% Alpha
gal=zeros(size(t));
galp=tau/exp(1);
gal(round(ts/dt):length(t))=tr.*exp(-tr/tau)/galp;

% Dual exponential
gde=zeros(size(t));
tp=(tau*tau2)*log(tau2/tau)/(tau2-tau);
gdep=(tau*tau2)*(exp(-tp/tau2)-exp(-tp/tau))/(tau2-tau);
gde(round(ts/dt):length(t))=(tau*tau2)*(exp(-tr/tau2)-exp(-tr/tau))/((tau2-tau)*gdep);


if exist('plots')
    if plots == 1
    figure
    plot(t,gde,'k-');
    title('Alpha function','FontSize',12,'FontName','Helvetica');
    xlabel('t (msecs)','FontSize',11,'FontName','Helvetica');
    axis([0 tmax 0 1.02]);
    set(gca,'Box','off');
    end
end
