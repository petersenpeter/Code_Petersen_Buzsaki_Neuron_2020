function [fit_params,f0] = customFit(x_out,y_out,x_bins,y_bins,StartPoint,LowerLimits,UpperLimits)
idx = find(x_out>x_bins(1) & x_out<x_bins(end) & y_out>y_bins(1) & y_out<y_bins(end));
x_out2 = x_out(idx);
y_out2 = y_out(idx);
Xedges = x_bins;
Yedges = y_bins;
[N,Xedges,Yedges] = histcounts2(x_out2,y_out2,Xedges,Yedges,'Normalization','probability');
% figure, imagesc(-Xedges,Yedges,N')

x3 = Xedges(1:end-1)+diff(Xedges(1:2))/2;
x = repmat(x3',1,size(N,2));
x = x(:);
y3 = Yedges(1:end-1)+diff(Yedges(1:2))/2;
y = repmat(y3,size(N,1),1);
y = y(:);
z = N(:);

% a: period along x axis
% b: period along y axis
% c: Envelope width along x
% d: Envelope width along y
% e: amplitude of envelopes
% f: offset
% g: offset

% x: time between fields
% y: offset within theta cycles

% Cosine envelope
% g = fittype('(sin(x*(2*pi/a)+y*(2*pi/b))*cos(x*(2*pi/c))*cos(y*(2*pi/d))+1)*e','dependent',{'z'},'independent',{'x','y'},'coefficients',{'a','b','c','d','e'});

% normpdf envelope
% g = fittype('(sin(x*(2*pi/a)+y*(2*pi/b))*normpdf(x,0,c)*normpdf(y,0,d))*e','dependent',{'z'},'independent',{'x','y'},'coefficients',{'a','b','c','d','e'});

% Gauss envelope
g_fit = fittype('((cos(x*(2*pi/(b*a))+y*(2*pi/b))+1)*exp(-(x+f*y).^2/(2*c^2))*exp(-(y-g).^2/(2*d^2)))*e','dependent',{'z'},'independent',{'x','y'},'coefficients',{'a','b','c','d','e','f','g'});
[f0,gof,output] = fit([-x,y],z,g_fit,'StartPoint',StartPoint,'Lower',LowerLimits,'Upper',UpperLimits,'TolFun',10^-12);

% Ellipse envelope
% g_fit = fittype('(cos(x*(2*pi/a)+y*(2*pi/b))+1)*mvnpdf([x,y],[0, 0],[c d;d f])*e','dependent',{'z'},'independent',{'x','y'},'coefficients',{'a','b','c','d','e','f'});
% [f0,gof,output] = fit([x,y],z,g_fit,'StartPoint',[30,90,10,0.3,0.00001,0.9],'Lower',[0,0,0.1,0.0001,0,0.1],'Upper',[300,500,100,0.5,1,10],'TolFun',10^-12);

fit_params = coeffvalues(f0);
rsquare = gof.rsquare;

clear a b c d e f g
a = fit_params(1); b = fit_params(2); c = fit_params(3); d = fit_params(4); e = fit_params(5); f = fit_params(6); g = fit_params(7);
figure, subplot(1,2,1)
imagesc(Yedges,-Xedges,N), hold on, % plot(-x3,x3*b/a,'--w')%,ylim([-200,200])
for i = -5:5
    plot(x3/a+i*b,-x3,'--r')
end

subplot(1,2,2)
z_fit = ((cos(x*(2*pi/(b*a))+y*(2*pi/b))+1).*exp(-(x+f*y).^2/(2*c^2)).*exp(-(y-g).^2/(2*d^2)))*e;
% (cos(x*(2*pi/a)+y*(2*pi/b))+1).*mvnpdf([x,y],[0,0],[c,d;d,f])*e;
z_fit2 = reshape(z_fit,size(N));

imagesc(y3,x3,z_fit2),hold on,
for i = -5:5
%     plot(-x3,x3*b/a+i*[b],'--r')
    plot(x3/a+i*b,-x3,'--r')
end
% fit_params = [fit_params,b/a];
