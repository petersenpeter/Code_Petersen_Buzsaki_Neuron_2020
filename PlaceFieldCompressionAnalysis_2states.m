ids = [126,127,139,140,92,93,78,79,80,81,   168,166,153,151,149,94,91,88,83];
% ids = [143]; % 127 % ,144
for i = 1:length(ids)
    id = ids(i)
%     id = 92
    generateFigures(id)
    close all
end

%%
clear all, close all
ids = [126,127,140,92,93,78,79,80,81,   168,166,153,151,149,94,91,88,83]; % 139 not good in post-cooling     
% idsToLoad = [126,127,140,93,78,80,81]; % 79
% 140 very weak response, 92 too few cells, 93 too few cells
MedialSeptum_Recordings
placefield_difference = {[],[],[]};
ccg_delay = {[],[],[]};
placefield_speed = {[],[],[]};
placefield_time_offset = {[],[],[]};
precession_slope = {[],[],[]};
ccg_delay_phase = {[],[],[]};
ccg_delay_time_peaks = {[],[],[]};
ccg_delay_phase_peaks = {[],[],[]};
placefield_ccgs_time = {[],[],[]};
placefield_ccgs_phase = {[],[],[]};
ACGs = {[],[],[]};
placefield_interneurons_ccgs_time = {[],[],[]};

pairIDs_PyrInt = {[],[],[]};
pairIDs_Pyr = {[],[],[]};
pairIDs_CCGs_time = {[],[],[]};
pairIDs_CCGs_phase = {[],[],[]};

% conditions = {'Pre','Cooling','Post'};
conditions = {'NoCooling','Cooling'};
colors = {'r','b','g'};
conserveIds = 1;

for i = 1:length(ids)
    kept_ids = [];
    id = ids(i);
    recording = recordings(id);

    cd(fullfile(datapath, recording.animal_id, recording.name))
    load('PlaceFields2.mat')
    if conserveIds
        % Pyramidal pairs
        kept_ids = [];
        A = pairIDs_Pyr_all{1}(:,1)*1000 + pairIDs_Pyr_all{1}(:,2);
        B = pairIDs_Pyr_all{2}(:,1)*1000 + pairIDs_Pyr_all{2}(:,2);
%         C = pairIDs_Pyr_all{3}(:,1)*1000 + pairIDs_Pyr_all{3}(:,2);
        [Com,kept_ids{1},kept_ids{2},kept_ids{3}] = intersect3(A,B,B);
        
        % CCGs time
        kept_ids2 = [];
        A = pairIDs_CCGs_time_all{1}(:,1)*1000 + pairIDs_CCGs_time_all{1}(:,2);
        B = pairIDs_CCGs_time_all{2}(:,1)*1000 + pairIDs_CCGs_time_all{2}(:,2);
%         C = pairIDs_CCGs_time_all{3}(:,1)*1000 + pairIDs_CCGs_time_all{3}(:,2);
        [Com,kept_ids2{1},kept_ids2{2},kept_ids2{3}] = intersect3(A,B,B);
        
        % CCGs phase
        kept_ids3 = [];
        A = pairIDs_CCGs_phase_all{1}(:,1)*1000 + pairIDs_CCGs_phase_all{1}(:,2);
        B = pairIDs_CCGs_phase_all{2}(:,1)*1000 + pairIDs_CCGs_phase_all{2}(:,2);
%         C = pairIDs_CCGs_phase_all{3}(:,1)*1000 + pairIDs_CCGs_phase_all{3}(:,2);
        [Com,kept_ids3{1},kept_ids3{2},kept_ids3{3}] = intersect3(A,B,B);
        
        % Pyramidal-Interneuron pairs
        kept_ids4 = [];
        if ~any(cellfun(@isempty,pairIDs_PyrInt_all))
            A = pairIDs_PyrInt_all{1}(:,1)*1000 + pairIDs_PyrInt_all{1}(:,2);
            B = pairIDs_PyrInt_all{2}(:,1)*1000 + pairIDs_PyrInt_all{2}(:,2);
%             C = pairIDs_PyrInt_all{3}(:,1)*1000 + pairIDs_PyrInt_all{3}(:,2);
            [Com,kept_ids4{1},kept_ids4{2},kept_ids4{3}] = intersect3(A,B,B);
        end
    end
    
    for ii = 1:length(conditions)
        pairIDs_Pyr{ii} = [pairIDs_Pyr{ii};pairIDs_Pyr_all{ii}];
        pairIDs_PyrInt{ii} = [pairIDs_PyrInt{ii};pairIDs_PyrInt_all{ii}];
%         pairIDs_CCGs_time{ii} = [pairIDs_CCGs_time{ii};pairIDs_CCGs_time_all{ii}];
%         pairIDs_CCGs_phase{ii} = [pairIDs_CCGs_phase{ii};pairIDs_CCGs_phase_all{ii}];
        
        placefield_difference{ii} = [placefield_difference{ii},placefield_difference_all{ii}(kept_ids{ii})];
        ccg_delay{ii} = [ccg_delay{ii},ccg_delay_all{ii}(kept_ids{ii})];
        placefield_speed{ii} = [placefield_speed{ii},placefield_speed_all{ii}(kept_ids{ii})];
        placefield_time_offset{ii} = [placefield_time_offset{ii},placefield_time_offset_all{ii}(kept_ids{ii})];
        precession_slope{ii} = [precession_slope{ii},precession_slope_all{ii}(kept_ids{ii},:)'];
        ccg_delay_phase{ii} = [ccg_delay_phase{ii},ccg_delay_phase_all{ii}(kept_ids{ii})];
        
        % Al ccg peaks
        ccg_delay_time_peaks{ii} =  [ccg_delay_time_peaks{ii},ccg_delay_out{ii}.time_all(kept_ids{ii})];
        ccg_delay_phase_peaks{ii} =  [ccg_delay_phase_peaks{ii},ccg_delay_out{ii}.phase_all(kept_ids{ii})];
        
        % All ccg traces
        placefield_ccgs_time{ii} = [placefield_ccgs_time{ii};placefield_ccgs_time_all{ii}(kept_ids2{ii},:)];
        placefield_ccgs_phase{ii} = [placefield_ccgs_phase{ii};placefield_ccgs_phase_all{ii}(kept_ids3{ii},:)];
        
        % ACGs
        ACGs{ii} = [ACGs{ii};ACGs_all{ii}];
        
        % CCG between pyramidal cells and interneurons
        if ~isempty(kept_ids4)
            placefield_interneurons_ccgs_time{ii} = [placefield_interneurons_ccgs_time{ii};placefield_interneurons_ccgs_time_all{ii}(kept_ids4{ii},:)];
        end
    end
end

%%
figure(3)
x_bins = [-50:2:50];
t_bins = [-0.6:.03:0.6];
t_comp_bins = [-100:5:100];
phase_bins = [-2*pi:pi/10:2*pi];
distanceBetweenFields = {};
timeLagBetweenThetaSpiking = {};
timeBetweenFields = {};
phaseLagBetweenFields = {};
for iii = 1:length(conditions)
    figure(3)
    subplot(3,3,1+iii-1), hold on
    x = placefield_difference{iii}; 
    y1 = ccg_delay{iii};
    subset = find( y1 < 90 & y1 > -90 & x > -60 & x < 60  & precession_slope{iii}(1,:)<-0.005 & precession_slope{iii}(1,:)>-0.03 & precession_slope{iii}(2,:)<-0.005 & precession_slope{iii}(2,:)>-0.03 );
    x = x(subset); y1 = y1(subset);
    plot(x,y1,'.'), hold on
    P = polyfit(x,y1,1); yfit = P(1)*x+P(2);
    [R,P] = corrcoef(x,y1);
    text(-100,190,['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k')
    xlabel('Distance between fields (cm)'), ylabel('Time lag between theta spiking (ms)'), title(['Compression: ' conditions{iii}]), 
    xlim([-100,100]), ylim([-150,150]), grid on, hold on
%     [slope1,~,~] = CircularLinearRegression(y1/10,x,1);
%     plot(x,2*pi*slope1*x*10,'k-','linewidth',1.5), text(50,-100,num2str(slope1))
    slope2 = fitEllipse([x;y1]'); axis tight, text(50,-70,['Ellipse: ' num2str(slope2)])

    figure(2)
    subplot(2,2,1), hold on
    [param]=sigm_fit(x,y1);
    text(-100+iii*50,100,['Slope: ' num2str(param(4),3)],'Color','k')
    figure(4)
    subplot(2,3,1), hold on, title('Distance between fields (cm)'), axis tight
    histogram(x,x_bins)
    subplot(2,3,4), hold on, title('Time lag between theta spiking (ms)'), axis tight
    histogram(y1,t_comp_bins)
    distanceBetweenFields{iii} = x;
    timeLagBetweenThetaSpiking{iii} = y1;
    
    figure(3)
    subplot(3,3,4+iii-1), hold on
    x = placefield_time_offset{iii};%(placefield_difference{iii}./placefield_speed{iii});  % 
    y1 = ccg_delay{iii};
    subset = find( y1 < 90 & y1 > -90 & x > -0.600 & x < 0.600  & precession_slope{iii}(1,:)<-0.005 & precession_slope{iii}(1,:)>-0.03 & precession_slope{iii}(2,:)<-0.005 & precession_slope{iii}(2,:)>-0.03 );
    x = x(subset); y1 = y1(subset);
    plot(x,y1,'.'), hold on
    P = polyfit(x,y1,1); yfit = P(1)*x+P(2);
    [R,P] = corrcoef(x,y1);
    text(-0.8,0.130,['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k')
    xlabel('Time between fields (s)'), ylabel('Time lag between theta spiking (s)')
    xlim([-0.90,0.90]), ylim([-0.150,0.150]), grid on, hold on
%     [slope1,~,~] = CircularLinearRegression(y1*5,x,1);
%     plot(x,2*pi*slope1*x/5,'k-','linewidth',1.5), text(0.50,120,num2str(slope1));
%     GMModel = fitgmdist([linc;circ]',1);
%     plot(GMModel.mu,'or')
%     ezcontour(@(x1,x2)pdf(GMModel,[x1 x2]),get(gca,{'XLim','YLim'}))
%     plot(x,2*pi*slope1*x,'r-','linewidth',1.5), text(50,-5,num2str(slope1))
    slope2 = fitEllipse([x;y1]'); axis tight, text(0.50,-70,['Ellipse: ' num2str(slope2)])

    figure(2)
    subplot(2,2,2), hold on
    [param]=sigm_fit(x,y1);
    text(-1.+iii*0.5,0.17,['Slope: ' num2str(param(4),3)],'Color','k')
    figure(4)
    subplot(2,3,2), hold on, title('Time between fields (cm)'), axis tight
    histogram(x,t_bins)
    subplot(2,3,5), hold on, title('Time lag between theta spiking (s)'), axis tight
    histogram(y1,t_comp_bins)
    timeBetweenFields{iii} = x;
    
    % Phase
    figure(3)
    subplot(3,3,7+iii-1), hold on
    x = placefield_difference{iii}%placefield_time_offset{iii};%(placefield_difference{iii}./placefield_speed{iii}); 
    y1 = ccg_delay_phase{iii};
    subset = find( y1 < 5 & y1 > -5 & x > -50 & x < 50  & precession_slope{iii}(1,:)<-0.005 & precession_slope{iii}(1,:)>-0.03 & precession_slope{iii}(2,:)<-0.005 & precession_slope{iii}(2,:)>-0.03 );
    x = x(subset); y1 = y1(subset);
    plot(x,y1,'.')
    P = polyfit(x,y1,1); yfit = P(1)*x+P(2);
    [R,P] = corrcoef(x,y1);
    text(-50,7.130,['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k')
    xlabel('Distance between fields (cm)'), ylabel('Phase lag between fields')
    xlim([-100,100]), ylim([-2*pi,4*pi]), grid on, hold on
    [slope1,~,~] = CircularLinearRegression(y1,x,1);
    plot(x,2*pi*slope1*x,'r-','linewidth',1.5), text(20,-3.,['Phase precession: ',num2str(slope1)]), axis tight
    slope2 = fitEllipse([x;y1]'); axis tight, text(20,-4,['Ellipse: ' num2str(slope2)])
    
    figure(2)
    subplot(2,2,3), hold on
    [param]=sigm_fit(x,y1);
    text(-80+iii*40,5,['Slope: ' num2str(param(4),3)],'Color','k')
    figure(4)
    subplot(2,3,3), hold on, title('Distance between fields (cm)'), axis tight
    histogram(x,x_bins)
    subplot(2,3,6), hold on, title('Phase lag between fields'), axis tight
    histogram(y1,phase_bins)
    phaseLagBetweenFields{iii} = y1;
    
%     subplot(3,2,5), hold on
%     histogram(x,[-1:0.05:1],'Normalization','probability'), grid on, axis tight, legend(conditions)
%     subplot(3,2,6), hold on
%     histogram(y1,[-0.15:0.01:0.15],'Normalization','probability'), grid on, axis tight, legend(conditions)
end

distanceBetweenFields;
timeLagBetweenThetaSpiking;
timeBetweenFields;
phaseLagBetweenFields;

data = distanceBetweenFields;
[h1,p1] =  kstest2(data{1},data{2});
[h2,p2] =  kstest2(data{2},data{3});
[h3,p3] =  kstest2(data{1},data{3});
figure(4)
subplot(2,3,1)
text(-50,33,[num2str(p1),',  ',num2str(h1)],'Rotation',45);
text(0,33,[num2str(p2),'  ,',num2str(h2)],'Rotation',45);
text(50,33,[num2str(p3),',  ',num2str(h3)],'Rotation',45);

data = timeLagBetweenThetaSpiking;
[h1,p1] =  kstest2(data{1},data{2});
[h2,p2] =  kstest2(data{2},data{3});
[h3,p3] =  kstest2(data{1},data{3});
figure(4)
subplot(2,3,4)
text(-100,55,[num2str(p1),',  ',num2str(h1)],'Rotation',45);
text(0,55,[num2str(p2),'  ,',num2str(h2)],'Rotation',45);
text(100,55,[num2str(p3),',  ',num2str(h3)],'Rotation',45);



data = timeBetweenFields;
[h1,p1] =  kstest2(data{1},data{2});
[h2,p2] =  kstest2(data{2},data{3});
[h3,p3] =  kstest2(data{1},data{3});
figure(4)
subplot(2,3,2)
text(-0.6,55,[num2str(p1),',  ',num2str(h1)],'Rotation',45);
text(0,55,[num2str(p2),'  ,',num2str(h2)],'Rotation',45);
text(0.6,55,[num2str(p3),',  ',num2str(h3)],'Rotation',45);



data = phaseLagBetweenFields;
[p1,h1] = kstest2(data{1},data{2});
[p2,h2] = kstest2(data{2},data{3});
[p3,h3] = kstest2(data{1},data{3});
figure(4)
subplot(2,3,6)
text(-6,70,[num2str(p1),',  ',num2str(h1)],'Rotation',45);
text(0,70,[num2str(p2),'  ,',num2str(h2)],'Rotation',45);
text(6,70,[num2str(p3),',  ',num2str(h3)],'Rotation',45);

%% % All peaks
fit_params_all1 = [];
fit_params_all2 = [];
fit_params_all3 = [];
for iii = 1:length(conditions)
    figure(10)
    subplot(3,3,1+iii-1), hold on
    x = placefield_difference{iii}; 
    y1 = ccg_delay_time_peaks{iii};
    subset = find(x > -100 & x < 100  & precession_slope{iii}(1,:)<-0.005 & precession_slope{iii}(1,:)>-0.03 & precession_slope{iii}(2,:)<-0.005 & precession_slope{iii}(2,:)>-0.03 );
    x = x(subset); y1 = y1(subset);
    
    circ = [];
    linc = [];
    for i = 1:length(y1)
        circ = [circ,y1{i}'];
        linc = [linc,x(i)*ones(1,length(y1{i}))];
    end
    linc(isnan(circ)) = [];
    circ(isnan(circ)) = [];
    plot(linc,circ,'.k')
    xlabel('Distance between fields (cm)'), ylabel('Time lag between theta spiking (ms)'), title(['Compression: ' conditions{iii}]), 
    xlim([-100,100]), ylim([-490,490]), grid on, hold on,% axis tight
%     slope2 = fitEllipse([linc;circ]'); 
    axis tight, 
%     text(60,-900,['Ellipsoid: ',num2str(slope2)])
    
    % Fitting a multi sinosoidal curve to the data
    x_out = linc;
    y_out = circ;
    x_bins = -50:0.5:50;
    y_bins = [-500:5:500];
    StartPoint =  [   2,  100,  30,  500, 0.00001,   0.03,    0];
    LowerLimits = [ 0.5,   50,  10,  200,       0,      0,  -10];
    UpperLimits = [   8,  200, 100, 1500,    0.01,    0.1,   100];
    [fit_params,f01] = customFit(x_out,y_out,x_bins,y_bins,StartPoint,LowerLimits,UpperLimits);
    title(['Theta time offset vs position: ' conditions{iii}])
    fit_params_all1(iii,:) = fit_params;
    f0s{iii,1} = f01;
    % a: period along x axis
    % b: period along y axis
    % c: Envelope width along x
    % d: Envelope width along y
    % e: amplitude of envelopes
    figure(10)
    subplot(3,3,1+iii-1), hold on 
    for i = -5:5
        plot(x_bins,x_bins*fit_params(1)+i*fit_params(2),'--r')
    end
    ylim([-200,200]),xlim([-40,40])
    figure(20)
    subplot(1,3,1)
    for i = -5:5
        plot(x_bins,x_bins*fit_params(1)+i*fit_params(2),colors{iii}), hold on
    end
    ylim([-200,200]),xlim([-40,40]),title('Distance')

    figure(11)
    subplot(3,2,1), hold on
    xbins = [-450:5:450];
    indx = find( linc > -10 & linc < 10);
    [N,edges] = histcounts(circ(indx),xbins,'normalization','probability'); 
    plot(xbins(1:end-1),nanconv(N,gausswin(20)','edge')); hold on
    title('Time lag between theta cycles (ms, from position)'), plot(0,0,'.'), axis tight
    
    figure(10)
    subplot(3,3,3+iii), hold on
    x = placefield_difference{iii}./placefield_speed{iii}*1000; % placefield_time_offset{iii}*1000;%();  % 
    y1 = ccg_delay_time_peaks{iii};
    subset = find(x > -590 & x < 590  & precession_slope{iii}(1,:)<-0.005 & precession_slope{iii}(1,:)>-0.03 & precession_slope{iii}(2,:)<-0.005 & precession_slope{iii}(2,:)>-0.03 );
    x = x(subset); y1 = y1(subset);
    
    circ = [];
    linc = [];
    for i = 1:length(y1)
        circ = [circ,y1{i}'];
        linc = [linc,x(i)*ones(1,length(y1{i}))];
    end
    
    linc(isnan(circ)) = [];
    circ(isnan(circ)) = [];
    plot(linc,circ,'.k')
    xlabel('Time between fields (s)'), ylabel('Time lag between theta spiking (s)')
    xlim([-650,650]), ylim([-490,490]), grid on, %axis tight
%     slope2 = fitEllipse([linc;circ]'); 
    axis tight, 
%     text(0.60,-900,['Ellipsoid: ',num2str(slope2/1000)])
    
    % Fitting a multi sinosoidal curve to the data
    x_out = linc;
    y_out = circ;
    x_bins = -400:8:400;
    y_bins = [-490:10:490]; 
    StartPoint =  [0.2,   100,  400,   800,  0.00001,  0.6,     0];
    LowerLimits = [0.1,   50,  100,   400, 0.000001, 0.2,  -500];
    UpperLimits = [0.3,  150,  700,  1500,    0.001,  1.5,   500];
%     pd = fitdist([x_out;y_out],'SpatialTemporalCompression')
    [fit_params,f02] = customFit(x_out,y_out,x_bins,y_bins,StartPoint,LowerLimits,UpperLimits);
    title(['Theta time offset vs time: ' conditions{iii}])
    fit_params_all2(iii,:) = fit_params;
    f0s{iii,2} = f02;
    
    % a: period along x axis
    % b: period along y axis
    % c: Envelope width along x
    % d: Envelope width along y
    % e: amplitude of envelopes
    
    figure(10)
    subplot(3,3,3+iii), hold on
    for i = -5:5
        plot(x_bins,x_bins*fit_params(1)+i*fit_params(2),'--r')
    end
    ylim([-200,200]),xlim([-400,400])
    
    figure(20),
    subplot(1,3,2)
    for i = -5:5
        plot(x_bins,x_bins*fit_params(1)+i*fit_params(2),colors{iii}), hold on
    end
    ylim([-200,200]),xlim([-400,400]),title('Time')
    
    figure(11)
    subplot(3,2,3), hold on
    xbins = [-450:1:450]';
    indx = find( linc > -100 & linc < 100);
    [N,edges] = histcounts(circ(indx),xbins,'normalization','probability'); 
    plot(xbins(1:end-1),nanconv(N,gausswin(20)','edge')); hold on
    title('Time lag between theta cycles (ms, from time difference)'), plot(0,0,'.'), axis tight
    
    % Phase
    figure(10)
    subplot(3,3,7+iii-1), hold on
    x = placefield_difference{iii}; % placefield_time_offset{iii};%(placefield_difference{iii}./placefield_speed{iii}); 
    y1 = ccg_delay_phase_peaks{iii};
    subset = find(~isempty(y1) & x > -50 & x < 50  & precession_slope{iii}(1,:)<-0.005 & precession_slope{iii}(1,:)>-0.03 & precession_slope{iii}(2,:)<-0.005 & precession_slope{iii}(2,:)>-0.03 );
    x = x(subset); y1 = y1(subset);
    circ = [];
    linc = [];
    for i = 1:length(y1)
        circ = [circ,y1{i}'];
        linc = [linc,x(i)*ones(1,length(y1{i}))];
    end
    linc(isnan(circ)) = [];
    circ(isnan(circ)) = [];
    temp = find(circ > -2*pi & circ < 2*pi);
    plot(linc,circ,'.k')
    xlabel('Distance between fields (cm)'), ylabel('Phase lag between fields')
    xlim([-50,50]), ylim([-10*pi,10*pi]), grid on, %axis tight
    indx = find( linc > -10 & linc < 10);
    [slope1,~,~] = CircularLinearRegression(circ(temp),linc(temp),1);
    plot(x,2*pi*slope1*x,'b-','linewidth',2), text(40,50,['Phase precession: ', num2str(slope1)])
    %    ezcontour(@(x1,x2)pdf(GMModel,[x1 x2]),get(gca,{'XLim','YLim'}))
%     GMModel = fitgmdist([linc;circ]',1);
%     plot(GMModel.mu,'or')
%     slope2 = fitEllipse([linc;circ]'); 
    axis tight, 
%     text(40,-45,['Ellipsoid: ',num2str(slope2(1))])
    
    % Fitting a multi sinosoidal curve to the data
    x_out = linc;
    y_out = circ;
    x_bins = -40:0.8:40;
    y_bins = [-20:0.2:20];
    StartPoint = [0.1,6.28,40,20,0.00001,0,0];
    LowerLimits = [0,0,10,5,0,-1,-50];
    UpperLimits = [1,10,200,200,0.01,1,50];
    [fit_params,f03] = customFit(x_out,y_out,x_bins,y_bins,StartPoint,LowerLimits,UpperLimits);
    title(['Phase vs position: ' conditions{iii}])
    fit_params_all3(iii,:) = fit_params;
    f0s{iii,3} = f03;
    figure(10)
    for i = -5:5
        plot(x_bins,x_bins*fit_params(1)+i*fit_params(2),'--r')
    end
    ylim([-10,10]),xlim([-40,40])
    figure(20)
    subplot(1,3,3)
    for i = -5:5
        plot(x_bins,x_bins*fit_params(1)+i*fit_params(2),colors{iii}), hold on
    end
    ylim([-10,10]),xlim([-40,40]),title('Phase')
    
    figure(11)
    subplot(3,2,5), hold on
    xbins = [-30:0.05:30]';
    [N,edges] = histcounts(circ(indx),xbins,'normalization','probability'); 
    plt1(iii) = plot(xbins(1:end-1),nanconv(N,gausswin(20)','edge')); hold on
    title('Time lag between theta cycles (phase, from position)'), plot(0,0,'.'), axis tight,
    subplot(3,2,6), hold on
    x_interval = 20;
    for j = 1:5
        indx = find( linc > -60+j*x_interval & linc < -40+j*x_interval);
        [N,edges] = histcounts(circ(indx),xbins); 
        [pks_temp3,locs_temp3] =  findpeaks(nanconv(N,gausswin(50)','edge'));
        locs_temp3(pks_temp3<0.4) = [];
        pks_temp3(pks_temp3<0.4) = [];
        plot(-70+j*x_interval*ones(length(locs_temp3),1)'+iii*2,xbins(locs_temp3)',['.',colors{iii}]), hold on
    end
end
subplot(3,2,5)
legend(plt1,conditions)

fit_xticklabels = {'Slope','Y period','X envelope','Y envelope','Amplitude','xy shift','Y offset'};
colors2 = {'r','b','g'};
figure
% fit_params_all11 = [fit_params_all1(1,:);confint(f0s{1,1});fit_params_all1(2,:);confint(f0s{2,1});fit_params_all1(3,:);confint(f0s{3,1})]';
fit_params_all11 = [fit_params_all1(1,:);confint(f0s{1,1});fit_params_all1(2,:);confint(f0s{2,1})]';
for j = 1:length(conditions)
    for i = 1:7
        subplot(3,7,i)
        temp = fit_params_all11(i,[1:3]+(j-1)*3);
        plot(j,temp([2,3]),['o-',colors2{j}]); hold on
        plot(j,temp(1),'x','color',colors2{j},'linewidth',2), 
        title(fit_xticklabels{i}), xticks([1:length(conditions)]), xticklabels(conditions)
    end
end

% fit_params_all22 = [fit_params_all2(1,:);confint(f0s{1,2});fit_params_all2(2,:);confint(f0s{2,2});fit_params_all2(3,:);confint(f0s{3,2})]';
fit_params_all22 = [fit_params_all2(1,:);confint(f0s{1,2});fit_params_all2(2,:);confint(f0s{2,2})]';
for j = 1:length(conditions)
    for i = 1:7
        subplot(3,7,i+7)
        temp = fit_params_all22(i,[1:3]+(j-1)*3);
        plot(j,temp([2,3]),['o-',colors2{j}]); hold on
        plot(j,temp(1),'x','color',colors2{j},'linewidth',2), 
        title(fit_xticklabels{i}), xticks([1:length(conditions)]), xticklabels(conditions)
    end
end

% fit_params_all33 = [fit_params_all3(1,:);confint(f0s{1,3});fit_params_all3(2,:);confint(f0s{2,3});fit_params_all3(3,:);confint(f0s{3,3})]';
fit_params_all33 = [fit_params_all3(1,:);confint(f0s{1,3});fit_params_all3(2,:);confint(f0s{2,3})]';
for j = 1:length(conditions)
    for i = 1:7
        subplot(3,7,i+14)
        temp = fit_params_all33(i,[1:3]+(j-1)*3);
        plot(j,temp([2,3]),['o-',colors2{j}]); hold on
        plot(j,temp(1),'x','color',colors2{j},'linewidth',2), 
        title(fit_xticklabels{i}), xticks([1:length(conditions)]), xticklabels(conditions)
    end
end
save(['Z:\peterp03\IntanData\Analysis\PlaceFieldCompressionAnalysis2_',num2str(id),'.mat'],'fit_params_all11','fit_params_all22','fit_params_all33')
disp('Saved PlaceFieldCompressionAnalysis2')
%%
fit_params_all11 = [];
fit_params_all22 = [];
fit_params_all33 = [];
idsToLoad = [126,127,140,93,78,81,168,166,151,149]; % 79, 80

for i = 1:length(idsToLoad)
    temp = load(['Z:\peterp03\IntanData\Analysis\PlaceFieldCompressionAnalysis2_',num2str(idsToLoad(i)),'.mat']);
    fit_params_all11 = [fit_params_all11,temp.fit_params_all11(:,[1,4])];
    fit_params_all22 = [fit_params_all22,temp.fit_params_all22(:,[1,4])];
    fit_params_all33 = [fit_params_all33,temp.fit_params_all33(:,[1,4])];
end

fit_xticklabels = {'Slope','Y period','X envelope','Y envelope','Amplitude','xy shift','Y offset'};
colors2 = {'r','b','g'};
idx = [1:2:length(idsToLoad)*2];
figure
for i = 1:7
    subplot(3,7,i)
    for j = 1:length(conditions)
        temp = fit_params_all11(i,idx+j-1);
        plot(j,temp,'.','color',colors2{j},'linewidth',2), hold on
        title(fit_xticklabels{i}), xticks([1,2]), xticklabels(conditions)
    end
    axis tight
    data = reshape(fit_params_all11(i,:),2,[]);
    plot(data,'k')
    [h1,p1] = signrank(data(1,:),data(2,:));
    temp = get(gca,'ylim');
    h1 = text(1.1,temp(2),['h=', num2str(h1),',p=',num2str(p1)])
    set(h1,'Rotation',30)
end

for i = 1:7
    for j = 1:length(conditions)
        subplot(3,7,i+7)
        temp = fit_params_all22(i,idx+j-1);
        plot(j,temp,'.','color',colors2{j},'linewidth',2), hold on
        title(fit_xticklabels{i}), xticks([1,2,3]), xticklabels(conditions)
    end
        axis tight
    data = reshape(fit_params_all22(i,:),2,[]);
    plot(data,'k')
    [h1,p1] = signrank(data(1,:),data(2,:));
    temp = get(gca,'ylim');
    h1 = text(1.1,temp(2),['h=', num2str(h1),',p=',num2str(p1)])
    set(h1,'Rotation',30); 
end

for i = 1:7
    for j = 1:length(conditions)
        subplot(3,7,i+14)
        temp = fit_params_all33(i,idx+j-1);
        diff(temp)
        plot(j,temp,'.','color',colors2{j},'linewidth',2), hold on
        title(fit_xticklabels{i}), xticks([1,2,3]), xticklabels(conditions)
    end
        axis tight
    data = reshape(fit_params_all33(i,:),2,[]);
    plot(data,'k')
    [h1,p1] = signrank(data(1,:),data(2,:));
    temp = get(gca,'ylim');
    h1 = text(1.1,temp(2),['h=', num2str(h1),',p=',num2str(p1)])
    set(h1,'Rotation',30);
end

%% Average CCG for the three states

figure % Time
for j = 1:length(conditions)
    temp = [];
    temp2 = [];
    idx_offset = [];
    idx_offset2 = [];
    
    temp = placefield_ccgs_time{j}(find(sum(placefield_ccgs_time{j},2)>0),:);
    for iii = 1:size(temp,1)
        [pks_temp3,locs_temp3] =  findpeaks(temp(iii,:));
        locs_temp3(temp(iii,locs_temp3)<0.5) = [];
        locs_temp3(locs_temp3<16) = []; locs_temp3(locs_temp3>length(temp(iii,:))-16) = [];
        locs_temp3 = locs_temp3(find(temp(iii,locs_temp3+15)<temp(iii,locs_temp3) & temp(iii,locs_temp3-15)<temp(iii,locs_temp3)));
        
        t03 = 600;
        if any(locs_temp3<=t03) & any(locs_temp3>t03)
            [~,idx] = sort(abs(locs_temp3-t03));
            temp333 = find(locs_temp3<=t03);
            temp333 = locs_temp3(temp333(end));
            temp334 = locs_temp3(find(locs_temp3>t03,1));
            if temp(iii,temp333) >= temp(iii,temp334)
                idx2 = temp333-t03;
            else
                idx2 = temp334-t03;
            end
        end
        if abs(idx2)<100
            temp2(:,iii) = temp(iii,100+idx2:end+idx2-100);
            idx_offset2(iii) = idx2;
        end
        [~,idx_offset(iii)] = max(temp(iii,:));
        
        
        
%         [~,idx] = min(abs(ccg_delay_time_peaks{j}{iii}));
%         idx2 = ccg_delay_time_peaks{j}{iii}(idx);
%         if abs(idx2)<100
%             temp2(:,iii) = temp(iii,100+idx2:end+idx2-101);
%         end
%         idx_offset(iii) = idx2;
    end
%     Arowmax = max(temp2', [], 2);
%     [~,idx] = sort(Arowmax, 'descend'); % By ampltude
    if j ==1
        [~,idx3] = sort(idx_offset, 'descend'); % By offset
    end
    temp2 = temp2(:,idx3);
    temp2 = temp2./max(temp2);
    subplot(2,3,j)
    imagesc([-500:500],[1:size(temp2,2)],temp2')
    title(conditions{j}), xlabel('Time (ms)')
    subplot(2,2,3)
    plot([-500:501],nanmean(temp2')/max(nanmean(temp2')),'color',colors2{j}), hold on
    title('Time'), 
    subplot(2,2,4)
    plot(idx_offset2,idx_offset,'.','color',colors2{j}), hold on, axis tight
end
subplot(2,2,3)
legend(conditions), axis tight, grid on

figure % Phase
for j = 1:length(conditions)
    temp = [];
    temp2 = [];
    idx_offset = [];
    idx_offset2 = [];
    
    temp = placefield_ccgs_phase{j}(find(sum(placefield_ccgs_phase{j},2)>0),:);
    for iii = 1:size(temp,1)
        [pks_temp3,locs_temp3] =  findpeaks(temp(iii,:));
        locs_temp3(temp(iii,locs_temp3)<0.8) = [];
        locs_temp3(locs_temp3<16) = []; locs_temp3(locs_temp3>length(temp(iii,:))-16) = [];
        locs_temp3 = locs_temp3(find(temp(iii,locs_temp3+15)<temp(iii,locs_temp3) & temp(iii,locs_temp3-15)<temp(iii,locs_temp3)));
        
        t03 = 500;
        if any(locs_temp3<=t03) & any(locs_temp3>t03)
            [~,idx] = sort(abs(locs_temp3-t03));
            temp333 = find(locs_temp3<=t03);
            temp333 = locs_temp3(temp333(end));
            temp334 = locs_temp3(find(locs_temp3>t03,1));
            if temp(iii,temp333) > temp(iii,temp334)
                idx2 = temp333-500;
            else
                idx2 = temp334-500;
            end
        end
        if abs(idx2)<100
            temp2(:,iii) = temp(iii,100+idx2:end+idx2-100);
            idx_offset2(iii) = idx2;
        end
        [~,idx_offset(iii)] = max(temp(iii,:));
    end
%     Arowmax = max(temp2', [], 2);
%     [~,idx] = sort(Arowmax, 'descend'); % Amplitude
    if j ==1
    [~,idx3] = sort(idx_offset, 'descend'); % By offset
    end
    temp2 = temp2(:,idx3);

    temp2 = temp2./max(temp2);
    subplot(2,3,j)
    imagesc([-400:401]*0.0628,[1:size(temp2,2)],temp2'),
    title(conditions{j}), xlabel('Phase')
    
    subplot(2,2,3)
    plot([-400:401]*0.0628,nanmean(temp2')/max(nanmean(temp2')),'color',colors2{j}), hold on
    title('Phase'), 
    
    subplot(2,2,4)
    plot(idx_offset2,idx_offset,'.','color',colors2{j}), hold on, axis tight
end
subplot(2,2,3)
legend(conditions), axis tight, grid on

%% Temporal compression from the CCGs for the three states

% Time
for j = 1:3
    temp = [];
    temp2 = [];
    idx_offset = [];
    idx_offset2 = [];
    idx_offset3 = [];
    
    temp = placefield_ccgs_time{j}(find(sum(placefield_ccgs_time{j},2)>0),:);
    for iii = 1:size(temp,1)
        [pks_temp3,locs_temp3] =  findpeaks(temp(iii,:));
        locs_temp3(temp(iii,locs_temp3)<0.5) = [];
        locs_temp3(locs_temp3<16) = []; locs_temp3(locs_temp3>length(temp(iii,:))-16) = [];
        locs_temp3 = locs_temp3(find(temp(iii,locs_temp3+15)<temp(iii,locs_temp3) & temp(iii,locs_temp3-15)<temp(iii,locs_temp3)));
        
        t03 = 600;
        if any(locs_temp3<=t03) & any(locs_temp3>t03)
            [~,idx] = sort(abs(locs_temp3-t03));
            temp333 = find(locs_temp3<=t03);
            temp333 = locs_temp3(temp333(end));
            temp334 = locs_temp3(find(locs_temp3>t03,1));
            if temp(iii,temp333) > temp(iii,temp334)
                idx2 = temp333-t03;
            else
                idx2 = temp334-t03;
            end
        end
        if abs(idx2)<100
            temp2(:,iii) = temp(iii,100+idx2:end+idx2-100);
            idx_offset2(iii) = idx2;
            idx_offset3(iii) = idx2;
        else
            idx_offset3(iii) = -100000;
        end
        [~,idx_offset(iii)] = max(temp(iii,:));
        
        
        
%         [~,idx] = min(abs(ccg_delay_time_peaks{j}{iii}));
%         idx2 = ccg_delay_time_peaks{j}{iii}(idx);
%         if abs(idx2)<100
%             temp2(:,iii) = temp(iii,100+idx2:end+idx2-101);
%         end
%         idx_offset(iii) = idx2;
    end
    
    % By ampltude
%     Arowmax = max(temp2', [], 2);
%     [~,idx] = sort(Arowmax, 'descend'); 

    % By offset
%     if j ==1
        [~,idx3] = sort(idx_offset); 
%     end
    
    % By spatial offset
    [test,idx4] = sort(placefield_difference{j});
    
     % By temperal offset

    
    % By spatio temporal offset (correct by speed)
    [test,idx5] = sort(placefield_time_offset{j}.*placefield_speed{j});
    
    % by zero peak
    [test,idx6] = sort(idx_offset3);
    
    % sort by xcorr
%     r = corrcoef(temp');
%     B = r > 0.90;
%     idx7 = symrcm(B);
    
    temp2 = temp2(:,idx3);
    temp2 = temp2./max(temp2);
    figure(100)
    subplot(2,3,j)
    imagesc([1:size(temp2,2)],[-500:500],temp2)
    title(conditions{j}), ylabel('Time (ms)')
    subplot(2,2,3)
    plot([-500:501],nanmean(temp2')/max(nanmean(temp2')),'color',colors2{j}), hold on
    title('Time'), 
    figure(101)
    subplot(1,3,j)
    imagesc([1:size(temp,2)],[-500:500],temp(idx6,:)'./max(temp(idx6,:)')), grid on
    title(conditions{j}), xlabel('Cell pairs ordered by temporal offset'), ylabel('Time (theta cycles; ms)'),set(gca,'Ydir','normal')
end
figure(100)
subplot(2,2,3)
legend(conditions), axis tight, grid on

% Phase
for j = 1:3
    temp = [];
    temp2 = [];
    idx_offset = [];
    idx_offset2 = [];
    idx_offset3 = [];
    temp = placefield_ccgs_phase{j}(find(sum(placefield_ccgs_phase{j},2)>0),:);
    for iii = 1:size(temp,1)
        [pks_temp3,locs_temp3] =  findpeaks(temp(iii,:));
        locs_temp3(temp(iii,locs_temp3)<0.8) = [];
        locs_temp3(locs_temp3<16) = []; locs_temp3(locs_temp3>length(temp(iii,:))-16) = [];
        locs_temp3 = locs_temp3(find(temp(iii,locs_temp3+15)<temp(iii,locs_temp3) & temp(iii,locs_temp3-15)<temp(iii,locs_temp3)));
        
        t03 = 500;
        if any(locs_temp3<=t03) & any(locs_temp3>t03)
            [~,idx] = sort(abs(locs_temp3-t03));
            temp333 = find(locs_temp3<=t03);
            temp333 = locs_temp3(temp333(end));
            temp334 = locs_temp3(find(locs_temp3>t03,1));
            if temp(iii,temp333) > temp(iii,temp334)
                idx2 = temp333-500;
            else
                idx2 = temp334-500;
            end
        end
        if abs(idx2)<100
            temp2(:,iii) = temp(iii,100+idx2:end+idx2-100);
            idx_offset2(iii) = idx2;
            idx_offset3(iii) = idx2;
        else
            idx_offset3(iii) = -100000;
        end
        [~,idx_offset(iii)] = max(temp(iii,:));
    end
%     Arowmax = max(temp2', [], 2);
%     [~,idx] = sort(Arowmax, 'descend'); % Amplitude
%     if j ==1
    [~,idx3] = sort(idx_offset); % By offset
%     end
    
    % By spatial offset
    [test,idx4] = sort(placefield_difference{j});
    
     % By temperal offset
%     [test,idx4] = sort(placefield_time_offset{j});
    
    % By spatio temporal offset (correct by speed)
    [test,idx5] = sort(placefield_time_offset{j}.*placefield_speed{j});
%     placefield_ccgs_phase

    % by zero peak
    [test,idx6] = sort(idx_offset3);
    
    % sort by xcorr
%     r = corrcoef(temp');
%     B = r > 0.90;
%     idx7 = symrcm(B);
    
    temp2 = temp2(:,idx3);

    temp2 = temp2./max(temp2);
    
    figure(102)
    subplot(2,3,j)
    imagesc([-400:401]*0.0628,[1:size(temp2,2)],temp2'),
    title(conditions{j}), xlabel('Phase')
    subplot(2,2,3)
    plot([-400:401]*0.0628,nanmean(temp2')/max(nanmean(temp2')),'color',colors2{j}), hold on
    title('Phase'), 
    subplot(2,2,4)
    plot(idx_offset2,idx_offset,'.','color',colors2{j}), hold on, axis tight
    
    figure(103)
    subplot(1,3,j)
    imagesc([1:size(temp,2)],[-500:500],temp(idx6,:)'./max(temp(idx6,:)')), grid on
    title(conditions{j}), xlabel('Cell pairs ordered by temporal offset'), ylabel('Phase'),set(gca,'Ydir','normal')
end
figure(102)
subplot(2,2,3)
legend(conditions), axis tight, grid on
%%
figure, 
subplot(1,3,1)
imagesc(r)
subplot(1,3,2)
imagesc(r(idx6,:))
B = r > 0.80;
p = symrcm(B);
r2 = r(p, p);
subplot(1,3,3)
imagesc(r2)
%% % ACGs

 figure
for j = 1:3
%     
%     for iii = 1:size(temp,1)
%         [pks_temp3,locs_temp3] =  findpeaks(temp(iii,:));
%         locs_temp3(temp(iii,locs_temp3)<0.8) = [];
%         locs_temp3(locs_temp3<16) = []; locs_temp3(locs_temp3>length(temp(iii,:))-16) = [];
%         locs_temp3 = locs_temp3(find(temp(iii,locs_temp3+15)<temp(iii,locs_temp3) & temp(iii,locs_temp3-15)<temp(iii,locs_temp3)));
%         
%         t03 = 500;
%         if any(locs_temp3<=t03) & any(locs_temp3>t03)
%             [~,idx] = sort(abs(locs_temp3-t03));
%             temp333 = find(locs_temp3<=t03);
%             temp333 = locs_temp3(temp333(end));
%             temp334 = locs_temp3(find(locs_temp3>t03,1));
%             if temp(iii,temp333) > temp(iii,temp334)
%                 idx2 = temp333-500;
%             else
%                 idx2 = temp334-500;
%             end
%         end
%         if abs(idx2)<100
%             temp2(:,iii) = temp(iii,100+idx2:end+idx2-100);
%         end
%     end
    Arowmax = max(ACGs{j}, [], 2);
    [~,idx] = sort(Arowmax, 'descend'); % Amplitude
%     [~,idx] = sort(idx_offset, 'descend'); % By offset
    temp2 = ACGs{j}(idx,:)';
    
%     temp2 = sort(ACGs{j}',2);
    
    temp2 = temp2./max(temp2);
    subplot(2,3,j)
    imagesc([-600:600],[1:size(temp2,2)],temp2'),
    title(conditions{j}), xlabel('Time (ms)'), xlim([-500,500])
    subplot(2,1,2)
    plot([-600:600],nanmean(temp2')/max(nanmean(temp2'))), hold on, xlim([-500,500])
    title('ACGs - time'), xlabel('Time (ms)'),
end
legend(conditions), axis tight, grid on, xlim([-500,500])

%% % CCG between place fields and interneurons for the three states
figure
for j = 1:3
    temp = [];
    temp2 = [];
    idx_offset = [];
    idx_offset2 = [];
    temp = placefield_interneurons_ccgs_time{j}(find(sum(placefield_interneurons_ccgs_time{j},2)>0),:);
    for iii = 1:size(temp,1)
        [pks_temp3,locs_temp3] =  findpeaks(temp(iii,:));
        locs_temp3(temp(iii,locs_temp3)<0.5) = [];
        locs_temp3(locs_temp3<16) = []; locs_temp3(locs_temp3>length(temp(iii,:))-16) = [];
        locs_temp3 = locs_temp3(find(temp(iii,locs_temp3+15)<temp(iii,locs_temp3) & temp(iii,locs_temp3-15)<temp(iii,locs_temp3)));
        
        t03 = 600;
        if any(locs_temp3<=t03) & any(locs_temp3>t03)
            [~,idx] = sort(abs(locs_temp3-t03));
            temp333 = find(locs_temp3<=t03);
            temp333 = locs_temp3(temp333(end));
            temp334 = locs_temp3(find(locs_temp3>t03,1));
            if temp(iii,temp333) > temp(iii,temp334)
                idx2 = temp333-t03;
            else
                idx2 = temp334-t03;
            end
        end
        if abs(idx2)<100
            temp2(:,iii) = temp(iii,100+idx2:end+idx2-100);
            idx_offset2(iii) = idx2;
            [~,idx_offset(iii)] = max(temp(iii,:));
        end
        
%         [~,idx] = min(abs(ccg_delay_time_peaks{j}{iii}));
%         idx2 = ccg_delay_time_peaks{j}{iii}(idx);
%         if abs(idx2)<100
%             temp2(:,iii) = temp(iii,100+idx2:end+idx2-101);
%         end
%         idx_offset(iii) = idx2;
    end
    Arowmax = max(temp2', [], 2);
%     [~,idx] = sort(Arowmax, 'descend'); % By ampltude
    [~,idx] = sort(idx_offset, 'descend'); % By offset
    temp2 = temp2(:,idx);

    temp2 = temp2./max(temp2);
    subplot(2,3,j)
    imagesc([-500:500],[1:size(temp2,2)],temp2')
    title(conditions{j}), xlabel('Time (ms)'), axis tight
    subplot(2,2,3)
    plot([-500:501],nanmean(temp2')/max(nanmean(temp2')),'color',colors2{j}), hold on
    title('Time'), axis tight, grid on
    subplot(2,2,4)
    plot(idx_offset2,idx_offset,'.','color',colors2{j}), hold on, axis tight
end
subplot(2,2,3)
legend(conditions)
%%
xticks([-6*pi -4*pi -2*pi -0 2*pi 4*pi 6*pi])
xticklabels({'-6\pi','-4\pi','-2\pi','0','2\pi','4\pi','6\pi'})