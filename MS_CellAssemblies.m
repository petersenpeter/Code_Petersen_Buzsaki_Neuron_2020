% close all
% MS10: 61, 62, 63, 64 % gamma id: 62
% MS12: 78 (Peter_MS12_170714_122034_concat), 79 (Peter_MS12_170715_111545_concat), 80 (Peter_MS12_170716_172307_concat), 81 (Peter_MS12_170717_111614_concat),
% MS13: 92 (Peter_MS13_171129_105507_concat), 93 (Peter_MS13_171130_121758_concat)
% MS14: 
% MS18: 
% MS21: 126 (Peter_MS21_180629_110332_concat), 140 (Peter_MS21_180627_143449_concat), 143 (control: Peter_MS21_180719_155941_concat)
% MS22: 139 (Peter_MS22_180628_120341_concat), 127 (Peter_MS22_180629_110319_concat), 144 (control: Peter_MS22_180719_122813_concat)
% id = 126 % 62
ids = [61, 62,64,78,79,80,81,92,93,126,140,139,127]; %63
cellAssemblies = {};
for i = 1:length(ids)
    i
    cellAssemblies{i} = getCellAssemblies(ids(i));
end

%% Fraction of cells being active
figure
colors= {'r', 'b', 'g'};
temp23 = nan(length(ids),200);
for i = 1:length(ids)
    test332423 = [1,cumsum(cellfun(@length, {cellAssemblies{i}{1}.spike_length,cellAssemblies{i}{2}.spike_length,cellAssemblies{i}{3}.spike_length}))];
    temp1 =  mean(cellAssemblies{i}{1}.spike_length);
    temp2 =  std(cellAssemblies{i}{1}.spike_length);
    for j = 1:3
        subplot(3,1,1)
        plot(cellAssemblies{i}{j}.xbins,cellAssemblies{i}{j}.hist,'color',colors{j}), hold on
        subplot(3,1,2)
        temp23(i,[1:length(cellAssemblies{i}{j}.spike_length)]+test332423(j)) = (cellAssemblies{i}{j}.spike_length-temp1)/temp2;
        plot([1:length(cellAssemblies{i}{j}.spike_length)]+test332423(j),(cellAssemblies{i}{j}.spike_length-temp1)/temp2,'color',colors{j}), hold on
    end
    subplot(3,1,3)
    plot(i*[cellAssemblies{i}{1}.h,cellAssemblies{i}{2}.h,cellAssemblies{i}{3}.h]), hold on
end
subplot(3,1,2)
plot(nanmean(temp23),'k','linewidth',2), grid on
subplot(3,1,3)
xticks([1,2,3])
xticklabels({'Pre/Cooling','Cooling/Post','Pre/Post'})
subplot(3,1,1)
xlabel('Fraction of cells being active'), ylabel('Fraction of theta cycles'), title('Cell assemblies')
xlim([0,0.8])

%% Range of space being expressed per theta cycle
% ids = [61,62,64,78,79,80,81,92,93,126,140,139,127]; % 63
ids = [61,62,64,78,79,80,81,83,92,93,88,91,94,126,149,153,139,127,168,166,140];
cellAssembliesSpatailCoverage = {};
for i = 1:length(ids)
    i
    cellAssembliesSpatailCoverage{i} = getCellAssembliesSpatailCoverage(ids(i));
    drawnow
end

%% Continued
colors= {'r', 'b', 'g'};
bins_speed = [0:10:150];
states = {'Pre','Cooling','Post'};
for i = 1:length(ids)
    figure
    for jj = 1:3
        subplot(2,1,1)
        plot(cellAssembliesSpatailCoverage{i}.position{jj},cellAssembliesSpatailCoverage{i}.speed{jj},'.','color',colors{jj}), hold on
        subplot(2,1,2)
        N = histcounts(cellAssembliesSpatailCoverage{i}.speed{jj},bins_speed,'Normalization','probability');
        plot(bins_speed(1:end-1)-5,N,'color',colors{jj}), hold on
    end
    legend(states)
    
    figure(100)
    subplot(2,1,1), hold on
    plot([1,2,3],[mean(cellAssembliesSpatailCoverage{i}.speed{1}),mean(cellAssembliesSpatailCoverage{i}.speed{2}),mean(cellAssembliesSpatailCoverage{i}.speed{3})])
    subplot(2,1,2), hold on
    plot([1,2,3],[median(cellAssembliesSpatailCoverage{i}.p2{1}),median(cellAssembliesSpatailCoverage{i}.p2{2}),median(cellAssembliesSpatailCoverage{i}.p2{3})])
end

%% Pairwise distance being expressed between pyramidal cells
% ids = [61,62,64,78,79,80,81,92,93,126,140,139,127]; % 63
ids = [61,62,64,78,79,80,81,83,92,93,88,91,94,126,149,153,139,127,168,166,140];
cellAssembliesSpatailCoverage2 = {};
for i = 1:length(ids)
    i
    cellAssembliesSpatailCoverage2{i} = getCellAssembliesSpatailCoverage2(ids(i));
    drawnow
end

%%  Pairwise distance across session analysis
colors = {'r', 'b', 'g'};
colors2 = [1,0,0;0,0,1;0,1,0];
states = {'Pre','Cooling','Post'};

bins_position = [0:20:350];
bins_speed = [0:5:150];
bins_temporal = [0:4:200];
bins_phase = [0:0.1:2*pi];
bins_assembly_size = [0:0.005:0.3];
step_size = 0.003;
bins_speed2 = [0.05:step_size:0.2];
data1 = zeros(length(bins_speed2)-1,3,length(ids));
data2 = zeros(length(bins_temporal)-1,3,length(ids));
data3 = zeros(length(bins_assembly_size)-1,3,length(ids));
data4 = zeros(length(bins_phase)-1,3,length(ids));
data5 = zeros(length(bins_speed)-1,3,length(ids));
data6 = nan(200,3,length(ids));
data1_1 = [];
data2_1 = [];
data3_1 = [];
data4_1 = [];
data5_1 = [];
data6_1 = [];

fig=figure;
set(fig,'defaultAxesColorOrder',colors2)
for i = 1:length(ids)
    for jj = 1:3
        subplot(6,2,1) % Time
        plot(bins_speed2(1:end-1)+step_size/2,cellAssembliesSpatailCoverage2{i}.hist_speed_normalized{jj},'color',colors{jj}), hold on
        title('Time (sec)'), axis tight
        data1(:,jj,i) = cellAssembliesSpatailCoverage2{i}.hist_speed_normalized{jj};
        data1_1(jj,i) = nanmean(cellAssembliesSpatailCoverage2{i}.slope_temporal_phase{jj}(find(cellAssembliesSpatailCoverage2{i}.slope_temporal_phase{jj}>0.06 & cellAssembliesSpatailCoverage2{i}.slope_temporal_phase{jj}<0.2 & ~isnan(cellAssembliesSpatailCoverage2{i}.slope_temporal_phase{jj}))));
        
        subplot(6,2,3) % Temporal offset
        plot(bins_temporal(1:end-1),cumsum_normalized(cellAssembliesSpatailCoverage2{i}.cumsum_temporal_offset{jj}),'color',colors{jj}), hold on
        title('Temporal offset'), axis tight
        data2(:,jj,i) = cumsum_normalized(cellAssembliesSpatailCoverage2{i}.cumsum_temporal_offset{jj});
        data2_1(jj,i) = nanmean(cellAssembliesSpatailCoverage2{i}.temporal_offset{jj});
        
        subplot(6,2,5) % Assembly size
        plot(bins_assembly_size(1:end-1),cumsum_normalized(cellAssembliesSpatailCoverage2{i}.cumsum_assembly_size{jj}),'color',colors{jj}), hold on
        title('Assembly size'), axis tight
        data3(:,jj,i) = cumsum_normalized(cellAssembliesSpatailCoverage2{i}.cumsum_assembly_size{jj});
        data3_1(jj,i) = nanmean(cellAssembliesSpatailCoverage2{i}.assembly_size{jj});
        
        subplot(6,2,7) % Phase offset
        plot(bins_phase(1:end-1), cumsum_normalized(cellAssembliesSpatailCoverage2{i}.cumsum_phase_offset{jj}),'color',colors{jj}), hold on
        title('Phase offset'), axis tight
        data4(:,jj,i) = cumsum_normalized(cellAssembliesSpatailCoverage2{i}.cumsum_phase_offset{jj});
        data4_1(jj,i) = nanmean(cellAssembliesSpatailCoverage2{i}.phase_offset{jj});
        
        subplot(6,2,9) % Speed
        
        N = histcounts(cellAssembliesSpatailCoverage2{i}.speed{jj},bins_speed,'Normalization','probability');
        plot(bins_speed(1:end-1), N,'color',colors{jj}), hold on
        title('Running speed'), axis tight
        data5(:,jj,i) = N;
        speed_limits = [0,140];
        data5_1(jj,i) = nanmean(cellAssembliesSpatailCoverage2{i}.speed{jj}(find(cellAssembliesSpatailCoverage2{i}.speed{jj} > speed_limits(1) & cellAssembliesSpatailCoverage2{i}.speed{jj} < speed_limits(2))));
        
        subplot(6,2,11) % Spike count
        plot(cellAssembliesSpatailCoverage2{i}.cumsum_spike_count{jj},'color',colors{jj}), hold on
        title('Spike count'), axis tight
        data6([1:length(cellAssembliesSpatailCoverage2{i}.cumsum_spike_count{jj})],jj,i) = cellAssembliesSpatailCoverage2{i}.cumsum_spike_count{jj};
        spike_count_limits = [0,80];
        data6_1(jj,i) = nanmean(cellAssembliesSpatailCoverage2{i}.spike_count{jj}(find(cellAssembliesSpatailCoverage2{i}.spike_count{jj} > spike_count_limits(1) & cellAssembliesSpatailCoverage2{i}.spike_count{jj} < spike_count_limits(2) )));
    end
end

subplot(6,2,2) % Speed
plot_error_patch(bins_speed2(1:end-1)+step_size/2,data1,colors,data1_1)
title('Time (s)'), axis tight
subplot(6,2,4) % Temporal offset
plot_error_patch(bins_temporal(1:end-1),data2,colors,data2_1)
title('Temporal offset'), axis tight, %xlim([0,120])
subplot(6,2,6) % Assembly size
plot_error_patch(bins_assembly_size(1:end-1),data3,colors,data3_1)
title('Assembly size'), axis tight
subplot(6,2,8) % Phase offset
plot_error_patch(bins_phase(1:end-1),data4,colors,data4_1)
title('Phase offset'), axis tight, xticks([-2*pi,-pi,0,pi,2*pi]), xticklabels({'-2\pi','-pi','0','pi','2\pi'})
subplot(6,2,10) % Speed
plot_error_patch(bins_speed(1:end-1),data5,colors,data5_1)
title('Running speed (cm/s)'), axis tight
subplot(6,2,12) % Spike count
plot_error_patch([1:200],data6,colors,data6_1)
title('Spike count'), axis tight, %xlim([0,70])


%% Population vector analysis
ids = [61,62,64,78,79,80,81,83,92,93,88,91,94,126,149,153,139,127,168,166,140];
PopulationVectorAnalysisLength = [];
for i = 1:length(ids)
    i
    PopulationVectorAnalysisLength(:,i) = PopulationVectorAnalysis(ids(i));
end
figure, 
subplot(2,1,1)
plot([1,2,3]'*ones(1,length(ids))+[1:length(ids)]/300,PopulationVectorAnalysisLength+rand(size(PopulationVectorAnalysisLength))/4,'o-'), xlim([0.8,3.2]), ylim([10,40])
trials_errors = PopulationVectorAnalysisLength;
[p1,h1] = signrank(trials_errors(1,:),trials_errors(2,:));
[p2,h2] = signrank(trials_errors(2,:),trials_errors(3,:));
[p3,h3] = signrank(trials_errors(1,:),trials_errors(3,:));
text(1.2,40,[num2str(p1),',  ',num2str(h1)],'Rotation',45);
text(2.8,40,[num2str(p2),'  ,',num2str(h2)],'Rotation',45);
text(2,40,[num2str(p3),',  ',num2str(h3)],'Rotation',45);
subplot(2,1,2)
x_data = mean(PopulationVectorAnalysisLength');
x_std = std(PopulationVectorAnalysisLength');
plot(x_data,'k','linewidth',2), hold on
plot(x_data+x_std,'-ok')
plot(x_data-x_std,'-ok'), xlim([0.8,3.2]), ylim([10,40])
path = 'K:\Dropbox\Buzsakilab Postdoc\MatlabFigures\PopulationVector';
print(gcf, [path,'\summary'],'-dpdf')