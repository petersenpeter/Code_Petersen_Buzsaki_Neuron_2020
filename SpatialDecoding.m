% Decoding animal position from theta lfp and spikes

% MS10: 61 (Peter_MS10_170317_153237_concat) OK
%       62 (Peter_MS10_170314_163038) % gamma id: 62 OK
%       63 (Peter_MS10_170315_123936) 
%       64 (Peter_MS10_170307_154746_concat) OK 
% MS12: 78 (Peter_MS12_170714_122034_concat) OK 
%       79 (Peter_MS12_170715_111545_concat) OK 
%       80 (Peter_MS12_170716_172307_concat) OK 
%       81 (Peter_MS12_170717_111614_concat) OK 
%       83 (Peter_MS12_170719_095305_concat) OK 
% MS13: 92 (Peter_MS13_171129_105507_concat) OK 
%       93 (Peter_MS13_171130_121758_concat) OK 
%       88 (Peter_MS13_171110_163224_concat) No good cooling behavior
%       91 (Peter_MS13_171128_113924_concat) OK 
%       94 (Peter_MS13_171201_130527_concat) OK 
% MS21: 126 (Peter_MS21_180629_110332_concat) OK 
%       140 (Peter_MS21_180627_143449_concat) OK 
%       143 (Peter_MS21_180719_155941_concat, control)
%       149 (Peter_MS21_180625_153927_concat) OK 
%       153 (Peter_MS21_180712_103200_concat) OK 
%       151 (Peter_MS21_180628_155921_concat) OK 
%       154 (Peter_MS21_180719_122733_concat, control, duplicated)
%       159 (Peter_MS21_180807_122213_concat, control)
% MS22: 139 (Peter_MS22_180628_120341_concat) OK 
%       127 (Peter_MS22_180629_110319_concat) OK 
%       144 (Peter_MS22_180719_122813_concat, control)
%       168 (Peter_MS22_180720_110055_concat) OK 
%       166 (Peter_MS22_180711_112912_concat) OK 

sessionNames = {'Peter_MS22_180629_110319_concat','Peter_MS21_180627_143449_concat','Peter_MS21_180629_110332_concat'};
iii = 2;
[session, basename, basepath, clusteringpath] = db_set_session('sessionName',sessionNames{iii});

% Loading preprocessed data
speed_threshold = 20;
spikes = loadSpikes('session',session);
for i = 1:size(spikes.times,2)
    idx = find(spikes.speed{i} > speed_threshold);
    spikes.times{i} = spikes.times{i}(idx);
    spikes.pos_linearized{i} = spikes.pos_linearized{i}(idx);
    spikes.theta_phase{i} = spikes.theta_phase{i}(idx);
    spikes.trials{i} = spikes.trials{i}(idx);
end
% load([basename, '.temperature.timeseries.mat'])
% load([basename, '.cooling.manipulation.mat'])
% load([basename, '.animal.behavior.mat'])
temperature = loadStruct('temperature','timeseries','session',session);
cooling = loadStruct('cooling','manipulation','session',session);
animal = loadStruct('animal','behavior','session',session);
trials = loadStruct('trials','behavior','session',session);

if length(trials.cooling) ~= length(trials.stat)
    trials.cooling = [trials.cooling,zeros(1,length(trials.stat) - length(trials.cooling))];
end
lfp5 = bz_GetLFP(session.channelTags.Theta.channels-1);

% TODO
% Take out interneurons
% Optimize smoothing
% Optize interation count

%% Loading and downsample LFP
lfp = bz_GetLFP('all','restrict',[trials.start/animal.sr-0.2;trials.end/animal.sr+0.2]');
lfp2 = {};
% lfp2.data = (downsample(vertcat(lfp.data),32));
% lfp2.timestamps = downsample(vertcat(lfp.timestamps),32);
lfp2.data = vertcat(lfp.data);
lfp2.timestamps = vertcat(lfp.timestamps);
lfp2.samplingRate = lfp(1).samplingRate;
LFPs = LFPsDemodulation(lfp2);

% figure, plot(lfp2.data(:,1))

%% bz_positionDecodingMaxCorr
positionDecodingMaxCorr1 = {};
smoothingRange1 = [30:5:50];
training_ratio = [0.8];
bin_size = [10]; % in ms

trials.intervals = animal.time([trials.start;trials.end]');
animal.pos_linearized_norm = ceil(100*animal.pos_linearized/(animal.pos_linearized_limits(2)));
conditions = {'Pre','Cooling','Post'};
for i = 1:length(bin_size)
    i
    smoothingRange = smoothingRange1(i);
%     training_ratio = training_ratio1(i);
%     bin_size = bin_size1(i);
    
    [positionDecodingMaxCorr1{i},confusionMatrix] = positionDecodingMaxCorr(spikes,animal,trials,lfp5,smoothingRange,'plotting',0,'saveMat',1,'training_ratio',training_ratio,'bin_size',bin_size);
    testLabels = {'Pre','Cooling','Post','Own'};
    figure('name',['Testing on own condition, smoothingRange=' num2str(smoothingRange),', training_ratio=',num2str(training_ratio),', bin_size=',num2str(bin_size)]),
    subplot(1,3,1)
    plot(positionDecodingMaxCorr1{i}.results.condition,positionDecodingMaxCorr1{i}.results.mse_rate,'o'), ylim([0,500])
    title('mse rate'), xticks([1 2 3]), xticklabels(conditions), xlim([0.7,3.3]), ylabel('mse')
    subplot(1,3,2)
    plot(positionDecodingMaxCorr1{i}.results.condition,positionDecodingMaxCorr1{i}.results.mse_phase_all,'o'), ylim([0,500])
    title('mse phase all'), xticks([1 2 3]), xticklabels(conditions), xlim([0.7,3.3])
    subplot(1,3,3)
    plot(positionDecodingMaxCorr1{i}.results.condition,positionDecodingMaxCorr1{i}.results.mse_chance,'o'), ylim([0,2500])
    title('mse chance'), xticks([1 2 3]), xticklabels(conditions), xlim([0.7,3.3])
    drawnow
end

figure,
for i = 1:length(smoothingRange1)
    subplot(1,3,1)
    plot(i,positionDecodingMaxCorr1{i}.results.mse_rate,'.'), hold on
    plot(i,median(positionDecodingMaxCorr1{i}.results.mse_rate),'ok')
    xticks(1:length(smoothingRange1)), xticklabels(strsplit(num2str(smoothingRange1)))
    subplot(1,3,2)
    plot(i,positionDecodingMaxCorr1{i}.results.mse_phase_all,'.'), hold on
    plot(i,median(positionDecodingMaxCorr1{i}.results.mse_phase_all),'ok')
    xticks(1:length(smoothingRange1)), xticklabels(strsplit(num2str(smoothingRange1)))
    subplot(1,3,3)
    plot(i,positionDecodingMaxCorr1{i}.results.mse_chance,'.'), hold on
    xticks(1:length(smoothingRange1)), xticklabels(strsplit(num2str(smoothingRange1)))
end
%%
positionDecodingMaxCorr1 = {};
smoothingRange = [40];
training_ratio = [0.8];
bin_size = [10]; % in ms
trials.intervals = animal.time([trials.start;trials.end]');
animal.pos_linearized_norm = ceil(100*animal.pos_linearized/(animal.pos_linearized_limits(2)));
positionDecodingMaxCorr1{4} = positionDecodingMaxCorr(spikes,animal,trials,lfp5,smoothingRange,'plotting',0,'saveMat',1,'training_ratio',training_ratio,'bin_size',bin_size);
testLabels = {'Pre','Cooling','Post','Own'};
figure('name',['Testing on own condition, smoothingRange=' num2str(smoothingRange),', training_ratio=',num2str(training_ratio),', bin_size=',num2str(bin_size)]),
conditions = {'Pre','Cooling','Post'};
subplot(1,3,1)
plot(positionDecodingMaxCorr1{4}.results.condition,positionDecodingMaxCorr1{4}.results.mse_rate,'o'), ylim([0,500])
title('mse rate'), xticks([1 2 3]), xticklabels(conditions), xlim([0.7,3.3]), ylabel('mse')
subplot(1,3,2)
plot(positionDecodingMaxCorr1{4}.results.condition,positionDecodingMaxCorr1{4}.results.mse_phase_all,'o'), ylim([0,500])
title('mse phase all'), xticks([1 2 3]), xticklabels(conditions), xlim([0.7,3.3])
subplot(1,3,3)
plot(positionDecodingMaxCorr1{4}.results.condition,positionDecodingMaxCorr1{4}.results.mse_chance,'o'), ylim([0,2500])
title('mse chance'), xticks([1 2 3]), xticklabels(conditions), xlim([0.7,3.3])
drawnow
    
for j = 1:3
    j
    trials.intervals = animal.time([trials.start;trials.end]');
    animal.pos_linearized_norm = ceil(100*animal.pos_linearized/(animal.pos_linearized_limits(2)));
    positionDecodingMaxCorr1{j} = positionDecodingMaxCorr(spikes,animal,trials,lfp5,smoothingRange,'plotting',0,'saveMat',1,'testCondition',j,'training_ratio',training_ratio,'bin_size',bin_size);
    
    figure('name',['Testing on ',testLabels{j},' condition, smoothingRange = ' num2str(smoothingRange),', training_ratio=',num2str(training_ratio),', bin_size=',num2str(bin_size)]),
    conditions = {'Pre','Cooling','Post'};
    subplot(1,3,1)
    plot(positionDecodingMaxCorr1{j}.results.condition,positionDecodingMaxCorr1{j}.results.mse_rate,'o'), ylim([0,2500])
    title('mse rate'), xticks([1 2 3]), xticklabels(conditions), xlim([0.7,3.3]), ylabel('mse')
    subplot(1,3,2)
    plot(positionDecodingMaxCorr1{j}.results.condition,positionDecodingMaxCorr1{j}.results.mse_phase_all,'o'), ylim([0,2500])
    title('mse phase all'), xticks([1 2 3]), xticklabels(conditions), xlim([0.7,3.3])
    subplot(1,3,3)
    plot(positionDecodingMaxCorr1{j}.results.condition,positionDecodingMaxCorr1{j}.results.mse_chance,'o'), ylim([0,2500])
    title('mse chance'), xticks([1 2 3]), xticklabels(conditions), xlim([0.7,3.3])
end

% Decoding matrix
figure,
colorrange = [400,2000];
% mse_rate
decodingPerformaceMatrix_mse_rate = [];
for j = 1:4
    for i = 1:3
        decodingPerformaceMatrix_mse_rate(j,i) = mean(positionDecodingMaxCorr1{j}.results.mse_rate(find(positionDecodingMaxCorr1{j}.results.condition==i)));
    end
end
subplot(2,3,1), imagesc(decodingPerformaceMatrix_mse_rate)
xticks([1 2 3]), xticklabels(conditions), yticks([1 2 3 4]), yticklabels(testLabels), clim(colorrange), title('mse rate')
% mse_phase_all
decodingPerformaceMatrix_mse_phase_all = [];
for j = 1:4
    for i = 1:3
        decodingPerformaceMatrix_mse_phase_all(j,i) = mean(positionDecodingMaxCorr1{j}.results.mse_phase_all(find(positionDecodingMaxCorr1{j}.results.condition==i)));
    end
end
subplot(2,3,2), imagesc(decodingPerformaceMatrix_mse_phase_all)
xticks([1 2 3]), xticklabels(conditions), yticks([]), clim(colorrange), title('mse phase all')
% mse_chance
decodingPerformaceMatrix_mse_chance = [];
for j = 1:4
    for i = 1:3
        decodingPerformaceMatrix_mse_chance(j,i) = mean(positionDecodingMaxCorr1{j}.results.mse_chance(find(positionDecodingMaxCorr1{j}.results.condition==i)));
    end
end
subplot(2,3,3), imagesc(decodingPerformaceMatrix_mse_chance)
xticks([1 2 3]), xticklabels(conditions), yticks([]), clim(colorrange), title('mse chance')
colorbar
subplot(2,3,4), plot(decodingPerformaceMatrix_mse_rate'), xticks([1 2 3]), xticklabels(conditions), ylim([0,2000])
subplot(2,3,5), plot(decodingPerformaceMatrix_mse_phase_all'), xticks([1 2 3]), xticklabels(conditions), ylim([0,2000])
subplot(2,3,6), plot(decodingPerformaceMatrix_mse_chance'), xticks([1 2 3]), xticklabels(conditions), ylim([0,2000])
legend(testLabels)

figure,
for i = 1:3
    subplot(1,3,i)
    imagesc(confusionMatrix{i})
end