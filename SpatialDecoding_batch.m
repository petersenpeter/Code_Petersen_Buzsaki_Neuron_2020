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

speed_threshold = 20;
smoothingRange = [40];
training_ratio = [0.8];
bin_size = [10]; % in ms 
sessionNames = {
    'Peter_MS12_170714_122034_concat','Peter_MS12_170715_111545_concat','Peter_MS12_170716_172307_concat','Peter_MS12_170717_111614_concat','Peter_MS12_170719_095305_concat'...
    'Peter_MS12_170717_111614_concat','Peter_MS13_171129_105507_concat','Peter_MS13_171130_121758_concat','Peter_MS13_171128_113924_concat','Peter_MS13_171201_130527_concat',...
    'Peter_MS21_180629_110332_concat','Peter_MS21_180627_143449_concat','Peter_MS21_180625_153927_concat','Peter_MS21_180712_103200_concat','Peter_MS21_180628_155921_concat',...
    'Peter_MS22_180628_120341_concat','Peter_MS22_180629_110319_concat','Peter_MS22_180720_110055_concat','Peter_MS22_180711_112912_concat'};
sessionNames = {'Peter_MS22_180719_122813_concat','Peter_MS21_180807_122213_concat','Peter_MS21_180719_122733_concat'}
testLabels = {'Pre','Cooling','Post','Own'};
conditions = {'Pre','Cooling','Post'};
% 2,10
for iii = 3%1:length(sessionNames)%[1,3:9,11:length(sessionNames)]
    disp(['*** Processing sessions: ', num2str(iii),'/', num2str(length(sessionNames)),' sessions: ' sessionNames{iii}])
    [session, basename, basepath, clusteringpath] = db_set_session('sessionName',sessionNames{iii});
    
    % Loading preprocessed data
    spikes = loadSpikes('session',session);
    for i = 1:size(spikes.times,2)
        idx = find(spikes.speed{i} > speed_threshold);
        spikes.times{i} = spikes.times{i}(idx);
        spikes.pos_linearized{i} = spikes.pos_linearized{i}(idx);
        spikes.theta_phase{i} = spikes.theta_phase{i}(idx);
        spikes.trials{i} = spikes.trials{i}(idx);
    end
    
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
    % Optize interation count
    
    positionDecodingMaxCorr1 = {};
    confusionMatrix = {};
    trials.intervals = animal.time([trials.start;trials.end]');
    animal.pos_linearized_norm = ceil(100*animal.pos_linearized/(animal.pos_linearized_limits(2)));
    [positionDecodingMaxCorr1{4},confusionMatrix{4}] = positionDecodingMaxCorr(spikes,animal,trials,lfp5,smoothingRange,'plotting',0,'saveMat',1,'training_ratio',training_ratio,'bin_size',bin_size);
    
    for j = 1:3
        j
        trials.intervals = animal.time([trials.start;trials.end]');
        animal.pos_linearized_norm = ceil(100*animal.pos_linearized/(animal.pos_linearized_limits(2)));
        [positionDecodingMaxCorr1{j},confusionMatrix{j}] = positionDecodingMaxCorr(spikes,animal,trials,lfp5,smoothingRange,'plotting',0,'saveMat',1,'testCondition',j,'training_ratio',training_ratio,'bin_size',bin_size);
    end
    save('positionDecodingMaxCorr1.mat','positionDecodingMaxCorr1','confusionMatrix')
    
    % PLOTS
    figure('name',[basename ', smoothingRange=' num2str(smoothingRange),', training_ratio=',num2str(training_ratio),', bin_size=',num2str(bin_size)],'position',[50,50,1000,800]),
    for j = 1:4
        subplot(3,3,7)
        plot(positionDecodingMaxCorr1{j}.results.condition+j/5-0.5,positionDecodingMaxCorr1{j}.results.mse_rate,'o'), ylim([0,2500])
        title('mse rate'), xticks([1 2 3]), xticklabels(conditions), xlim([0.6,3.4]), ylabel('mse'), hold on
        subplot(3,3,8)
        plot(positionDecodingMaxCorr1{j}.results.condition+j/5-0.5,positionDecodingMaxCorr1{j}.results.mse_phase_all,'o'), ylim([0,2500])
        title('mse phase all'), xticks([1 2 3]), xticklabels(conditions), xlim([0.6,3.4]), hold on
        subplot(3,3,9)
        plot(positionDecodingMaxCorr1{j}.results.condition+j/5-0.5,positionDecodingMaxCorr1{j}.results.mse_chance,'o'), ylim([0,2500])
        title('mse chance'), xticks([1 2 3]), xticklabels(conditions), xlim([0.6,3.4]), hold on
    end
    
    % Decoding matrix
    % figure,
    colorrange = [400,2000];
    colorrange2 = [0,500];
    % mse_rate
    decodingPerformaceMatrix_mse_rate = [];
    for j = 1:4
        for i = 1:3
            decodingPerformaceMatrix_mse_rate(j,i) = mean(positionDecodingMaxCorr1{j}.results.mse_rate(find(positionDecodingMaxCorr1{j}.results.condition==i)));
        end
    end
    subplot(3,3,1), imagesc(decodingPerformaceMatrix_mse_rate)
    colorbar
    xticks([1 2 3]), xticklabels(conditions), yticks([1 2 3 4]), yticklabels(testLabels), clim(colorrange2), title('mse rate')
    % mse_phase_all
    decodingPerformaceMatrix_mse_phase_all = [];
    for j = 1:4
        for i = 1:3
            decodingPerformaceMatrix_mse_phase_all(j,i) = mean(positionDecodingMaxCorr1{j}.results.mse_phase_all(find(positionDecodingMaxCorr1{j}.results.condition==i)));
        end
    end
    subplot(3,3,2), imagesc(decodingPerformaceMatrix_mse_phase_all)
    colorbar
    xticks([1 2 3]), xticklabels(conditions), yticks([]), clim(colorrange2), title('mse phase all')
    
    % mse_chance
    decodingPerformaceMatrix_mse_chance = [];
    for j = 1:4
        for i = 1:3
            decodingPerformaceMatrix_mse_chance(j,i) = mean(positionDecodingMaxCorr1{j}.results.mse_chance(find(positionDecodingMaxCorr1{j}.results.condition==i)));
        end
    end
    subplot(3,3,3), imagesc(decodingPerformaceMatrix_mse_chance)
    xticks([1 2 3]), xticklabels(conditions), yticks([]), clim(colorrange), title('mse chance')
    colorbar
    subplot(3,3,4), plot(decodingPerformaceMatrix_mse_rate'), xticks([1 2 3]), xticklabels(conditions), ylim([0,2000])
    subplot(3,3,5), plot(decodingPerformaceMatrix_mse_phase_all'), xticks([1 2 3]), xticklabels(conditions), ylim([0,2000])
    subplot(3,3,6), plot(decodingPerformaceMatrix_mse_chance'), xticks([1 2 3]), xticklabels(conditions), ylim([0,2000])
    legend(testLabels)
    
    figure('name',[basename ', smoothingRange=' num2str(smoothingRange),', training_ratio=',num2str(training_ratio),', bin_size=',num2str(bin_size)],'position',[50,50,1000,800]),
    for j = 1:4
        for i = 1:3
            subplot(4,3,i+(j-1)*3)
            imagesc(confusionMatrix{j}(:,:,i)), title(['Con: ' conditions{i},'. test: ' testLabels{j}])
        end
    end
    drawnow
end

%% Plots
decodingPerformaceMatrix_mse_rate_summary = [];
colors = {'b','r','g','m'};
sessions = {};
confusionMatrix1 = {zeros(100,100,3),zeros(100,100,3),zeros(100,100,3),zeros(100,100,3)};
for iii = [1,3:9,11:length(sessionNames)]
    disp(['*** Processing sessions: ', num2str(iii),'/', num2str(length(sessionNames)),' sessions: ' sessionNames{iii}])
    [session, basename, basepath, clusteringpath] = db_set_session('sessionName',sessionNames{iii});
    sessions{i} = session;
    switch session.animal.name
        case 'MS12'
            clr = colors{1};
        case 'MS13'
            clr = colors{2};
        case 'MS21'
            clr = colors{3};
        case 'MS22'
            clr = colors{4};
    end

    load('positionDecodingMaxCorr1.mat')
    figure('name',[basename ', smoothingRange=' num2str(smoothingRange),', training_ratio=',num2str(training_ratio),', bin_size=',num2str(bin_size)],'position',[50,50,1000,800]),
    for j = 1:4
        subplot(3,3,7)
        plot(positionDecodingMaxCorr1{j}.results.condition+j/5-0.5,positionDecodingMaxCorr1{j}.results.mse_rate,'o'), ylim([0,2500])
        title('mse rate'), xticks([1 2 3]), xticklabels(conditions), xlim([0.6,3.4]), ylabel('mse'), hold on
        subplot(3,3,8)
        plot(positionDecodingMaxCorr1{j}.results.condition+j/5-0.5,positionDecodingMaxCorr1{j}.results.mse_phase_all,'o'), ylim([0,2500])
        title('mse phase all'), xticks([1 2 3]), xticklabels(conditions), xlim([0.6,3.4]), hold on
        subplot(3,3,9)
        plot(positionDecodingMaxCorr1{j}.results.condition+j/5-0.5,positionDecodingMaxCorr1{j}.results.mse_chance,'o'), ylim([0,2500])
        title('mse chance'), xticks([1 2 3]), xticklabels(conditions), xlim([0.6,3.4]), hold on
    end
    
    % Decoding matrix
    % figure,
    colorrange = [400,2000];
    colorrange2 = [0,500];
    
    % mse_rate
    decodingPerformaceMatrix_mse_rate = [];
    for j = 1:4
        for i = 1:3
            decodingPerformaceMatrix_mse_rate(j,i) = median(positionDecodingMaxCorr1{j}.results.mse_rate(find(positionDecodingMaxCorr1{j}.results.condition==i)));
        end
    end
    subplot(3,3,1), imagesc(decodingPerformaceMatrix_mse_rate)
    colorbar
    xticks([1 2 3]), xticklabels(conditions), yticks([1 2 3 4]), yticklabels(testLabels), clim(colorrange2), title('mse rate')
    % mse_phase_all
    decodingPerformaceMatrix_mse_phase_all = [];
    for j = 1:4
        for i = 1:3
            decodingPerformaceMatrix_mse_phase_all(j,i) = median(positionDecodingMaxCorr1{j}.results.mse_phase_all(find(positionDecodingMaxCorr1{j}.results.condition==i)));
        end
    end
    subplot(3,3,2), imagesc(decodingPerformaceMatrix_mse_phase_all)
    colorbar
    xticks([1 2 3]), xticklabels(conditions), yticks([]), clim(colorrange2), title('mse phase all')
    
    % mse_chance
    decodingPerformaceMatrix_mse_chance = [];
    for j = 1:4
        for i = 1:3
            decodingPerformaceMatrix_mse_chance(j,i) = median(positionDecodingMaxCorr1{j}.results.mse_chance(find(positionDecodingMaxCorr1{j}.results.condition==i)));
        end
    end
    subplot(3,3,3), imagesc(decodingPerformaceMatrix_mse_chance)
    xticks([1 2 3]), xticklabels(conditions), yticks([]), clim(colorrange), title('mse chance')
    colorbar
    subplot(3,3,4), plot(decodingPerformaceMatrix_mse_rate'), xticks([1 2 3]), xticklabels(conditions), ylim([0,2000])
    subplot(3,3,5), plot(decodingPerformaceMatrix_mse_phase_all'), xticks([1 2 3]), xticklabels(conditions), ylim([0,2000])
    subplot(3,3,6), plot(decodingPerformaceMatrix_mse_chance'), xticks([1 2 3]), xticklabels(conditions), ylim([0,2000])
    legend(testLabels)
    
    figure(100)
    for i = 1:4
        subplot(2,2,i)
        plot(positionDecodingMaxCorr1{i}.results.condition+iii/40-0.25,positionDecodingMaxCorr1{i}.results.mse_rate,['.',clr]), hold on
        ylim([0,1500])
        title(['Testing on ', testLabels{i}]), xticks([1 2 3]), xticklabels(conditions), ylabel('mse'), xlim([0.45,3.55])
        plot([1,2,3]+iii/40-0.25,decodingPerformaceMatrix_mse_rate(i,:)',['o-',clr]), xticks([1 2 3]), xticklabels(conditions)
        text(2+iii/40-0.25,decodingPerformaceMatrix_mse_rate(i,2),num2str(session.spikeSorting{1}.cellCount))
    end
    drawnow
    decodingPerformaceMatrix_mse_rate_summary(iii,:) = decodingPerformaceMatrix_mse_rate(4,:);
    
    figure(101)
    subplot(1,2,1)
    plot(decodingPerformaceMatrix_mse_rate_summary(iii,:)'/decodingPerformaceMatrix_mse_rate_summary(iii,1),['o-',clr]), xticks([1 2 3]), xticklabels(conditions), xlim([0.6,3.4]), hold on
    subplot(1,2,2)
    plot(decodingPerformaceMatrix_mse_rate_summary(iii,1)/decodingPerformaceMatrix_mse_rate_summary(iii,2),decodingPerformaceMatrix_mse_rate_summary(iii,3)/decodingPerformaceMatrix_mse_rate_summary(iii,2),['o',clr]), hold on
    figure('name',[basename ', smoothingRange=' num2str(smoothingRange),', training_ratio=',num2str(training_ratio),', bin_size=',num2str(bin_size)],'position',[50,50,1000,800]),
    for j = 1:4
        for i = 1:3
            subplot(4,3,i+(j-1)*3)
            imagesc(confusionMatrix{j}(:,:,i)), title(['Con: ' conditions{i},'. test: ' testLabels{j}])
        end
    end
    drawnow
    for j = 1:4
        confusionMatrix1{j} = confusionMatrix1{j} + confusionMatrix{j};
    end
end
figure(100)
plot([0.5,2.5],[0.5,2.5],'--k')

figure('name',['CONFUSION MATRIX smoothingRange=' num2str(smoothingRange),', training_ratio=',num2str(training_ratio),', bin_size=',num2str(bin_size)],'position',[50,50,1000,800]),
for j = 1:4
    for i = 1:3
        subplot(4,3,i+(j-1)*3)
        imagesc(zscore(confusionMatrix1{j}(:,:,i))), title(['Con: ' conditions{i},'. test: ' testLabels{j}]), 
        temp2 = clim;
        clim(temp2/2)
    end
end
