function monosynOut = monosynapticCoolingEffect(recording)
% By Peter Petersen
% petersen.peter@gmail.com
% 26-05-2019

[cell_metrics_idxs,cell_metrics] = get_CellMetrics('session',recording.name);

prebehaviortime = 0;
if recording.concat_behavior_nb > 0
    prebehaviortime = 0;
    if all(recording.concat_behavior_nb > 1)
        for i = 1:recording.concat_behavior_nb-1
            fullpath = fullfile([recording.datapath,recording.animal_id], recording.concat_recordings{i}, [recording.concat_recordings{i}, '.dat']);
            temp2_ = dir(fullpath);
            prebehaviortime = prebehaviortime + temp2_.bytes/recording.nChannels/2/recording.sr;
        end
    end
    behaviortime = 0;
    for i = 1:length(recording.concat_behavior_nb)
        i1 = recording.concat_behavior_nb(i);
        fullpath = fullfile([recording.datapath, recording.animal_id], recording.concat_recordings{i1}, [recording.concat_recordings{i1}, '.dat']);
        temp2_ = dir(fullpath);
        behaviortime = behaviortime+temp2_.bytes/recording.nChannels/2/recording.sr;
    end
    
else
    temp_ = dir(fname);
    behaviortime = temp_.bytes/recording.nChannels/2/recording.sr;
end

% Cooling
cooling = recording.cooling;
if recording.cooling_session == 0
    error('This is a control session')
    cooling.onsets = animal.time(round(length(animal.time)./3));
    cooling.offsets = animal.time(round(2*length(animal.time)./3));
    cooling.cooling = [cooling.onsets(1)+10;cooling.offsets(1)];
    cooling.nocooling = [[1,cooling.onsets(1)];[cooling.offsets(1)+120,behaviortime+prebehaviortime]]';
else
    if recording.ch_temp ~= 0
        load('temperature.mat')
        temp_range = [32,34];% temp_1 defines the upper limit on cooling, temp_2 the lower limit on no cooling
        t_start = find(temperature.time>prebehaviortime,1);
        t_end = find(temperature.time>prebehaviortime+behaviortime,1);
        if isempty(t_end)
            t_end = length(temperature.time);
        end
        test = find(diff(temperature.temp(t_start:t_end) < temp_range(1),2)== 1);
        test = test+t_start;
        test(diff(test)<10*temperature.sr)=[];
        cooling.onsets = temperature.time(test(temperature.temp(test+1)-temperature.temp(test)<0));
        cooling.offsets = temperature.time(test(temperature.temp(test+1)-temperature.temp(test)>0));
        if length(cooling.offsets)<length(cooling.onsets)
            cooling.offsets = [cooling.offsets,temperature.time(end)];
        end
        cooling.cooling = [cooling.onsets;cooling.offsets];
        cooling.cooling2 = [cooling.onsets-20;cooling.offsets+180];
        cooling.nocooling = reshape([prebehaviortime;cooling.cooling2(:);prebehaviortime+behaviortime],[2,size(cooling.cooling2,2)+1]);
    elseif recording.ch_CoolingPulses ~= 0
        load('digitalchannels.mat')
        cooling.onsets = digital_on{recording.ch_CoolingPulses}/recording.sr;
        cooling.offsets = cooling.onsets + 12*60;
        cooling.cooling = [cooling.onsets(1)+10;cooling.offsets(1)];
        cooling.nocooling = [[1,cooling.onsets'];[cooling.offsets'+120,behaviortime]]';
    else
        disp('Getting cooling intervals from recording metadata')
        cooling.onsets = recording.cooling_onsets;
        cooling.offsets = recording.cooling_offsets;
        cooling.cooling = [];
        cooling.nocooling = [];
        for i = 1:size(cooling.onsets,2)
            cooling.cooling = [cooling.cooling;[cooling.onsets(i)+10,cooling.offsets(i)]+prebehaviortime];
            if i == 1
                cooling.nocooling = [cooling.nocooling;[1,cooling.onsets(1)]+prebehaviortime];
            else
                cooling.nocooling = [cooling.nocooling;[cooling.offsets(i-1),cooling.onsets(i)]+prebehaviortime];
            end
        end
        cooling.nocooling = [cooling.nocooling;[cooling.offsets(end),behaviortime]+prebehaviortime]';
        cooling.cooling = cooling.cooling';
    end
end

spikes = loadSpikes('clusteringpath',recording.SpikeSorting.path,'clusteringformat',recording.SpikeSorting.method,'basename',recording.name);
timeRestriction_nocooling = cooling.nocooling;
timeRestriction_pre = cooling.nocooling(:,1);
timeRestriction_cooling = cooling.cooling;
timeRestriction_post = cooling.nocooling(:,2);

timeRestriction_all = {timeRestriction_pre,timeRestriction_cooling};

ccg = {};
indeces2keep = {};
colors = {'r','b','g'};
binSize = 0.00005;
conv_win = 0.015;
nPreUnitSpikes = {};
postfiringRate = {};
sessions = db_load_sessions('session',recording.name);
session = sessions{1};
waveforms = {};
nSpikes = {};
figure
for j_state = 1:2
    timeRestriction = timeRestriction_all{j_state}';
    spikes2 = spikes;
    for j = 1:size(spikes2.times,2)
        indeces2keep = [];
        indeces2keep = find(any(spikes2.times{j} >= timeRestriction(:,1)' & spikes2.times{j} <= timeRestriction(:,2)', 2));
%         indeces2keep2 = find(spikes2.speed{j}>5);
%         indeces2keep = intersect(indeces2keep,indeces2keep2);
        spikes2.ts{j} =  spikes2.ts{j}(indeces2keep);
        spikes2.times{j} =  spikes2.times{j}(indeces2keep);
        spikes2.ids{j} =  spikes2.ids{j}(indeces2keep);
        spikes2.amplitudes{j} =  spikes2.amplitudes{j}(indeces2keep);
        spikes2.total(j) =  length(indeces2keep);
        nPreUnitSpikes{j_state}(j) = spikes2.total(j);
        postfiringRate{j_state}(j) = 1/mean(diff(spikes2.times{j}));
    end
    keptUnits{j_state} = find(spikes2.total>0);
    indeces2keep = any(spikes2.spindices(:,1) >= timeRestriction(:,1)' & spikes2.spindices(:,1) <= timeRestriction(:,2)', 2);
    spikes2.spindices = spikes2.spindices(indeces2keep,:);
    %     spikes2 = generateSpinDices(spikes2);
    spike_times = spikes2.spindices(:,1);
    spike_cluster_index = spikes2.spindices(:,2);
    [~, ~, spike_cluster_index] = unique(spike_cluster_index);
    [ccg{j_state},time2] = CCG(spike_times,spike_cluster_index,'binSize',binSize,'duration',0.100);
    plot(spikes2.spindices(:,1),spikes2.spindices(:,2),['.',colors{j_state}]), hold on
    xml = LoadXml(fullfile([recording.name, '.xml']));
    spikes2 = GetWaveformsFromDat(spikes2,pwd,recording.name,'session',session);
    waveforms{j_state}.filtWaveform = spikes2.filtWaveform;
    waveforms{j_state}.rawWaveform = spikes2.rawWaveform;
    nSpikes{j_state} = spikes2.total;
end

ccg_mean = [];
ccg_std  = [];
ccg_peak  = [];
ccg_peak_time  = [];
trans = [];
postRate = [];
ccg_single_out = [];
t0 = find(time2>=0,1);
figure
if ~isempty(cell_metrics.putativeConnections.excitatory)
    for i = 1:size(cell_metrics.putativeConnections.excitatory,1)
        if rem(i,12)==0
            figure
        end
        subplot(4,3,rem(i,12)+1), hold on
        for j = 1:length(ccg)
            pair_id = [find(keptUnits{j} == cell_metrics.putativeConnections.excitatory(i,1)),find(keptUnits{j} == cell_metrics.putativeConnections.excitatory(i,2))];
            if length(pair_id)==2
                ccg_single = nanconv(ccg{j}(:,pair_id(1),pair_id(2)),gausswin(19,1)/sum(gausswin(19,1)),'edge');
                plot(time2,ccg_single,colors{j})
                temp11 = nanconv(ccg_single,gausswin(400,1)/sum(gausswin(400,1)),'edge');
                ccg_mean(i,j) = mean(ccg_single-temp11);
                ccg_std(i,j) = std(ccg_single-temp11);
                [ccg_peak(i,j),ccg_peak_time(i,j)] = max(ccg_single(t0:t0+100));
                ccg_single_out(:,i,j) = ccg_single(t0:t0+100)-temp11(t0:t0+100);
                [ccg_peak23,~] = max(ccg_single(t0:t0+100)-temp11(t0:t0+100));
                ccg_peak_strenth(i,j) = (ccg_peak23-ccg_mean(i,j))/ccg_std(i,j);
                ccg_peak_time(i,j) = 1000*time2(ccg_peak_time(i,j)+t0-1);
                
                [trans2,prob,prob_uncor,pred] = GetTransProb(ccg_single',nPreUnitSpikes{j}(cell_metrics.putativeConnections.excitatory(i,1)),binSize,conv_win);
                trans(i,j) = trans2;
                postRate(i,j) = postfiringRate{j}(cell_metrics.putativeConnections.excitatory(i,2));
                plot(ccg_peak_time(i,j)/1000,ccg_peak(i,j),'ok')
            end
        end
        if length(pair_id)==2
            ylabel(['Transprob ', num2str(trans(i,:))]), xlabel(['peaktime ' num2str(ccg_peak_time(i,:)),', strenth ' num2str(ccg_peak_strenth(i,:))])
        end
        grid on, title(['Unit pairs: ' num2str(cell_metrics.putativeConnections.excitatory(i,:))])
        
    end
    
    monosynOut.ccg_mean = ccg_mean;
    monosynOut.ccg_std = ccg_std;
    monosynOut.ccg_peak = ccg_peak;
    monosynOut.ccg_peak_time = ccg_peak_time;
    monosynOut.ccg_peak_strenth = ccg_peak_strenth;
    monosynOut.trans = trans;
    monosynOut.postRate = postRate;
    monosynOut.ccg_single_out1 = ccg_single_out(:,:,1);
    monosynOut.ccg_single_out2 = ccg_single_out(:,:,2);
    monosynOut.waveforms1 = waveforms{1};
    monosynOut.waveforms2 = waveforms{2};
    monosynOut.nSpikes1 = nSpikes{1};
    monosynOut.nSpikes2 = nSpikes{2};
else
    disp('No monosynaptic connections detected')
    monosynOut.ccg_mean = [];
    monosynOut.ccg_std = [];
    monosynOut.ccg_peak = [];
    monosynOut.ccg_peak_time = [];
    monosynOut.ccg_peak_strenth = [];
    monosynOut.trans = [];
    monosynOut.postRate = [];
    monosynOut.ccg_single_out1 = [];
    monosynOut.ccg_single_out2 = [];
    monosynOut.waveforms1 = [];
    monosynOut.waveforms2 = [];
    monosynOut.nSpikes1 = [];
    monosynOut.nSpikes2 = [];
end
