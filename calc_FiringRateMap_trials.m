function RateMapTrials = calc_FiringRateMap_trials(animal,spikes,trials)
step_size = 1;
steps_to_compare = 5;
FiringRateMap_trials = [];
x_hist2 = [animal.pos_linearized_limits(1):1:animal.pos_linearized_limits(2)];
RateMapTrials.x_bins = x_hist2;
RateMapTrials.boundaries = [x_hist2(1),x_hist2(2)];
RateMapTrials.x_bins_smoothing_width = 30;
RateMapTrials.x_bins_smoothing_type = 'gausswin';
FiringRateMap_trials = zeros(length(x_hist2),ceil(trials.total/step_size),size(spikes.ts,2));
FiringRateMap_trials_normalized = zeros(length(x_hist2),ceil(trials.total/step_size),size(spikes.ts,2));
position_counts = zeros(length(x_hist2),ceil(trials.total/step_size));

for i = 1:trials.total/step_size
    ii = i*2-1;
    indexes2 = find(trials.trials2 ==ii |trials.trials2 ==ii+1);
    position_counts(:,i) = nanconv(histc(animal.pos_linearized(indexes2),x_hist2),gausswin(30)./sum(gausswin(30)),'edge');
    
    for j = 1:size(spikes.ts,2)
        indexes = find(spikes.trials{j} == ii | spikes.trials{j} == ii+1);
        FiringRateMap_trials(:,i,j) = nanconv(histc(spikes.pos_linearized{j}(indexes),x_hist2),gausswin(30)./sum(gausswin(30)),'edge');
        FiringRateMap_trials_normalized(:,i,j) = animal.sr*FiringRateMap_trials(:,i,j)./(position_counts(:,i));
        RateMapTrials.map{j}(:,i) = FiringRateMap_trials(:,i,j);
    end
end
FiringRateMap_trials_normalized(isnan(FiringRateMap_trials_normalized)) = 0;
FiringRateMap_trials_normalized(isinf(FiringRateMap_trials_normalized)) = 0;

units_to_plot = 1:size(spikes.ts,2);

for ww = 1:length(units_to_plot)
    unit_to_plot=units_to_plot(ww);
    
    figure, 
    subplot(2,3,1)
    imagesc(x_hist2,1:step_size:trials.total,FiringRateMap_trials_normalized(:,:,unit_to_plot)')
    
    subplot(2,3,2)
    plot(x_hist2,animal.sr*sum(FiringRateMap_trials(:,:,unit_to_plot)')./(sum(position_counts,2)')),axis tight, hold on
    plot(x_hist2,animal.sr*sum(FiringRateMap_trials(:,1:steps_to_compare,unit_to_plot)')./(sum(position_counts(:,1:steps_to_compare),2)'))
    plot(x_hist2,animal.sr*sum(FiringRateMap_trials(:,end-steps_to_compare:end,unit_to_plot)')./(sum(position_counts(:,end-steps_to_compare:end),2)')), legend({'All trials','First 20','Last 20'})
    title(['Unit ' num2str(unit_to_plot)])
    
    subplot(2,3,3)
    imagesc(x_hist2,1:step_size:trials.total,corr(FiringRateMap_trials_normalized(:,:,unit_to_plot),'Type','Spearman')), hold on
    
    subplot(2,3,4)
    test=[animal.sr.*sum(FiringRateMap_trials(:,1:steps_to_compare,unit_to_plot)')./(sum(position_counts(:,1:steps_to_compare),2)');FiringRateMap_trials_normalized(:,:,unit_to_plot)';animal.sr.*sum(FiringRateMap_trials(:,end-steps_to_compare:end,unit_to_plot)')./(sum(position_counts(:,end-steps_to_compare:end),2)')];
    test(isnan(test)) = 0;
    test(isinf(test)) = 0;
    test2 = corr(test','Type','Spearman');
    plot(test2(1,:),'b'),hold on, plot(test2(end,:),'r'), grid on
    
    subplot(2,3,5)
    plot(test2(end,:)-test2(1,:),'k'), grid on
    test_difference(ww,:) = test2(end,:)-test2(1,:);
    if isfield(trials,'optogenetics')
        gridxy(find(diff(trials.optogenetics))/step_size)
    end
    
    subplot(2,3,6)
    temp = corr(FiringRateMap_trials_normalized(:,:,unit_to_plot),'Type','Spearman');
    corr_curve=[];
    for i = 2:size(temp,1)-1
        corr_curve(i) = nansum([-nanmean(temp(1:(i-1),1:(i-1))),nanmean(temp(i:end,i:end))]);
    end
    corr_curve(1)=0;
    corr_curve2(ww,:) = corr_curve;
    plot(corr_curve), xlabel('Trials'), ylabel('Correlation'), hold on,
   
    offDiag = [];
    for i = 1:size(temp,1)-1
        offDiag(i) = nansum(nansum(temp.*diag(ones(size(temp,1)-i,1),i)))/(size(temp,1)-i);
    end
    plot(offDiag,'r')
    
    figure(1001), hold on
    plot(offDiag), hold on,
    
    figure(1002), hold on
    plot(corr_curve), hold on,
    
    figure(1000), hold on
    plot(test2(end,:)-test2(1,:)),hold on, title('Sam''s measure')

end
figure(1000)
test_difference_mean = nanmean(test_difference);
test_difference_std = nanstd(test_difference);
time_test = 1:length(test_difference_mean);
test_isnan = find(~isnan(test_difference_mean));
patch([time_test(test_isnan),flip(time_test(test_isnan))], [test_difference_mean(test_isnan)+test_difference_std(test_isnan),flip(test_difference_mean(test_isnan)-test_difference_std(test_isnan))],'black','EdgeColor','none','FaceAlpha',.2), hold on
plot(time_test(test_isnan),test_difference_mean(test_isnan),'k','linewidth',2), grid on

title('Post-pre place representation'), xlabel('Trials'), ylabel('Probabilities')
if isfield(trials,'optogenetics')
        gridxy(find(diff(trials.optogenetics))/step_size)
    end
save('SpatialDrift.mat','animal','spikes','trials')

figure(1002)
test_difference_mean = nanmean(corr_curve2);
test_difference_std = nanstd(corr_curve2);
time_test = 1:length(test_difference_mean);
test_isnan = find(~isnan(test_difference_mean));
patch([time_test(test_isnan),flip(time_test(test_isnan))], [test_difference_mean(test_isnan)+test_difference_std(test_isnan),flip(test_difference_mean(test_isnan)-test_difference_std(test_isnan))],'black','EdgeColor','none','FaceAlpha',.2), hold on
plot(time_test(test_isnan),test_difference_mean(test_isnan),'k','linewidth',2), grid on

title('Correlation'), xlabel('Trials'), ylabel('Correlation')
if isfield(trials,'optogenetics')
        gridxy(find(diff(trials.optogenetics))/step_size)
    end

