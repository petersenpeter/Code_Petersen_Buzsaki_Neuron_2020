function [offDiagSlope,offDiagConstant,corr_curve3] = calc_FiringRateMap_trials2(animal,spikes,trials,input111)
steps_to_compare = 20;
FiringRateMap_trials = [];
offDiagSlope = [];
offDiagConstant = [];

x_hist2 = [animal.pos_linearized_limits(1):1:animal.pos_linearized_limits(2)];
FiringRateMap_trials.xhist = x_hist2;
FiringRateMap_trials = zeros(length(x_hist2),ceil(trials.total)-1,size(spikes.ts,2));
FiringRateMap_trials_normalized = zeros(length(x_hist2),ceil(trials.total)-1,size(spikes.ts,2));
position_counts = zeros(length(x_hist2),ceil(trials.total)-1);

k = 1;
for i = 1:trials.total-2
    ii = i;
    indexes2 = find(trials.trials2 ==ii | trials.trials2 ==ii+1 | trials.trials2 ==ii+2);
    position_counts(:,i) = histc(animal.pos_linearized(indexes2),x_hist2);
    
    for j = 1:size(spikes.ts,2)
        indexes = find(spikes.trials{j} == ii | spikes.trials{j} == ii+1  | spikes.trials{j} == ii+2);
        FiringRateMap_trials(:,i,j) = histc(spikes.pos_linearized{j}(indexes),x_hist2);
        FiringRateMap_trials_normalized(:,i,j) = nanconv(animal.sr*FiringRateMap_trials(:,i,j)./(position_counts(:,i)),gausswin(50)./sum(gausswin(50)),'edge');
    end
end
FiringRateMap_trials_normalized(isnan(FiringRateMap_trials_normalized)) = 0;
FiringRateMap_trials_normalized(isinf(FiringRateMap_trials_normalized)) = 0;

units_to_plot = 1:size(spikes.ts,2);

for ww = 1:length(units_to_plot)
    unit_to_plot=units_to_plot(ww);
    
    figure, 
    subplot(2,3,1)
    imagesc(x_hist2,1:trials.total-1,FiringRateMap_trials_normalized(:,:,unit_to_plot)')
    
    subplot(2,3,2)
    plot(x_hist2,nanmean(FiringRateMap_trials_normalized(:,:,unit_to_plot)')),axis tight, hold on
    plot(x_hist2,nanmean(FiringRateMap_trials_normalized(:,1:steps_to_compare,unit_to_plot)'))
    plot(x_hist2,nanmean(FiringRateMap_trials_normalized(:,end-steps_to_compare:end,unit_to_plot)')), legend({'All trials','First 20','Last 20'})
    title(['Unit ' num2str(unit_to_plot)])
    
    subplot(2,3,3)
    imagesc(x_hist2,1:trials.total-1,corr(FiringRateMap_trials_normalized(:,:,unit_to_plot))), hold on
    
    temp = corr(FiringRateMap_trials_normalized(:,:,unit_to_plot));
    corr_curve=[];
    for i = 5:size(temp,1)-1
        temp11 = temp(1:i,1:i);
        temp22 = temp(i+1:end,i+1:end);
        temp33 = temp(1:i,i+1:end);
        corr_curve(i) = (1+nanmean(temp33(:))) / nanmean([temp11(:);temp22(:)]);
    end
    corr_curve3{ww} = corr_curve;
    corr_curve2(ww,:) = corr_curve;
    
    subplot(2,3,4)
    
    subplot(2,3,5)
    plot(corr_curve), xlabel('Trials'), ylabel('Correlation'), hold on, grid on, title('Corr curve'), axis tight
    gridxy(find(diff(trials.optogenetics)))
    
    subplot(2,3,6)
    offDiag = [];
    for i = 1:size(temp,1)-1
        offDiag(i) = nansum(nansum(temp.*diag(ones(size(temp,1)-i,1),i)))/(size(temp,1)-i);
    end
    plot(offDiag,'r')
    xlabel('Trials'), ylabel('Correlation'), title('diagonal correlation'), axis tight
    gridxy(find(diff(trials.optogenetics)))
    
    figure(1001), hold on
    subplot(4,5,input111), hold on
    plot(offDiag), hold on, axis tight
    elements = 2:ceil(length(offDiag)/2);
    x = 1:length(elements);
    y1 = offDiag(elements);
    P = polyfit(x,y1,1); yfit = P(1)*x+P(2);
    offDiagSlope(k) = P(1);
    offDiagConstant(k) = P(2);
    k = k + 1;
    figure(1002)
    subplot(4,5,input111), hold on
    plot(corr_curve)
end

figure(1002)
subplot(4,5,input111)
test_difference_mean = nanmean(corr_curve2);
test_difference_std = nanstd(corr_curve2);
time_test = 1:length(test_difference_mean);
test_isnan = find(~isnan(test_difference_mean)); axis tight
% patch([time_test(test_isnan),flip(time_test(test_isnan))], [test_difference_mean(test_isnan)+test_difference_std(test_isnan),flip(test_difference_mean(test_isnan)-test_difference_std(test_isnan))],'black','EdgeColor','none','FaceAlpha',.2), hold on
% plot(time_test(test_isnan),test_difference_mean(test_isnan),'k','linewidth',2), 
grid on
% title('Correlation'), xlabel('Trials'), ylabel('Correlation')
