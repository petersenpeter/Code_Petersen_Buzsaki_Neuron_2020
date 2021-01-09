function [tri,out] = calc_trial_averages(spikes,UID,timestamps_in,phasepresession_sign,phasepresession_data)
trial_step_size = 20;
% Missing fields: % error_rate
for k = 1:max(vertcat(spikes.trials{:}))-trial_step_size
    timestamps2 = find(spikes.trials{UID}(timestamps_in) >= k & spikes.trials{UID}(timestamps_in) < k+trial_step_size);
    timestamps = timestamps_in(timestamps2);
    tri.theta_freq(k) = mean(spikes.theta_freq{UID}(timestamps));
    tri.speed(k) = mean(spikes.speed{UID}(timestamps));
    tri.temperature(k) = mean(spikes.temperature{UID}(timestamps));
    tri.spike_count(k) = length(spikes.times{UID}(timestamps));
    if length(timestamps)>50
        % Number of theta cycles
        kk2 = 1;
        theta_cycles = [];
        if ~isempty(timestamps)
            temp = spikes.theta_phase2{UID}(timestamps);
            theta_cycles = temp(end)-temp(1);
        end
        tri.theta_cycles(k) = mean(theta_cycles(theta_cycles>0))/(2*pi);
        
        % Oscillation frequency
        timestamps2 = spikes.times{UID}(timestamps);
        timestamps2 = timestamps2 - timestamps2(1);
        interval = zeros(1,ceil(timestamps2(end)*1000+50));
        interval(1+round(timestamps2*1000)) = 1;
        interval = nanconv(interval,gausswin(80)','edge');
        xcorr_spikes =xcorr(interval,180);
        [~,locs] = findpeaks(xcorr_spikes(181+50:181+150),'SortStr','descend');
        if length(locs)>0
            tri.oscillation_freq(k) = 1/abs(locs(1)+50)*1000;
        else
            tri.oscillation_freq(k) = nan;
        end
        % Precession slope
        [tri.slope1(k),tri.offset1(k),tri.R1(k)] = CircularLinearRegression(phasepresession_data(1,timestamps),phasepresession_data(2,timestamps), phasepresession_sign);
    end
end
out.trial_step_size = trial_step_size;
out.trials_axis = 1:max(vertcat(spikes.trials{:}))-trial_step_size;