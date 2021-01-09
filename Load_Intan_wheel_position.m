 function wheeldata = Load_Intan_wheel_position(Intan_rec_info,ch_wheel_pos)
% Saving analog wheel data (temperature and wheel data
num_channels = length(Intan_rec_info.board_adc_channels); % ADC input info from header file
fileinfo = dir('analogin.dat');
num_samples = fileinfo.bytes/(num_channels * 2); % uint16 = 2 bytes
fid = fopen('analogin.dat', 'r');
v = fread(fid, [num_channels, num_samples], 'uint16');
fclose(fid);
% v = v * 0.000050354; % convert to volts
downsample_n2 = 200;
wheel_raw = medfilt1(v(ch_wheel_pos,:),200) * 0.000050354;
wheel_rad = 14.86; % Radius of the wheel in cm

% Wheel position
wheel_pos = downsample(wheel_raw,downsample_n2);
clear wheel_raw
sr_wheel_pos = Intan_rec_info.frequency_parameters.amplifier_sample_rate/downsample_n2;
while sum((wheel_pos < 0.1))
    wheel_pos(find(wheel_pos < 0.1)) = wheel_pos(find(wheel_pos < 0.1)-1);
end
wheel_pos_polar = 2*pi*(wheel_pos-min(wheel_pos))/(max(wheel_pos)-min(wheel_pos));
clear wheel_raw wheel_pos
% Calibration for the case where the V/wheel pos is not linear
calibration = 0;
if calibration == 1
    figure(2)
    ax1(1) = subplot(2,1,1); plot([1:length(wheel_pos_polar)]/sr_wheel_pos,wheel_pos_polar,'.'), hold on
    wheel_reset_left = find(diff(wheel_pos_polar)>2*pi*0.9)+1;
    wheel_reset_right = find(diff(wheel_pos_polar)<-2*pi*0.9);
    plot(wheel_reset_left/sr_wheel_pos,2*pi*ones(length(wheel_reset_left),1),'*r'),
    plot((wheel_reset_left-1)/sr_wheel_pos,zeros(length(wheel_reset_left),1),'*r')
    plot(wheel_reset_right/sr_wheel_pos,2*pi*ones(length(wheel_reset_right),1),'*b'),
    plot((wheel_reset_right-1)/sr_wheel_pos,zeros(length(wheel_reset_right),1),'*b')

    indices_chosen  = indices_right;
    [B,I] = sort([wheel_reset_left,wheel_reset_right]);
    indices_left = find(diff(I) == 1 & I(1:end-1) < length(wheel_reset_left) & diff(B) < 150);
    indices_right = (find(diff(I) == 1 & I(1:end-1) > length(wheel_reset_left)))-length(wheel_reset_left);
    plot(B(indices_left)/sr_wheel_pos,2*pi*ones(length(indices_left),1),'*y'),
    plot(B(indices_left+1)/sr_wheel_pos,zeros(length(indices_left),1),'*k')
    % plot(wheel_reset_right,2*pi*ones(length(wheel_reset_right),1),'*b'),
    % plot(wheel_reset_right-1,zeros(length(wheel_reset_right),1),'*b')
    p = [];
    figure(3),
    for i = 1:length(indices_left)
        x_span = length(B(indices_left(i)):B(indices_left(i)+1)-1);
        subplot(3,1,1)
        plot(wheel_pos_polar(B(indices_left(i)):B(indices_left(i)+1)-1)), hold on
        subplot(3,1,2)
        plot(wheel_pos_polar(B(indices_left(i)):B(indices_left(i)+1)-1),2*pi*[x_span:-1:1]/x_span), hold on
        p(i,:) = polyfit(wheel_pos_polar(B(indices_left(i)):B(indices_left(i)+1)-1),2*pi*[x_span:-1:1]/x_span,8);
    end
    plot(2*pi*[0,1],2*pi*[0,1],'k','linewidth',2)
    p2 = mean(p);
    x1 = wheel_pos_polar(B(indices_left(i)):B(indices_left(i)+1)-1);
    y1 = polyval(p2,x1);
    plot(x1,y1,'k','linewidth',2)
    subplot(3,1,3)
    for i = 1:length(indices_left)
        x_span = length(B(indices_left(i)):B(indices_left(i)+1)-1);
        plot(2*pi*[x_span:-1:1]/x_span,polyval(p2,wheel_pos_polar(B(indices_left(i)):B(indices_left(i)+1)-1))), hold on
    end
    plot(2*pi*[0,1],2*pi*[0,1],'k','linewidth',2)
    wheel_pos_polar_call = polyval(p2,wheel_pos_polar);
else
    wheel_pos_polar_call = wheel_pos_polar;
end

wheel_velocity = diff(unwrap(wheel_pos_polar_call));

wheeldata.wheel_position = wheel_pos_polar_call;
wheeldata.time = [1:length(wheel_pos_polar_call)]/sr_wheel_pos;
wheeldata.sr = sr_wheel_pos;
wheeldata.wheel_velocity = [sr_wheel_pos*wheel_rad*nanconv(wheel_velocity,gausswin(250)'/sum(gausswin(250)),'edge'),0];

save('wheeldata.mat','wheeldata')
disp('wheel data saved')

% 
% while sum(abs(wheel_velocity) > 3/2*pi)
%     wheel_velocity(find(wheel_velocity > pi)) = 2*pi-wheel_velocity(wheel_velocity > pi);
%     wheel_velocity(find(wheel_velocity < -pi)) = 2*pi+wheel_velocity(wheel_velocity < -pi);
% end
% while sum(wheel_velocity > 0.2)
%     wheel_velocity(find((wheel_velocity) > 0.2)) = wheel_velocity(find((wheel_velocity) > 0.2)-1);
% end
% while sum(wheel_velocity < -0.2)
%     wheel_velocity(find((wheel_velocity) < -0.2)) = wheel_velocity(find((wheel_velocity) < -0.2)-1);
% end

% Plots
figure,
ax1(1) = subplot(2,1,1)
plot(wheeldata.time,wheeldata.wheel_position,'.k')
ylabel('Position (rad)'), title('Analog channels')
ax1(2) = subplot(2,1,2); plot(wheeldata.time,abs(wheeldata.wheel_velocity))
ylabel('Velocity (cm/s)')
linkaxes(ax1,'x'), axis tight
