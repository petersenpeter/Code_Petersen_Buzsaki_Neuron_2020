function Optitrack = LoadOptitrack(filename,LengthUnit,arena, apply_head_displacement,apply_pca)
% Loading the position tracking data
switch nargin
    case 4
        disp('')
        apply_pca = 0;
    case 3
        disp('')
        apply_pca = 0;
        apply_head_displacement = 0;
    case 2
        disp('')
        apply_pca = 0;
%         arena = 'LinearTrack';
        apply_head_displacement = 0;
    case 1
       apply_head_displacement = 0;
       apply_pca = 0;
%        arena = 'LinearTrack';
       LengthUnit = 1;
end
% filename = [datapath recording '/' recordings(id).tracking_file];
formatSpec = '%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%[^\n\r]';
header_length = 7;

if iscell(filename)
    fileID = fopen(filename{1},'r');
    dataArray = textscan(fileID, formatSpec, 'Delimiter', ',',  'ReturnOnError', false); 
    fclose(fileID);
    FramesPrFile = size(dataArray{1}(header_length:end),1);
    for i = 2:length(filename)
        fileID = fopen(filename{i},'r');
        dataArray_temp = textscan(fileID, formatSpec, 'Delimiter', ',',  'ReturnOnError', false); 
        fclose(fileID);
        for j = 1:length(dataArray)
            dataArray{j} = [dataArray{j};dataArray_temp{j}(header_length:end)];
        end
        FramesPrFile = [FramesPrFile, size(dataArray_temp{1}(header_length:end),1)];
    end
else
    fileID = fopen(filename,'r');
    dataArray = textscan(fileID, formatSpec, 'Delimiter', ',',  'ReturnOnError', false);
    fclose(fileID);
end

Optitrack = [];
Optitrack.Frame = str2double(dataArray{1}(header_length:end));
Optitrack.Time = str2double(dataArray{2}(header_length:end));
Optitrack.Xr = str2double(dataArray{3}(header_length:end));
Optitrack.Yr = str2double(dataArray{4}(header_length:end));
Optitrack.Zr = str2double(dataArray{5}(header_length:end));
Optitrack.Wr = str2double(dataArray{6}(header_length:end));
Optitrack.X = str2double(dataArray{7}(header_length:end));
Optitrack.Y = str2double(dataArray{8}(header_length:end));
Optitrack.Z = str2double(dataArray{9}(header_length:end));
Optitrack.TotalFrames = str2double(dataArray{12}(1));
Optitrack.TotalExportedFrames = str2double(dataArray{14}(1));
Optitrack.RotationType = dataArray{16}(1);
Optitrack.LenghtUnit = dataArray{18}(1);
Optitrack.CoorinateSpace = dataArray{20}(1);
Optitrack.FrameRate = str2double(dataArray{6}{1});
if exist('FramesPrFile')
    Optitrack.FramesPrFile = FramesPrFile;
end
clear dataArray
clearvars filename formatSpec fileID dataArray header_length;

position = 100*[-Optitrack.X,Optitrack.Z,Optitrack.Y]/LengthUnit; % get position out in cm

if apply_head_displacement == 1
    disp('Applying head displacement')
    angles = SpinCalc('QtoEA321',[Optitrack.Xr,Optitrack.Yr,Optitrack.Zr,Optitrack.Wr],1e-5,1);
    v = [0;0;-8]; % Displacement vector (xyz, z is the vertical direction)
    y = rot_Peter(angles,v);
    position = position+y;
end

% Rotating the dimensions using PCA
if apply_pca == 1
    disp('Applying PCA')
    [coeff,position3D,latent] = pca(position);
    position1 = position3D(:,1);
    position = nanmedian(position1,10);
else
    position3D = position;
end

% Estimating the speed of the rat
% animal_speed = 100*Optitrack.FrameRate*(diff(Optitrack.X).^2+diff(Optitrack.Y).^2+diff(Optitrack.Z).^2).^0.5;
animal_speed3 = [Optitrack.FrameRate*sqrt(sum(diff(position)'.^2)),0];
% animal_speed3(animal_speed3>150) = 0;

animal_speed1 = [];
animal_speed = nanconv(animal_speed3,ones(1,10)/10,'edge');
animal_acceleration = [0,diff(animal_speed)];

animal_speed2 = Optitrack.FrameRate*sqrt(sum((diff(position3D).^2),2))';
animal_speed2(animal_speed2>150) = 0;
animal_speed3D = [];
for i = 1:length(animal_speed2)-10
    animal_speed3D(i) = median(animal_speed2(i:i+10));
end
animal_speed3D =[zeros(1,5),animal_speed3D, zeros(1,6)];

figure
subplot(1,2,1)
plot3(position3D(:,1),position3D(:,2),position3D(:,3)), title('3D position'), xlabel('X'), ylabel('Y'), zlabel('Z'),axis tight,view(2), hold on,
% switch arena{1}
%     case {'CircularTrack', 'Circular track'}
%         position = position + [5,-5,0]; % get position out in cm
%         position3D = position3D + [5,-5,0];
%         maze_dia_out = 116.5;
%         maze_dia_in = 96.5;
%         pos1 = [-maze_dia_out/2, -maze_dia_out/2, maze_dia_out, maze_dia_out];
%         pos2 = [-maze_dia_in/2, -maze_dia_in/2, maze_dia_in, maze_dia_in];
%         rectangle('Position',pos1,'Curvature',[1 1]), hold on
%         rectangle('Position',pos2,'Curvature',[1 1])
%         cross_radii = 47.9;
%         plot([-cross_radii -5 -5],[-5 -5 -cross_radii],'k'), hold on
%         plot([cross_radii 5 5],[5 5 cross_radii],'k')
%         plot([cross_radii 5 5],[-5 -5 -cross_radii],'k')
%         plot([-cross_radii -5 -5],[5 5 cross_radii],'k')
%         axis equal
%         xlim([-65,65]),ylim([-65,65]),
% 	case {'LinearTrack', 'Linear track'}
%         disp('Linear Track detected')
% end

subplot(1,2,2)
plot3(position3D(:,1),position3D(:,2),animal_speed), hold on
xlabel('X'), ylabel('Y'),zlabel('Speed'), axis tight

Optitrack.position1D = position(:,1)';
Optitrack.position3D = position3D';
Optitrack.animal_speed = animal_speed;
Optitrack.animal_acceleration = animal_acceleration;
