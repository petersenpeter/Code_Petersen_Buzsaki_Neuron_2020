% Impedance Measures
clear all
id = 1; % MS9
impedance(id).rat = 'MS9';
impedance(id).measures = {'MS9_Impedance1207','MS9_Impedance12072' ,'MS9_Impedance12121','MS9_Impedance12122','MS9_Impedance1213','MS9_Impedance12132','MS9_Impedance12133' ,'MS9_Impedance12134','MS9_Impedance1215','MS9_Impedance1216','MS9_Impedance1219','MS9_Impedance1220' ,'MS9_Impedance0105','MS9_Impedance0117','MS9_Impedance0130','MS9_Impedance_0131','MS9_Impedance_0201','MS9_Impedance_0207'};
impedance(id).path = 'D:\IntanData\Peter_MS9\Peter_MS9_ImpedanceMeasures';
impedance(id).probename = '64Ch Cambridge Neurotech 4 shank staggered probe';
impedance(id).probelayout = 1+[22,27,13,14,15,23,24,10,11,12,20,25,7,8,9,21;26,4,5,6,18,29,28,2,3,16,19,30,31,1,62,0;36,58,59,56,44,35,34,60,61,46,45,32,33,63,47,17;40,37,51,48,49,41,38,52,53,50,42,39,57,54,55,43]';
impedance(id).screwmoved = [3,8,9,10,11,12,13,15];

id = 2; % MS10
impedance(id).rat = 'MS10';
impedance(id).measures = {'MS10_Impedance_PreSurgery','MS10_Impedance_PreSurgery0215','MS10_Impedance_0223','MS10_Impedance_0223_3'};
impedance(id).path = 'D:\IntanData\Peter_MS10\Peter_MS10_ImpedanceMeasures';
impedance(id).probename = '64Ch NeuroNexus 5 shank staggered probe';
impedance(id).probelayout = 1+[50,57,51,56,48,59,49,58,53,54,52,55; 62,37,63,38,60,40,61,41,32,36,33,34; 46,16,45,19,44,18,42,20,47,17,43,21; 27,0,24,1,22,2,23,3,26,30,28,31;   7,12,6,13,5,14,4,15,8,11,9,10; 29,35,25,39,0,0,0,0,0,0,0,0]'; % 29,35,25,39
impedance(id).screwmoved = [3];

id = 3; % MS13
impedance(id).rat = 'MS13';
impedance(id).measures = {'Impedance-pre-implant','Impedance-post-implant-01-11-17','Impedance-07-11-17','Impedance-07-11-17 2','Impedance-08-11-17','Impedance-28-11-17','Impedance-01-12-17','Impedance-05-12-17'};
impedance(id).path = 'F:\IntanData\Peter_MS13\ImpedanceTests';
impedance(id).probename = '2x 64Ch Cambridge Neurotech 4 shank poly2 probe';
impedance(id).probelayout = 1+[42, 40, 57, 51, 55, 49, 43, 38, 54, 53, 39, 50, 52, 41, 48, 37; 45, 36, 33, 59, 47, 44, 17, 34, 63, 61, 32, 46, 60, 35, 56, 58 ;19, 26, 31, 5, 62, 18, 0, 28, 1, 3, 30, 16, 2, 29, 6, 4 ;20, 22, 7, 13, 9, 15, 21, 24, 8, 11, 25, 12, 10, 23, 14, 27;106, 104, 121, 115, 119, 113, 107, 102, 118, 117, 103, 114, 116, 105, 112, 101 ; 109, 100, 97, 123, 111, 108, 81, 98, 127, 125, 96, 110, 124, 99, 120, 122 ; 83, 90, 95, 69, 126, 82, 64, 92, 65, 67, 94, 80, 66, 93, 70, 68 ; 84, 86, 71, 77, 73, 79, 85, 88, 72, 75, 89, 76, 74, 87, 78, 91]';
impedance(id).screwmoved = [1];

id = 4; % YutaMouse
impedance(id).rat = 'YutaMouse #4,#5,#11';
impedance(id).measures = {'YMV04-day1-beforelesion','YMV04-day1-aferlesion','YMV05-day1','YMV05-day1-afterlesion','YMV11'};
impedance(id).path = 'F:\YutaImpedancetest';
impedance(id).probename = '64Ch Cambridge Neurotech 1 shank linear';
impedance(id).probelayout = 1+[22, 13, 15, 24, 11, 20, 7, 9, 21, 8, 25, 12, 10, 23, 14, 27, 26, 5, 18, 28, 3, 19, 31, 62, 0, 1, 30, 16, 2, 29, 6, 4, 36, 59, 44, 34, 61, 45, 33, 47, 17, 63, 32, 46, 60, 35, 56, 58, 40, 51, 49, 38, 53, 42, 57, 55, 43, 54, 39, 50, 52, 41, 48, 37]';
impedance(id).screwmoved = [1];

id = 5; % YutaMouse
impedance(id).rat = 'YutaMouse #6,#9';
impedance(id).measures = {'YMV06-day0','YMV06-day1','YMV09-08122017','YMV09-12122017'};
impedance(id).path = 'F:\YutaImpedancetest';
impedance(id).probename = '64Ch Cambridge Neurotech 1 shank linear';
impedance(id).probelayout = 1+[22, 13, 15, 24, 11, 20, 7, 9, 21, 8, 25, 12, 10, 23, 14, 27, 26, 5, 18, 28, 3, 19, 31, 62, 0, 1, 30, 16, 2, 29, 6, 4, 36, 59, 44, 34, 61, 45, 33, 47, 17, 63, 32, 46, 60, 35, 56, 58, 40, 51, 49, 38, 53, 42, 57, 55, 43, 54, 39, 50, 52, 41, 48, 37]';
impedance(id).screwmoved = [1];

id = 6; % MS14
impedance(id).rat = 'MS14';
impedance(id).measures = {'Impedance_MS14_15-01-2018','Impedance_MS14_17-01-2018','Impedance_MS14_21-01-2018','Impedance_MS14_22-01-2018','Impedance_MS14_25-01-2018','Impedance_MS14_05-02-2018','Impedance_MS14_06-02-2018'};
impedance(id).path = 'F:\IntanData\Peter_MS14\ImpedanceMeasures';
impedance(id).xmlfile = '';
impedance(id).probename = '2x 64Ch NeuroNexus poly3 probes';
impedance(id).probelayout = 1+[57, 50, 54, 61, 51, 56, 60, 48, 55, 63, 49, 59, 62, 53, 52, 58 ; 42, 36, 41, 47, 34, 44, 43, 33, 40, 39, 32, 45, 35, 37, 38, 46 ;23, 17, 20, 26, 21, 22, 28, 25, 18, 31, 29, 24, 30, 16, 19, 27 ; 8, 3, 7, 12, 2, 9, 13, 1, 6, 14, 0, 10, 15, 4, 5, 11 ; 121, 114, 118, 125, 115, 120, 124, 112, 119, 127, 113, 123, 126, 117, 116, 122 ; 106, 100, 105, 111, 98, 108, 107, 97, 104, 103, 96, 109, 99, 101, 102, 110 ; 87, 81, 84, 90, 85, 86, 92, 89, 82, 95, 93, 88, 94, 80, 83, 91 ; 72, 67, 71, 76, 66, 73, 77, 65, 70, 78, 64, 74, 79, 68, 69, 75]';
impedance(id).screwmoved = [];

% ImpedanceDataPath = 'G:\IntanData\Peter_MS9_ImpedanceMeasures\';
% Measures = {'MS9_Impedance1207','MS9_Impedance12072' ,'MS9_Impedance12121','MS9_Impedance12122','MS9_Impedance1213','MS9_Impedance12132','MS9_Impedance12133' ,'MS9_Impedance12134','MS9_Impedance1215','MS9_Impedance1216','MS9_Impedance1219','MS9_Impedance1220' ,'MS9_Impedance0105','MS9_Impedance0117','MS9_Impedance0130','MS9_Impedance_0131','MS9_Impedance_0201','MS9_Impedance_0207'};
% probe = 1+[22,27,13,14,15,23,24,10,11,12,20,25,7,8,9,21;26,4,5,6,18,29,28,2,3,16,19,30,31,1,62,0;36,58,59,56,44,35,34,60,61,46,45,32,33,63,47,17;40,37,51,48,49,41,38,52,53,50,42,39,57,54,55,43]';
% ScrewMoved = [3,8,9,10,11,12,13,15];

animal = 6;
rat = impedance(animal).rat;
Measures = impedance(animal).measures;
ImpedanceDataPath = impedance(animal).path;
probe = impedance(animal).probelayout;
probename = impedance(animal).probename;
ScrewMoved = impedance(animal).screwmoved;

cd(ImpedanceDataPath)
delimiter = ',';
startRow = 2;
formatSpec = '%s%s%s%s%s%s%s%s%[^\n\r]';
ImpedanceMagnitudeat1000Hzohms = [];
ImpedanceTimeStamp = [];
for i = 1:length(Measures)
    filename = [fullfile(ImpedanceDataPath, [Measures{i}, '.csv'])];
    fileID = fopen(filename,'r');
    dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    fclose(fileID);
    raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
    for col=1:length(dataArray)-1
        raw(1:length(dataArray{col}),col) = dataArray{col};
    end
    numericData = NaN(size(dataArray{1},1),size(dataArray,2));
    
    for col=[1,2,4,5,6,7,8]
        % Converts text in the input cell array to numbers. Replaced non-numeric
        % text with NaN.
        rawData = dataArray{col};
        for row=1:size(rawData, 1);
            % Create a regular expression to detect and remove non-numeric prefixes and
            % suffixes.
            regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
            try
                result = regexp(rawData{row}, regexstr, 'names');
                numbers = result.numbers;
                
                % Detected commas in non-thousand locations.
                invalidThousandsSeparator = false;
                if any(numbers==',');
                    thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                    if isempty(regexp(numbers, thousandsRegExp, 'once'));
                        numbers = NaN;
                        invalidThousandsSeparator = true;
                    end
                end
                % Convert numeric text to numbers.
                if ~invalidThousandsSeparator;
                    numbers = textscan(strrep(numbers, ',', ''), '%f');
                    numericData(row, col) = numbers{1};
                    raw{row, col} = numbers{1};
                end
            catch me
            end
        end
    end
    % Split data into numeric and cell columns.
    rawNumericColumns = raw(:, [1,2,4,5,6,7,8]);
    rawCellColumns = raw(:, 3);
    temp2 = cell2mat(rawNumericColumns(:, 4));
    ImpedanceMagnitudeat1000Hzohms(i,:) = temp2(end-length(probe(:))+1:end);
    test = dir(fullfile(ImpedanceDataPath, [Measures{i}, '.csv']));
    ImpedanceTimeStamp{i} = test.date;
end
colors = {'b','k','g','r','m','c','y',[0.5 0.5 0.5]};
figure
plot12 = [];
for j = 1:size(probe,2)
     plot12(:,j) = semilogy(ImpedanceMagnitudeat1000Hzohms(:,probe(:,j)),'color', colors{j}); hold on
     % semilogy(ScrewMoved,ImpedanceMagnitudeat1000Hzohms(ScrewMoved,probe(:,j)),'.','color',colors{j})
end
for j = 1:size(ImpedanceMagnitudeat1000Hzohms,1)
    ImpedanceMagnitudeat1000Hzohms2(j) = mean(ImpedanceMagnitudeat1000Hzohms(j,ImpedanceMagnitudeat1000Hzohms(j,:)<6000000));
end
semilogy(ImpedanceMagnitudeat1000Hzohms2,'k','linewidth',2);
semilogy([1,length(Measures)],[1,1]*1000000,'--','color',[0.1 0.1 0.1]),
axis tight

% gridxy(3,'linestyle','--')
xlabel('Recording time'),ylabel('Impedance (Ohm)'), title({['Impedance measures (' rat ')'] probename})
xticks([1:length(Measures)]), xticklabels(Measures), xtickangle(45)
%legend([plot12(1,:),plot12(2,:),plot12(3,:),plot12(4,:),plot12(5,:),plot12(6,:)],'Shank #1','Shank #2','Shank #3','Shank #4','Shank #5','Shank #6','Shank #7','Shank #8')
