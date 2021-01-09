angles = SpinCalc('QtoEA321',[Optitrack.Xr,Optitrack.Yr,Optitrack.Zr,Optitrack.Wr],1e-5,1);
v = [0;-6;0];
y = [];
for j = 1:size(angles,1)
    Rx = rotx(-angles(j,3));
    Ry = roty(-angles(j,2));
    Rz = rotz(-angles(j,1));
    y(j,:) = Rx*Ry*Rz*v;
end
position = 100*[Optitrack.X,Optitrack.Y,Optitrack.Z];
position2 = position+y;

i = 55500;
ii = 2000;
figure, 
subplot(2,1,1)
plot3(angles([1:ii]+i,3),angles([1:ii]+i,2),angles([1:ii]+i,1),'-o'),xlabel('Yaw (X)'),ylabel('Picth (Y)'),zlabel('Roll (Z)')
subplot(2,1,2)
plot3(position([1:ii]+i,3),position([1:ii]+i,2),position([1:ii]+i,1),'-o'), hold on
plot3(position2([1:ii]+i,3),position2([1:ii]+i,2),position2([1:ii]+i,1),'-or'),xlabel('X'),ylabel('Y'),zlabel('Z')
