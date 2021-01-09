function  [slope,offset,R_value] = CircularLinearRegression(circ,lin,sign)

switch nargin
    case 2
        a = -0.1:0.0001:0.1;
        sign = 0;
end

if sign > 0
    a = 0:0.0001:0.1;
    a = 0:0.0001:0.1;
end
if sign < 0
    a = -0.1:0.0001:0;
end

cosinepart=zeros(length(a),1);
sinepart=zeros(length(a),1);
R=zeros(length(a),1);

for i=1:length(a)
    cosinepart(i)=sum(cos(circ(:)-(2*pi*a(i)*lin(:))));
    sinepart(i)=sum(sin(circ(:)-(2*pi*a(i)*lin(:))));
    firstterm=(cosinepart(i)/length(circ))^2;
    secondterm=(sinepart(i)/length(circ))^2;
    R(i)=sqrt(firstterm+secondterm);
end
slope=a(R==max(R));
offset=atan2(sinepart(R==max(R)),cosinepart(R==max(R)));
R_value = max(R);
%figure, plot(a,R);
