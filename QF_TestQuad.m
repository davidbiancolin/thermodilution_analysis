
function Xest = QF_TestQuad(X)

load('algoFlow.mat','algoFlow')

gNodes = algoFlow{1};

x1n = algoFlow{2};
x2n = algoFlow{3};
x3n = algoFlow{4};

% Speed is x2
x2 = X(:,2);
l1 = ((x2-x2n(2))/(x2n(1) - x2n(2))) .* ((x2-x2n(3))/(x2n(1)-x2n(3)));
l2 = ((x2-x2n(1))/(x2n(2) - x2n(1))) .* ((x2-x2n(3))/(x2n(2)-x2n(3)));
l3 = ((x2-x2n(1))/(x2n(3) - x2n(1))) .* ((x2-x2n(2))/(x2n(3)-x2n(2)));

E2 = [l1 l2 l3];

% MCbin is x3
x3 = X(:,3);
l1 = ((x3-x3n(2))/(x3n(1) - x3n(2)));
l2 = ((x3-x3n(1))/(x3n(2) - x3n(1)));

E3 = [l1 l2]; 

En = [];
for jdx=1:size(E2,2)
    for kdx=1:size(E3,2)
        En = [En, E2(:,jdx).*E3(:,kdx)];
    end
end

% Create saddle points
x1np = En * x1n;

% Motor current is x1 variable
x1 = X(:,1);

% Separate into three elements
ep1 = (x1 - x1np(:,1)) ./ (x1np(:,2) - x1np(:,1));
ep2 = (x1 - x1np(:,2)) ./ (x1np(:,3) - x1np(:,2));
ep3 = (x1 - x1np(:,3)) ./ (x1np(:,4) - x1np(:,3));

ep = ep1;
flag_ep2 = (x1 > x1np(:,2) & x1 <= x1np(:,3));
flag_ep3 = (x1 > x1np(:,3));
flag_ep1 = ~(flag_ep2 | flag_ep3);
ep(flag_ep2) = ep2(flag_ep2);
ep(flag_ep3) = ep3(flag_ep3);

% Generate basis functions
l11 = 1 - ep;
l12 = ep;

l21 = 1 - ep;
l22 = ep;

l31 = 1 - ep;
l32 = ep;

% Create basis matrix
E1 = [l11 l12 l22 l32];

% Clear redundant data, populate overlapping columns
E1(flag_ep1,3:4) = 0;
E1(flag_ep2,1) = 0;
E1(flag_ep2,4) = 0;
E1(flag_ep3,1:2) = 0;

E1(flag_ep2,2) = l21(flag_ep2);
E1(flag_ep3,3) = l31(flag_ep3);

% Create scaled basis matrix
B1 = E1;

E = [];
for idx=1:size(E1,2)
    for jdx=1:size(E2,2)
        for kdx=1:size(E3,2)
            E = [E, B1(:,idx).*E2(:,jdx).*E3(:,kdx)];
        end
    end
end 




Xest = E * gNodes;


end