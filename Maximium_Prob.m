function [MProb1] = Maximium_Prob(G)
% Inverse Difference Movement
A=G(:,:,1)./sum(sum(G(:,:,1)));
B=G(:,:,2)./sum(sum(G(:,:,2)));
C=G(:,:,3)./sum(sum(G(:,:,3)));
D=G(:,:,4)./sum(sum(G(:,:,4)));
MProb1=max([max(A) max(B) max(C) max(D)]);
end
