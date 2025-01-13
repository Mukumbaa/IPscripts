%% Region growing
function [g, NR, SI, TI] = regiongrow(f, S, T)

    f = double(f);
    % if S is a scalar, obtain the seed image
    if numel(S) == 1
        SI = f == S;
        S1 = S;
    else
        SI = bwmorph(S, 'shrink', Inf);
        J = find(SI);
        S1 = f(J); % Array of seed values
    end
    TI = false(size(f));
    for K = 1:length(S1)
        seedvalue = S1(K);
        S = abs(f - seedvalue) <= T;
        TI = TI | S;
    end
    [g, NR] = bwlabel(imreconstruct(SI, TI));
end


f = imread('gel-image.tif');
subplot(1,2,1); imshow(f);
xlabel('Original image');
impixelinfo();

% Region growing
f1 = double(f);
S = max(f1(:));
T = 0.5*max(f1(:));

% Se f è l'immagine di input, i parametri s e t sono invece i valori che 
% determinano rispettivamente l'intensità dei semi e la massima differenza
% tollerabile tra il valore del pixel in esame e quello del seme. Come 
% output, si ha l'immagine segmentata g, il numero nr di regioni individuate, 
% l'immagine originale si coi semi e quella ti contenente i pixel che 
% soddisfano la sogliatura.
% Evidenziare i bordi

[g, NR, SI, TI] = regiongrow(f, S, T);

subplot(1,2,2), imshow(g);
xlabel('Regiongrow segmentation');