%% Lezione 1 lab

f = imread('cameraman.tif');
imshow(f);
fc = f(30:150, 30:200);
imshow(fc);
fp = f(end:-1:1, :);
imshow(fp);
fs = f(1:2:end, 1:2:end);
imshow(fs);
plot(f(150, :));

% Ramp
clc
clear all
close all

for i=1:256
    for j=1:256
        A(i,j) = j-1;
    end
end

A = mat2gray(A);
imshow(A);

% Circle

for i=1:256
    for j=1:256
        dist  = sqrt((i -128)^2 + (j-128)^2);
        if dist < 80 B(i,j) = 255;
        else B(i,j) = 0;
        end
    end
end
image(B);
colormap(gray(256));
axis('image');

% Graded circle
C = zeros(256,256);
for i=1:256
    for j=1:256
        C(i,j) = A(i,j) .* B(i,j)./255;
    end
end
imshow(C, []);

% RGB canals
clc
close all
clear all

f = imread('peppers.png');
figure, imshow(f);

A = f;
Red_A = A;
Red_A(:,:, 2:3) = 0;
Green_A = A;
Green_A(:,:,1) = 0;
Green_A(:,:,3) = 0;
Blu_A = A;
Blu_A(:,:, 1:2) = 0;
subplot(1,3,1), imshow(Red_A);
subplot(1,3,2), imshow(Green_A);
subplot(1,3,3), imshow(Blu_A);

% Problem: realize a new image from an existing BW image with
% each pixel shifted one position to the right 
clc
close all
clear all

F = imread('cameraman.tif');
subplot(1,2,1); imshow(F);

% Elegant solution
D = [F(:, 2:1:end) F(:, 1)];
subplot(1,2,2); imshow(D);
impixelinfo();

clc
close all
clear all

F = imread('cameraman.tif');
figure; imshow(F);
impixelinfo();
[M, N] = size(F);
for i = 1:M
    for j = 1:N
        if i == 256 
            C(i,j) = F(1, j);
        else 
            C(i,j) = F(i+1, j);
        end
    end
end
figure; imshow(C);
impixelinfo();

%% Lezione 2 lab

clc
close all
clear all

I = imread('pout.tif');

subplot(1,3,1); imhist(I);
[counts, binLocations] = imhist(I);

I = imread('football.jpg');
subplot(1,3,2); imshow(I);

Red = I(:, :, 1);
Green = I(:, :, 2);
Blue = I(:, :, 3);

% Histogram values for each channel

[yRed, x] =  imhist(Red);
[yGreen, x] =  imhist(Green);
[yBlue, x] =  imhist(Blue);

% Plot them togheter in one plot
subplot(1,3,3),
plot(x, yRed, 'Red', x, yGreen, 'Green', x, yBlue, 'Blue');

% Contrast

clc
clear all
close all

I = imread('pout.tif');
subplot(1,2,1), imshow(I);

[min(I(:)) max(I(:))]

I = im2double(I);

[min(I(:)) max(I(:))]

J = imadjust(I, [min(I(:)) max(I(:))], [0 1]);
subplot(1,2,2), imshow(J);

% Color image

clc
clear all
close all

RGB = imread('football.jpg');
subplot(1,2,1), imshow(RGB);

RGB2 = imadjust(RGB, [.2 .3 0; .6 .7 1], []);
subplot(1,2,2), imshow(RGB2);

% Histogram equalization
clc
clear all
close all

I = imread('tire.tif');
subplot(2,2,1), imshow(I);
subplot(2,2,2), imhist(I);

J = histeq(I);
subplot(2,2,3), imshow(J);
subplot(2,2,4), imhist(J);

% Using threshold to isolate the coins

clc
close all
clear all

A = imread('eight.tif');
subplot(2,2,1); imshow(A);
subplot(2,2,2); imhist(A);

[M N p] = size(A);

soglia = input('soglia: ');%180
B = A;
B(A>=soglia) = 0;
subplot(2,2,3), imshow(B);
xlabel(['soglia = ', num2str(soglia)]);
subplot(2,2,4), imhist(B);

% Highlighting inner and outer borders with threshold
clc
close all
clear all

A = imread('square.jpg');
subplot(3,2,1); imshow(A);title('original image');
subplot(3,2,2); imhist(A);title('Histogram O I');

B = imadjust(A);
subplot(3,2,3), imshow(B);title('image adjusted');
subplot(3,2,4), imhist(B);title('Histogram A I');

soglia1 = input('Soglia minima : ');%50
soglia2 = input('Soglia massima : ');%175
C = B;
%C(B <= soglia1) = 1;
%C(B >= soglia2) = 1;
C = (B>=soglia1) & (B <= soglia2);
subplot(3,2,5), imshow(C);title('Objects Borders');
% sovr = imadd(A, uint8(C));
%subplot(3,2,6), imshow(sovr); title('Boundary');
subplot(3,2,6), imshow(A + im2uint8(C)); title('Boundary');

% Isolating "croatia" with threshold
clc
close all
clear all

X = imread('croatia.jpg');
subplot(2,2,1), imshow(X);
subplot(2,2,2), imhist(X);
soglia1 = 180;
soglia2 = 250;
BB = (X >= 180) & (X <= 250);
subplot(2,2,3), imshow(BB);
subplot(2,2,4), imhist(BB);

% Highlighting objects in the bottle
clc
close all
clear all

Y = imread('bottle.tif');
subplot(2,2,1), imshow(Y);
subplot(2,2,2), imhist(Y);
soglia = 9;
YY = (Y >= soglia);
subplot(2,2, [3 4]), imshow(YY);

%% Lezione 3 lab

%

clc
close all
clear all

% averaging filter
clc
clear all
close all

I = imread('blood1.tif');
subplot(2,3,1), imshow(I), title('Original image');

h3 = ones(3,3) / 9;
h5 = ones(5,5) / 25;
h7 = ones(7,7) / 49;

J = imfilter(I, h3);
subplot(2,3,2); imshow(J); title('Filtered image f 3x3');

L = imfilter(I, h5);
subplot(2,3,3); imshow(L); title('Filtered image f 5x5');

M = imfilter(I, h7);
subplot(2,3,4); imshow(M); title('Filtered image f 7x7');

% fspecial
clc
close all
clear all

I = imread('moon.tif');
subplot(2,4,1), imshow(I);
title('Original image');

H = fspecial('motion', 50, 45);
MotionBlur = imfilter(I, H);
subplot(2,4,2); imshow(MotionBlur); 
title('Motion Blurred image');

H1 = fspecial('disk', 10);
blurred = imfilter(I, H1);
subplot(2,4,3); imshow(blurred); 
title('Blurred image');

H2 = fspecial('unsharp');
sharped = imfilter(I, H2);
subplot(2,4,4); imshow(sharped); 
title('sharpened image');

H4  = fspecial('laplacian', 0.0);
H8 = [1 1 1; 1 -8 1; 1 1 1];
I1 = im2double(I);
L4 = I1 - imfilter(I1, H4);
L8 = I1 - imfilter(I1, H8);
subplot(2,4,5); imshow(L4); 
title('Enhanced by H4');
subplot(2,4,6); imshow(L8); 
title('Enhanced by H8');

% Median filter
clear
clc
close all

I = imread('eight.tif');
subplot(3, 3, 1), imshow(I);

J = imnoise(I, 'salt & pepper', 0.02);
subplot(3, 3, 2), imshow(J);

% filtro mediano
K = medfilt2(J);
subplot(3, 3, 3), imshow(K);

% filtro media
h3 = ones(3, 3) / 9;
M = imfilter(J, h3);
subplot(3, 3, 4), imshow(M);

% ordfilt2
I = imread('circuit.tif');
subplot(3, 3, 5); imshow(I);
J = imnoise(I,'salt & pepper',0.02);
subplot(3, 3, 6), imshow(J);
K = ordfilt2(J,5,ones(3,3));
subplot(3, 3, 7), imshow(K);

% Telescopio
clc
close all
clear all

I = imread('telescopio.jpg');
subplot(1,3,1); imshow(I);

H = ones(15, 15) /225;
G = imfilter(I, H);
subplot(1,3,2); imshow(G);
th = max(G(:))*0.25;
G1 = G>th;

A = I;
A(G1==0) = 0;
subplot(1,3,3); imshow(A);

% Rumore Gaussiano
% Leggere l’immagine 'etna.jpg' e applicare un rumore gaussiano con media zero e varianza 0.02.
% Considerare l’immagine come se fosse composta da quattro regioni distinte:
% alto-sinistra, alto-destra, basso-sinistra, basso-destra
% a) lasciare inalterato il quadrante alto-sinistra;
% b) filtrare il quadrante alto-destra con il filtro media;
% c) filtrare il quadrante basso-sinistra con il filtro gaussiano;
% d) filtrare il quadrante basso-destra con il filtro mediano.
% Visualizzare il risultato in un’unica immagine.

clc
clear all
close all

I = imread('etna.jpg');
R=imnoise(I,'gaussian', 0, 0.02);
figure, imshow(R);

[M, N, C]=size(R);
a = 3; b = 3;
% NE
h=fspecial('average', [a b]);
R(1:M/2,N/2+1:N,1:3)=imfilter(R(1:M/2,N/2+1:N,1:3), h);
% SO
h1 = fspecial('gaussian', [3 3], 1.5);
R(M/2+1:M, 1:N/2, 1:3) = imfilter(R(M/2+1:M, 1:N/2, 1:3), h1);
% SE
R(M/2+1:M,N/2+1:N, 1) = medfilt2(R(M/2+1:M,N/2+1:N,1));
R(M/2+1:M,N/2+1:N, 2) = medfilt2(R(M/2+1:M,N/2+1:N,2));
R(M/2+1:M,N/2+1:N, 3) = medfilt2(R(M/2+1:M,N/2+1:N,3));

figure; imshow(R);

clc
clear all
close all

I = imread('etna.jpg');
R=imnoise(I,'gaussian', 0, 0.4);
subplot(2,2,1); imshow(I);

subplot(2,2,2); imshow(R);

F = R - I;
subplot(2,2,3); imshow(F);

%% Lezione 4 lab

% Ridimensionare senza imresize
clear
clc
close all

f = imread('lena.gif');
figure, imshow(f);

[M, N, D] = size(f);

k = 2;
g = f(1:k:M, 1:k:N);
figure, imshow(g);

% Ridimensionamento con imresize
clear
clc
close all

f = imread('lena.gif');
figure, imshow(f);

k= 0.75;

g = imresize(f, k, 'bilinear');
figure, imshow(g);

% Trasformazioni geometriche
clear
clc
close all

A = imread('lena.gif');
T = [cos(pi/4) sin(pi/4) 0; -sin(pi/4) cos(pi/4) 0; 0 0 1];

tform = maketform('affine', T);

G = imtransform(A, tform);
tform2 = affine2d(T);
G1 = imwarp(A, tform2);
subplot(1,4,1), imshow(A);
title('Immagine non ruotata');
subplot(1,4,2), imshow(G);
title('Rotazione 45° con maketform');

subplot(1,4,3), imshow(G1);
title('Rotazione 45° con affine2d e imwarp');
alfa = -45;
G2 = imrotate(A, alfa);
subplot(1,4,4), imshow(G2);
title('Rotazione 45° con imrotate');

% Ingrandimento di un particolare
clc
clear all
close all

A = imread('lena.gif');
figure, imshow(A); title('Lena originale');
impixelinfo();
% 240, 250 - 290, 290
A1 = A(240:290, 250:290);
figure, imshow(A1); title('Lena cropped');

T = [2 0 0; 0 2 0; 0 0 1];
Tform = maketform('affine', T);
A2 = imtransform(A1, Tform, 'nearest');
subplot(1,3,1), imshow(A2); title('Lena eye nearest');

Tform2 = maketform('affine', T);
A3 = imtransform(A1, Tform2, 'bilinear');
subplot(1,3,2), imshow(A3); title('Lena eye bilinear');

Tform3 = maketform('affine', T);
A4 = imtransform(A1, Tform2, 'bicubic');
subplot(1,3,3), imshow(A4); title('Lena eye bicubic');

% Interpolations combo
clc
clear all
close all

A = checkerboard(50);
A(A>0) = 1;
figure, imshow(A);

B = rot_dist(A, pi/4, 0.5);
figure, imshow(B);

% white noise
clc
close all
clear all

% Original image
f = imread('lena.gif');
subplot(2,2,1), imshow(f);

% standard deviation
d = 20;

% additive noise 
f = double(f);
n = d * randn(size(f));
noisy = f + n;
figure(1),
subplot(2,2,2), imshow(noisy, []);

% Mobile media filter loop
K = [3 5 7 9 11 13];
for k= 1 : length(K)
    h = fspecial('average', [K(k) K(k)]);
    g = imfilter(noisy, h, 'symmetric', 'same');
    MSE(k) = mean2((f-g).^2);
    figure(2),
    subplot(1,3,1); imshow(f, []); 
    title('original');
    subplot(1,3,2); imshow(noisy, []); 
    title('noisy');
    subplot(1,3,3); imshow(g, []); 
    title(['K= ', num2str(K(k)), ' MSE = ', num2str(MSE(k),4);]);
    
    MSE1(k)= immse(double(f), g);
    pause
end

figure(3); plot(K, MSE, 'r-*');
xlabel('windows dimension'); ylabel('MSE');

%% Lezione 5 lab
% esercizio 1 - isolare un pixel ed evidenziarlo
clc
clear all
close all

I = imread('test_pattern_with_single_pixel.tif');
subplot(2,2,1), imshow(I);

w = [ -1 -1 -1; -1 8 -1; -1 -1 -1];
g = abs(imfilter(double(I),w));
T = max(g(:));
g = g>=T;
subplot(2,2,2), imshow(g);

g1 = imdilate(g, strel('disk', 3));
subplot(2,2,3), imshow(I + im2uint8(g1));

% Esercizio 2 - isolare una linea 
clc
clear all
close all

I = imread('wirebond-mask.tif');
subplot(3,2,1), imshow(I);

w = [ 2 -1 -1; -1 2 -1; -1 -1 2];
g = imfilter(double(I),w);
subplot(3,2,2), imshow(g, []);

gtop=g(1:120,1:120);
subplot(3,2,3), imshow(gtop, [ ]);

gbot=g(end-119:end, end-119:end);
subplot(3,2,4), imshow(gbot, [ ]);

g = abs(g);
subplot(3,2,5), imshow(g, [ ]);

T = max(g(:));
g = g>=T;
subplot(3,2,6), imshow(g);

% Esercizio 3 - filtro sobel
clc
close all
clear all

I = imread('bld.tif');
subplot(2,3,1), imshow(I);title('Original');

[bv, t] = edge(I, 'sobel', 'vertical');
subplot(2,3,2), imshow(bv);title('Sobel vertical');

bvt = edge(I, 'sobel', 0.15, 'vertical');
subplot(2,3,3), imshow(bvt);title('Sobel vertical with 0.15 thrs');

bboth = edge(I, 'sobel', 0.15);
subplot(2,3,4), imshow(bboth);title('Sobel both with 0.15 thrs');

w45=[-2 -1 0; -1 0 1; 0 1 2];

b45 = imfilter(double(I), w45, 'replicate');
T=0.3*max(abs(b45(:)));
b45=b45>=T;
subplot(2,3,5), imshow(b45);title('Sobel 45°');

w_45=[0 1 2; -1 0 1; -2 -1 0];

b_45 = imfilter(double(I),w_45,'replicate');
T=0.3*max(abs(b_45(:)));
b_45=b_45>=T;
subplot(2,3,6), imshow(b_45);title('Sobel -45°');

% Esercizio 4 - Sobel, LoG and Canny detectors 
clc
close all
clear all

I = imread('bld.tif');
subplot(3,3,1), imshow(I);title('Original');

[b_sobel_default, ts] = edge(I, 'sobel');
subplot(3,3,2), imshow(b_sobel_default);title('Sobel default');
[b_log_default, tlog] = edge(I, 'log');
subplot(3,3,3), imshow(b_log_default);title('LoG default');

[b_canny_default, tc] = edge(I, 'canny');
subplot(3,3,4), imshow(b_canny_default);title('Canny default');

b_sobel_best = edge(I, 'sobel', 0.05);
subplot(3,3,5), imshow(b_sobel_best);title('Sobel best (th=0.05)');

b_log_best = edge(I, 'log', 0.003, 2.25);
subplot(3,3,6), imshow(b_log_best);title('LoG best (th = 0.05, σ = 2.25)');

b_canny_best = edge(I, 'canny', [0.04 0.10], 1.5);
subplot(3,3,7), imshow(b_canny_best);title('Canny best (th = [0.04 0.10], σ = 1.5)');

Temp = im2uint8(imcomplement(b_canny_best));
Temp(Temp==255) = 1;
sovr = I.*Temp;
subplot(3,3,8), imshow(sovr);xlabel('Canny sovrimpressed on original img', 'FontWeight','bold');

% Esercizio 5 - 

clear
clc
close all

I = imread('Lines1.jpg');
subplot(2,3,1), imshow(I);
% w=[-1 -1 -1; 2 2 2; -1 -1 -1]; % horizontal
w = [1 1 1; 0 0 0; -1 -1 -1];
g=imfilter(double(I),w);
subplot(2,3,2), imshow(uint8(g), [ ]);
% w1=[-1 2 -1; -1 2 -1; -1 2 -1]; % vertical
w1=[1 0 -1; 1 0 -1; 1 0 -1]; % vertical
g1=imfilter(double(I),w1);
subplot(2,3,3), imshow(uint8(g1), [ ]);


I1 = imread('Lines2.jpg');
subplot(2,3,4), imshow(I1);
%w2=[-1 -1 -1; 2 2 2; -1 -1 -1]; % horizontal
w2=[1 1 1; 0 0 0; -1 -1 -1]; % horizontal
g2=imfilter(double(I1),w2);
subplot(2,3,5), imshow(uint8(g2), [ ]);
% w3=[-1 2 -1; -1 2 -1; -1 2 -1]; % vertical
w3=[1 0 -1; 1 0 -1; 1 0 -1]; % vertical
g3=imfilter(double(I1),w3);
subplot(2,3,6), imshow(uint8(g3), [ ]);

% Esercizio 6 - 

clear
clc
close all

I = imread('scale.jpg');
subplot(3,3,1); imshow(I); title('Originale');

BW1 = edge(I,'sobel');
subplot(3,3,2); imshow(BW1); title('Sobel');
BW2 = edge(I,'prewitt');
subplot(3,3,3); imshow(BW2); title('Prewitt');
BW3 = edge(I,'roberts');
subplot(3,3,4); imshow(BW3); title('Roberts');
BW4 = edge(I,'log');
subplot(3,3,5); imshow(BW4); title('LoG');
BW5 = edge(I,'zerocross');
subplot(3,3,6); imshow(BW5); title('Zerocross');
BW6 = edge(I,'canny');
subplot(3,3,7); imshow(BW6); title('Canny');

BW6int = im2uint8(BW6);
C = I + BW6int;
subplot(3,3,8), imshow(C);

C1 = I; C1(BW6) = 255;
subplot(3,3,9), imshow(C1);

%% Lezione 6 lab
%Esercizio 1 - erosione e dilatazione

close all %chiusura finestre aperte da prima
clear %pulizia variabili presenti da prima
clc %pulizia command window

IM = imread("wirebond-mask.tif");
subplot(2,3,1),imshow(IM);title("Originale");
SE = strel("square", 5);
IM2 = imdilate(IM,SE);
subplot(2,3,2), imshow(IM2); title("dilatazione");

i = imread("small-squares.tif");
subplot(2,3,4), imshow(i);title("Originale");
se = strel("square", 13);
fe = imerode(i, se);
subplot(2,3,5), imshow(fe);title("erosione");

fed = imdilate(fe, se);
subplot(2,3,6), imshow(fed); title("dilatazione dopo erosione");

% Esercizio 2 - Opening and Closing

close all %chiusura finestre aperte da prima
clear %pulizia variabili presenti da prima
clc %pulizia command window

i = imread("shapes.tif");
subplot(2,3,1); imshow(i); title('Originale');

se = strel("square", 20);
i2 = imopen(i, se);
subplot(2,3,2), imshow(i2), title("opening");
i2 = imclose(i2, se);
subplot(2,3,3), imshow(i2), title("closing di opening")

i = imread("noisy-fingerprint.tif");
subplot(2,3,4), imshow(i); title('Originale');

%dobbiamo eliminare il rumore e riempire i buchini
se = strel("sphere", 1);
i2 = imopen(i, se);
subplot(2,3,5), imshow(i2), title("opening");
i2 = imclose(i2, se);
subplot(2,3,6), imshow(i2), title("closing dell'opening");

% Esercizio 3 Hit or Miss transformation
close all %chiusura finestre aperte da prima
clear %pulizia variabili presenti da prima
clc %pulizia command window

i = imread("small-squares.tif");
subplot(1,2,1), imshow(i); title('Originale');

% hit and miss con la seguente regola: un elemento deve avere pixel vicini pari ad uno a est e sud,
% e pari a zero a nord, nord est, nord ovest, ovest e sud ovest:
se1 = [0 0 0;
       0 1 1;
       0 1 1]; % (il pixel in basso a destra non ci interessa)
se2 = ~se1; % il secondo SE se sarà il complemento del primo

bw = bwhitmiss(i, se1, se2);
subplot(1,2,2), imshow(bw);title('Hit or Miss transform');

% Esercizio 4 Labels and Areas

close all %chiusura finestre aperte da prima
clear %pulizia variabili presenti da prima
clc %pulizia command window

%voglio visualizzare singolarmente ogni oggetto presente nell'immagine
i = imread("ten-objects.tif");
[L, NUM] = bwlabel(i);
subplot(2,NUM/2+1,1), imshow(i)
subplot(2,NUM/2+1,2), imshow(label2rgb(L))

numpixel = zeros(1,NUM); %vettore dove mettere il numero di pixel di ogni componente trovata

for i = 1:NUM
    C = L==i; %si prendono i pixel appartenenti alla componente in analisi
    subplot(2,NUM/2+1,i+2), imshow(C)
    xlabel(["componente: ", num2str(i)])
    numpixel(i) = sum(C(:)); %si sommano tutti i pixel bianchi della componente
end

%fprintf(num2str(max(numpixel))) %numero di pixel dell'oggetto più grande
numpixel

%% Lezione 7 lab

% Esercizio 1 - Mass center
clc
close all
clear all

f= imread('ten-objects.tif');
imshow(f);
[L, n]=bwlabel(f);

hold on; % to plot on the top of the image
for k=1:n
    [r,c] = find(L==k);
    rbar=mean(r);
    cbar=mean(c);
    plot(cbar, rbar, 'Marker', 'o', 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r', 'Markersize', 10);
    plot(cbar, rbar, 'Marker', '*', 'MarkerEdgeColor', 'y');
end

% Esercizio 2 - Thinning
clc
close all
clear all

f=imread('noisy-fingerprint.tif');
subplot(2,3,1), imshow(f); title('Input');
se = strel('square',3);
fo = imopen(f,se);
subplot(2,3,2), imshow(fo);title('Opening');
foc = imclose(fo,se);
subplot(2,3,3), imshow(foc);title('Closing dell''Opening');
g1 =bwmorph(foc,'thin',1);
subplot(2,3,4), imshow(g1);title('Thinning 1 time');
g2 =bwmorph(foc,'thin',2);
subplot(2,3,5), imshow(g2);title('Thinning 2 times');
ginf =bwmorph(foc,'thin',Inf);
subplot(2,3,6), imshow(ginf);title('Thinning until stability');

% Esercizio 3 - Skeleton
clc
close all
clear all

f=imread('bone.tif');
subplot(2,3,1), imshow(f)
fs = bwmorph(f,'skel',Inf);
subplot(2,3,2), imshow(fs);
for k=1:5
    fs = bwmorph(fs,'spur');
end
subplot(2,3,3), imshow(fs);
fc = ~fs;
subplot(2,3,4), imshow(fc);
ft = f & fc;
subplot(2,3,5), imshow(ft);

% Esercizio 4 - Morphological reconstruction
clc
close all
clear all

marker =imread('recon-marker.tif');
mask = imread('recon-mask.tif');

subplot(2,2,1), imshow(mask);title('Mask');
subplot(2,2,2), imshow(marker);title('Marker');

recon = imreconstruct(marker, mask);
subplot(2,2,3), imshow(recon);title('Reconstruct Grayscale');

mm=marker==255;
mm= uint8(mm*255);
recon = imreconstruct(mm, mask);
subplot(2,2,4), imshow(recon);title('Reconstruct Binary');

% Esercizio 5 - Morphological reconstruction
% Character relevation
clc
close all
clear all
% Characters with vertical strokes
f=imread('book-text.tif');
subplot(3,2,1), imshow(f);title('Original image');
fe=imerode(f,ones(51,1));
subplot(3,2,2), imshow(fe);title('Erosion');
fo=imopen(f, ones(51,1));
subplot(3,2,3), imshow(fo);title('Opening');
fobr=imreconstruct(fe,f);
subplot(3,2,4), imshow(fobr);title('Reconstruction (vertical strokes)');

% Esercizio 6 - Border cleaning
g = imclearborder(f);
subplot(3,2,5), imshow(g);title('Input without border characters');
subplot(3,2,6), imshow(f - g);title('Difference');

% Esercizio 7
% Segmentare l'immagine gel-image.tif mediante
% a) la trasformata watershed con markers;
% b) il metodo region growing;
% c) il metodo split-and-merge

clc
close all
clear all

f= imread('gel-image.tif');
subplot(2,2,1); imshow(f);
xlabel('Original image');
impixelinfo();

% watershed con marker
g = watershed_marker(f);
subplot(2,2,2), imshow(g);
xlabel('Watershed with marker segmentation');

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

subplot(2,2,3), imshow(g);
xlabel('Regiongrow segmentation');

% Split e merge
f2 = double(f);
g = splitmerge(f2, 2, @predicate);
subplot(2,2,4), imshow(g);
xlabel('Split merge segmentation');

%% Lezione 8 lab

% Esercizio 1 - Morphological Gradient

clc
close all
clear all

f = imread('aerial.tif');
subplot(2,2,1), imshow(f);title('original');

se = strel('square', 3);
f1 = imdilate(f, se);
subplot(2,2,2), imshow(f1);title('dilated');

f2 = imerode(f, se);
subplot(2,2,3), imshow(f1);title('eroded');

f3 = f1 - f2;
subplot(2,2,4), imshow(f3);title('Morph Gradient');

% Esercizio 2 - Smoothing morfologico
clc
close all
clear all

f = imread('dowels.tif');
subplot(1,2,1), imshow(f);title('original');

for i = 2:5
    se = strel('disk', i);
    f1 = imerode(imdilate(f, se), se);
    subplot(1,2,2), imshow(f1);title(['Clos->Open se with radius= ', num2str(i)]);
    pause
end

% Esercizio 2 - Top-hat transformation

close all %chiusura finestre aperte da prima
clear %pulizia variabili presenti da prima
clc %pulizia command window

f = imread("rice1.tif");
subplot(2,3,1), imshow(f);

t = 136;
ft1 = (f>=t);
subplot(2,3,2), imshow(ft1) %con la soglia sull'immagine si ottengono artefatti o si perdono pezzi dei chicchi

se = strel("disk",10);
op = imopen(f,se);
subplot(2,3,3), imshow(op)

tophat = f - op;
subplot(2,3,4), imshow(tophat)

t = 64; %(serve una soglia più piccola perché la top hat diminuisce la luminosità)
ft2 = (tophat>=t);
subplot(2,3,5), imshow(ft2) %con la soglia applicata alla top hat si ottiene il risultato migliore

% Esercizio 4 - Granulometry

close all %chiusura finestre aperte da prima
clear %pulizia variabili presenti da prima
clc %pulizia command window

f = imread("dowels.tif");
subplot(2,3,1), imshow(f)
t = 128;
a = zeros(30,1); %vettore contenente le aree di ogni iterazione
for i=1:30
    %si creano elementi strutturanti sempre più grandi e si salva ogni volta l'area nel vettore
    se = strel("disk",i);
    op = imopen(f,se);
    bw = op>=t;
    a(i) = sum(bw(:)); %(l'area è data dalla somma dei pixel che rispettano la soglia)
end

subplot(2,3,2), plot(a); %(grafico andamento area)
subplot(2,3,3), plot(abs(diff(a))); %(grafico andamento differenze)
se = strel("disk",3);
smooth = imclose(imopen(f,se),se);
subplot(2,3,4), imshow(smooth) %(facendo la stessa cosa sull'immagine smooth si avranno risultati diversi)

for i=1:30
    se = strel("disk",i);
    op = imopen(smooth,se);
    bw = op>=t;
    a(i) = sum(bw(:));
end

subplot(2,3,5), plot(a);
subplot(2,3,6), plot(abs(diff(a)));

% Esercizio 5 - Reconstruction

close all %chiusura finestre aperte da prima
clear %pulizia variabili presenti da prima
clc %pulizia command window

% Applicare la ricostruzione morfologica per elaborare una immagine con 
% sfondo complesso. Isolare i tasti dell immagine 'calculator.tif' e 
% mostrarli su uno sfondo omogeneo.

A = imread('calculator.tif');
subplot(2,4,1); imshow(A);title('Original');
xlabel('A');
% opening by reconstruction
erosion = imerode(A, ones(1,71));
reconstruction1 = imreconstruct(erosion, A);
subplot(2,4,2); imshow(reconstruction1); 
title('Opening by reconstruction of A');
xlabel('B');
% opening
opening = imopen(A, ones(1,71));
subplot(2,4,3); imshow(opening);
title('Opening of A');xlabel('C');
% tophat by reconstruction
tophatrecon = A - reconstruction1;
subplot(2,4,4); imshow(tophatrecon);
title('Tophat by reconstruction');
xlabel('D');
% tophat
tophat = A - opening;
subplot(2,4,5); imshow(tophat);
title('Tophat of C'); xlabel('E');
% opening by reconstruction di tophatre con con linea orizzontale
reconstruction2 = imreconstruct(imerode(tophatrecon, ones(1,11)), tophatrecon);
subplot(2,4,6); imshow(reconstruction2)
title('Opening by reconstruction of D'); xlabel('F');
% dilatazione di reconstruction 2 con linea orizzontale
dilation = imdilate(reconstruction2, ones(1,21));
subplot(2,4,7); imshow(dilation);
title('Dilation of F'); xlabel('G');
result = imreconstruct(dilation, reconstruction2);
subplot(2,4,8);imshow(result);
title('Result (Marker F, Mask G)'); xlabel('H');

%% Lezione 9 lab

% Esercizio 1 - Contorno
close all %chiusura finestre aperte da prima
clear %pulizia variabili presenti da prima
clc %pulizia command window

BW = imread("bone.tif");
figure, imshow(BW)
% bisogna trovare il punto di partenza con una scansione raster (appena trovo un punto lo prendo):
[M,N] = size(BW);
%for i = 1:M
%    for j = 1:N
%        if(BW(i,j) == 1)
%            a = i;
%            b = j;
%        end
%    end
%end
[a, b] = find(BW, 1, 'last'); % (metodo rapido)

% B è il contorno (boundary), è un vettore contenente le righe e le colonne dei punti del contorno
B = bwtraceboundary(BW, [a, b], "N", 8);
B = B(1:end-1, :); % (l'ultimo punto è uguale a quello di partenza e va ignorato)
L = size(B);

%bisogna plottare sul grafico i punti indicati nelle righe del vettore (ogni riga indica x e y di un punto del bordo)
figure %apriamo una nuova finestra in cui disegnare il grafico
axis equal %(evitiamo la distorsione del grafico)
for i=1:L(1) %(per ogni punto del contorno)
    hold on %(per disegnare senza cancellare il punto di prima)
    plot(B(i,2), -B(i,1), '.r', 'LineWidth', 1); %(B(i,2) prende la colonna (la x) del punto considerato, B(i,1) la riga (la y))
end

% per mostrare il contorno come immagine si crea una immagine di zeri e si mette ad 1 i pixel indicati in ogni riga di B:
c = zeros(M,N);
for i = 1:L(1)
    c(B(i,1), B(i,2)) = 1;
end
figure, imshow(c);

% Esercizio 2 - Descrittori

close all % chiusura finestre aperte da prima
clear % pulizia variabili presenti da prima
clc % pulizia command window

% lo shift viene fatto per lavorare più facilmente sul vettore
i = imread("shapes1.tif");
figure, imshow(i)

[L,n] = bwlabel(~i); % (la bwlabel guarda gli oggetti bianchi!)
disp(['Numero oggetti: ', num2str(n)])

% trovo i centroidi e li metto al centroide di ogni oggetto
hold on
prop = regionprops(L, "Centroid");
all_centr = cat(1, prop.Centroid); %(prop è un vettore di vettori, va scompattato per plottare tutti assieme)
title("Centroidi")
plot(all_centr(:,1), all_centr(:,2), 'r*')

% ora dove c'è il centroide di ogni figura mettiamo i valori dei descrittori area,
% area convessa, perimetro, circolarità, solidità e bounding box

% area
hold on
prop = regionprops(L, "Area");
all_area = cat(1, prop.Area);
figure,imshow(L), title("Area")
text(all_centr(:,1), all_centr(:,2), num2str(all_area(:)), "HorizontalAlignment", "center")

% area convessa
hold on
prop = regionprops(L, "ConvexArea");
all_convex_area = cat(1, prop.ConvexArea);
figure, imshow(L), title("Area Convessa")
text(all_centr(:,1), all_centr(:,2), num2str(all_convex_area(:)), "HorizontalAlignment", "center")

% perimetro
hold on
prop = regionprops(L, "Perimeter");
all_perimeter = cat(1, prop.Perimeter);
figure, imshow(L), title("Perimetro")
text(all_centr(:,1), all_centr(:,2), num2str(all_perimeter(:)), "HorizontalAlignment", "center")

% circolarità
hold on
prop = regionprops(L, "Circularity");
all_circularity = cat(1, prop.Circularity);
figure, imshow(L), title("Circolarità")
text(all_centr(:,1), all_centr(:,2), num2str(all_circularity(:)), "HorizontalAlignment", "center")

% solidità
hold on
prop = regionprops(L, "Solidity");
all_solidity = cat(1, prop.Solidity);
figure, imshow(L), title("Solidità")
text(all_centr(:,1), all_centr(:,2), num2str(all_solidity(:)), "HorizontalAlignment", "center")

% bounding box (ogni riga ha x, y (di partenza), ampiezza, altezza)
hold on
prop = regionprops(L, "BoundingBox");
all_bbox = cat(1, prop.BoundingBox);
figure, imshow(L), title("Bounding Box")
for k = 1 : n
    this_bb = all_bbox(k, :);
    rectangle("Position", [this_bb(1), this_bb(2), this_bb(3), this_bb(4)], "EdgeColor", "r", "LineWidth", 2);
end

% Esercizio 3 - Signature

close all % chiusura finestre aperte da prima
clear % pulizia variabili presenti da prima
clc % pulizia command window

% la signature è una funzione che mappa la distanza di ogni punto del contorno dal centroide
BW = imread("boundary_square.tif");
figure, imshow(BW);
impixelinfo;

prop = regionprops(BW, "Centroid");
centroid = cat(1, prop.Centroid);

%troviamo il primo punto da cui partire:
[a, b] = find(BW, 1, 'last');

% B è il contorno (boundary), è un vettore contenente le x e le y dei punti del contorno
B = bwtraceboundary(BW, [a, b], "N", 8);
B = B(1:end-1, :);
L = size(B);

% il contorno viene mostrato come funzione
figure, axis equal
for i=1:L(1)
    hold on
    plot(B(i,2), B(i,1), '.r', 'LineWidth', 1); %(appare a testa in giù)
end

%plottiamo centroide e punto di partenza del calcolo (ricorda che a sono righe e b colonne ma servono come x e y)
hold on, plot(centroid(1,1), centroid(1,2), 'r*');
hold on, plot(b, a, 'b*');
%vettore in cui mettere i valori della funzione signature:
F = zeros(L(1),1);
for i = 1:L
    F(i) = sqrt((centroid(1,1) - B(i,1)).^2 + (centroid(1,2) - B(i,2)).^2); % (distanza tra centroide e punto)
end
figure, plot(F(:));

% Esercizio 4 - Object circularity
close all % chiusura finestre aperte da prima
clear % pulizia variabili presenti da prima
clc % pulizia command window
% vogliamo trovare gli oggetti di forma circolare dell'immagine
i = imread("pillsetc.png");
figure, imshow(i)

% per prima cosa rendiamo l'immagine binaria con otsu
i = rgb2gray(i);
T = graythresh(i);
T = T*255; % (dato che è double)
%T = uint8(T); %(stessa cosa)
g = i>=T;
figure, imshow(g);

% togliamo gli artefatti e riempiamo gli anelli
se = strel('disk',1);
g = imclose(imopen(g, se), se);
g = imfill(g, "holes");
figure, imshow(g);

% segniamo i singoli oggetti (bwboundaries restituisce contorni e label):
[B, L] = bwboundaries(g, "noholes");
figure, imshow(label2rgb(L, "jet", [.5 .5 .5])) % (le quadre rendono lo sfondo grigio)

% per trovare gli oggetti circolari calcoliamo manualmente la circolarità (4*pi*area/perimetro^2)
stats = regionprops(L, "Area", "Perimeter", "Centroid");
treshold = 0.94; % soglia sopra la quale si è circolari

for i = 1:length(B)
    area = stats(i).Area;
    perimeter = stats(i).Perimeter;
    metric = 4*pi*area/perimeter.^2;
    %disegniamo gli asterischi sugli oggetti circolari:
    if metric > treshold
        hold on
        plot(stats(i).Centroid(:,1), stats(i).Centroid(:,2), "r*")
    end
end

%% Lezione 10 lab

% Esercizio 1 - Vertical and horizontal edges
close all %chiusura finestre aperte da prima
clear %pulizia variabili presenti da prima
clc %pulizia command window

f = imread("libri_noise.bmp");
subplot(3,1,1), imshow(f);

% pulizia (per quanto sia possibile):
f = imgaussfilt(f);
subplot(3,1,2), imshow(f);

% filtri per rilevare le linee orizzontali e verticali:
hr = [1 1 1; 0 0 0; -1 -1 -1];
vr = [1 0 -1; 1 0 -1; 1 0 -1];

h = imfilter(f, hr);
v = imfilter(f, vr);
g = h + v;
subplot(3,1,3), imshow(g);

% Esercizio 2 - Shape Signature
close all %chiusura finestre aperte da prima
clear %pulizia variabili presenti da prima
clc %pulizia command window

% letteralmente lo stesso esercizio della lezione 9 ma con un triangolo:
BW = imread("boundary_triangle.tif");
figure, imshow(BW)
prop = regionprops(BW, "Centroid");
centroid = cat(1, prop.Centroid);
% troviamo il primo punto da cui partire:
[a, b] = find(BW, 1, 'last');

% B è il contorno (boundary), è un vettore contenente le x e le y dei punti del contorno
B = bwtraceboundary(BW, [a, b], "N", 8);
B = B(1:end-1, :);
L = size(B);
F = zeros(L(1),1); % (vettore in cui mettere i valori della funzione signature)
for i = 1:L
    F(i) = sqrt((centroid(1,1) - B(i,1)).^2 + (centroid(1,2) - B(i,2)).^2);
end
figure, plot(F(:))

% Esercizio 3 - Segmentation
close all %chiusura finestre aperte da prima
clear %pulizia variabili presenti da prima
clc %pulizia command window

% Watershed & watershed_marker0
f = imread("gel-image.tif");
subplot(2,3,1), imshow(f); title('Original');
impixelinfo;
im = imextendedmin(f,2);
Lim = watershed(bwdist(im));
em = Lim == 0;
subplot(2,3,2), imshow(em);title('Watershed');

g = watershed_marker(f);
subplot(2,3,3), imshow(g);
xlabel('Watershed with marker segmentation', "FontWeight","bold");

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

subplot(2,3,4), imshow(g);
xlabel('Regiongrow segmentation', "FontWeight","bold");

% Split e merge
f2 = double(f);
g = splitmerge(f2, 2, @predicate);
subplot(2,3,5), imshow(g);
xlabel('Split merge segmentation', "FontWeight","bold");

% Highpass emphasize filtering
close all % chiusura finestre aperte da prima
clear     % pulizia variabili presenti da prima
clc       % pulizia command window

f = imread('chestXray_original.tif');
subplot(2, 2, 1), imshow(f); title('Original');

[PQ]=[2*size(f,1)-1 2*size(f,2)-1];
D0=0.05*PQ(2);

H = highpass_filter('gaussian', PQ(1), PQ(2), D0);
g = freq_dom_filter(f, H);
subplot(2,2,2); imshow(g, []);
xlabel('Gaussian filtered image');

l = 0.5*double(f) + 0.75*g;
subplot(2,2,3); imshow(l, []);
xlabel('HPE filtered image');

w = histeq(uint8(l));
subplot(2,2,4); imshow(w, []);
xlabel('Final image');

%% Funzioni
%% freq_dom_filter
function g = freq_dom_filter(f, H)
% g = FREQ_DOM_FILTER(F, H) filters F
% in the frequency domain using the
% filter transfer function H. The
% output g is the filtered image, which
% has the same size as F.
% FREQ_DOM_FILTER automatically pads F
% to be the same size as H.
% obtain the FFT of the padded input
F=fft2(f, size(H,1), size(H,2));
% perfom filtering
G=H.*F;
g = real(ifft2(G));
%crop the original size
g=g(1:size(f,1), 1:size(f,2));

%% Lowpass filter
function H = lowpass_filter(type, M, N, D0, n)
% LOWPASS_FILTER computes lowpass frequency domain filters. H = LOWPASS_FILTER(type, M, N, D0, n)
% creates the transfer function of a lowpass filter, H, of the specified TYPE and size (MxN).
% To view the filter as an image, it should be centered using H = fftshift(H).
% Valid values for TYPE, D0 and n are:
% 'ideal' Ideal lowpass filter with cutoff frequency D0. n need not to be supplied. DO must be ositive.
% 'btw' Butterworth lowpass filter of order n, and cutoff D0. The default value for n is 1.0. DO must be positive.
% 'gaussian' Gaussian lowpass filter with cutoff (std.) D0. n need not to be supplied. DO must be positive.
% Use freq_meshgrid to set up the meshgrid arrays needed for the required distance computation.
[U, V] = freq_meshgrid(M,N);
%Compute the distances D(U,V).
D = sqrt(U.^2 + V.^2);
% Begin filter computations.
switch type
    case 'ideal'
    H = double(D<=D0);
    case 'btw'
    if nargin == 4 n = 1; end
    H = 1./(1 + (D./D0).^(2*n));
    case 'gaussian'
    H = exp(-(D.^2)/(2*(D0^2)));
    otherwise
    error('Unknown filter type');
end
%% Highpass filter
function H = highpass_filter(type, M, N, D0, n)
    % HIGHPASS_FILTER computes highpass frequency domain filters.
    % H = HIGHPASS_FILTER(type, M, N, D0, n) creates the transfer function of a
    % highpass filter, H, of the specified TYPE and size (MxN).
    % To view the filter as an image, it should be centered using H = fftshift(H).
    % Valid values for TYPE, D0 and n are:
    % 'ideal' Ideal highpass filter with cutoff frequency D0.
    % n need not to be supplied. DO must be positive.
    % 'btw' Butterworth highpass filter of order n, and cutoff
    % D0. The default value for n is 1.0. DO must be positive.
    % 'gaussian' Gaussian highpass filter with cutoff (std.dev.)
    % D0. n need not to be supplied. DO must be positive.
    % The transfer function Hhp of a highpass filter is 1 - Hlp,
    % where Hlp is the transfer function of the corresponding
    % lowpass filter.
    if nargin==4 
        n = 1; 
    end
    % Generate highpass filter.
    Hlp = lowpass_filter(type, M, N, D0, n);
    H = 1 - Hlp;
    %% Iterative thresholding
    function [g, T] = iter_thresh(f)
    T = 0.5 * (double(max(f(:))) + double(min(f(:))));
    flag = false;
    while ~flag
        g  = f >=T;
        Tnext = 0.5 * (mean(f(g)) + mean(f(~g)));
        flag = abs(T-Tnext) < 0.5;
        T = Tnext;
    end
    T = uint8(T);
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

%% Rotazioni
function [g] = rot_dist(f, alfa, c)

% matrice T1 rotazione
T1=[cos(alfa), sin(alfa), 0; -sin(alfa), cos(alfa), 0; 0, 0, 1];

% matrice T2 distorsione(verticale)
T2=[1, 0, 0; c, 1, 0; 0, 0, 1];

% matrice definitiva (si possono usare affine2d e imwarp instead)
T=T2*T1;
tform =maketform('affine',T);

%figure, imshow(imtransform(f, tform, 'FillValues', 127));

g = imtransform(f, tform, 'FillValues', 127);

%% Split & merge
function g = splitmerge(f, mindim, fun)
%SPLITMERGE Segment an image using a split-and-merge algorithm.
%   G = SPLITMERGE(F, MINDIM, @PREDICATE) segments image F by using a
%   split-and-merge approach based on quadtree decomposition. MINDIM
%   (a positive integer power of 2) specifies the minimum dimension
%   of the quadtree regions (subimages) allowed. If necessary, the
%   program pads the input image with zeros to the nearest square  
%   size that is an integer power of 2. This guarantees that the  
%   algorithm used in the quadtree decomposition will be able to 
%   split the image down to blocks of size 1-by-1. The result is  
%   cropped back to the original size of the input image. In the  
%   output, G, each connected region is labeled with a different
%   integer.
%
%   Note that in the function call we use @PREDICATE for the value of 
%   fun.  PREDICATE is a function in the MATLAB path, provided by the
%   user. Its syntax is
%
%       FLAG = PREDICATE(REGION) which must return TRUE if the pixels
%       in REGION satisfy the predicate defined by the code in the
%       function; otherwise, the value of FLAG must be FALSE.
% 
%   The following simple example of function PREDICATE is used in 
%   Example 10.9 of the book.  It sets FLAG to TRUE if the 
%   intensities of the pixels in REGION have a standard deviation  
%   that exceeds 10, and their mean intensity is between 0 and 125. 
%   Otherwise FLAG is set to false. 
%
%       function flag = predicate(region)
%       sd = std2(region);
%       m = mean2(region);
%       flag = (sd > 10) & (m > 0) & (m < 125);

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.6 $  $Date: 2003/10/26 22:36:01 $

% Pad image with zeros to guarantee that function qtdecomp will
% split regions down to size 1-by-1.
Q = 2^nextpow2(max(size(f)));
[M, N] = size(f);
f = padarray(f, [Q - M, Q - N], 'post');

%Perform splitting first. 
S = qtdecomp(f, @split_test, mindim, fun);

% Now merge by looking at each quadregion and setting all its 
% elements to 1 if the block satisfies the predicate.

% Get the size of the largest block. Use full because S is sparse.
Lmax = full(max(S(:)));
% Set the output image initially to all zeros.  The MARKER array is
% used later to establish connectivity.
g = zeros(size(f));
MARKER = zeros(size(f));
% Begin the merging stage.
for K = 1:Lmax 
   [vals, r, c] = qtgetblk(f, S, K);
   if ~isempty(vals)
      % Check the predicate for each of the regions
      % of size K-by-K with coordinates given by vectors
      % r and c.
      for I = 1:length(r)
         xlow = r(I); ylow = c(I);
         xhigh = xlow + K - 1; yhigh = ylow + K - 1;
         region = f(xlow:xhigh, ylow:yhigh);
         flag = feval(fun, region);
         if flag 
            g(xlow:xhigh, ylow:yhigh) = 1;
            MARKER(xlow, ylow) = 1;
         end
      end
   end
end

% Finally, obtain each connected region and label it with a
% different integer value using function bwlabel.
g = bwlabel(imreconstruct(MARKER, g));

% Crop and exit
g = g(1:M, 1:N);

%% Split-test
function v = split_test(B, mindim, fun)
% THIS FUNCTION IS PART OF FUNCTION SPLIT-MERGE. IT DETERMINES 
% WHETHER QUADREGIONS ARE SPLIT. The function returns in v 
% logical 1s (TRUE) for the blocks that should be split and 
% logical 0s (FALSE) for those that should not.

% Quadregion B, passed by qtdecomp, is the current decomposition of
% the image into k blocks of size m-by-m.

% k is the number of regions in B at this point in the procedure.
k = size(B, 3);

% Perform the split test on each block. If the predicate function
% (fun) returns TRUE, the region is split, so we set the appropriate
% element of v to TRUE. Else, the appropriate element of v is set to
% FALSE.
v(1:k) = false;
for I = 1:k
   quadregion = B(:, :, I);
   if size(quadregion, 1) <= mindim
      v(I) = false;
      continue
   end
   flag = feval(fun, quadregion);
   if flag
      v(I) = true;
   end
end

%% Watershed with markers
function f2 = watershed_marker(f)

    h = fspecial('sobel');
    fd = double(f);

    % applico filtro
    g = sqrt(imfilter(fd, h, 'replicate').^2 + imfilter(fd, h, 'replicate').^2);

    % minimi regionali sul gradiente
    rm = imregionalmin(g); %individuazione markers interni

    % riduciamo minimi regionali
    im = imextendedmin(f,2); % differenza con pixel dell'intorno ai minimi almeno 2
    fim = f;
    fim(im) = 175; % li visualizzo con toni di grigio 175 solo a scopo visivo

    % individuo i markers esterni
    Lim = watershed(bwdist(im)); % individuo lo sfondo in mezzo tra due markers interni
    
    % segmento distanza tra due marker interni
    em = Lim == 0;
    
    % modifica immagine di partenza in cui minimi regionali appartengono a 
    % marker interni o a marker esterni
    g2 = imimposemin(g, im | em); % immagine su cui si elabora trasformata watershed
    L2 = watershed(g2);
    f2=f;
    f2(L2==0)=255; 

%% Predicato
%Gestione del flag
function flag = predicate(region)
sd= std2(region);
m = mean2(region);
flag = (sd> 7) & (m > 10) & (m < 175);

%% Fourier
%Implemented in MATLAB
clc
close all
clear all

M = 1024;
K = 8;
f = zeros(M,1);
f(1:K) = ones(K,1);
figure; plot(1:20, f(1:20), '.');
ylim([0 4]);
% R = zeros(M,1);
% I = zeros(M,1);
% for u = 0:M-1
%     for x = 0:M-1
%         R(u+1) = R(u+1) + ( (-1)^x * f(x+1) * cos(2*pi*u*x/M) );
%         I(u+1) = I(u+1) + ( (-1)^x * f(x+1) * sin(2*pi*u*x/M) );
%     end
%     F(u+1) = 1/M*sqrt(R(u+1)*R(u+1) + I(u+1)*I(u+1));
% end
F = zeros(M, 1);
for u = 0:M-1
    for x = 0:M-1
        F(u+1) = F(u+1) + ((-1)^x*f(x+1)*exp(-1i*2*pi*u*x/M));
    end
    F(u+1) = 1/ M*F(u+1);
end
figure; plot(1:M, abs(F(1:M)));
invF = zeros(M, 1);
for x = 0:M-1
    for u = 0:M-1
        invF(x+1) = invF(x+1) + F(u+1)*exp(1i*2*pi*u*x/M);
    end
    %invF(x+1) = (1/M)*invF(x+1);
end
figure; plot(1:20, abs(invF(1:20)), '.'); ylim([0 4]);


% trasformata di fourier
clc
close all
clear all

% 1 dimension
y = zeros(256,1);
y(1:20) = 2.0;
subplot(2,2,1); plot(y); ylim([0 4]);

yy = fft(y);
subplot(2,2,2); plot(abs(yy));
yyy = fftshift(yy);
subplot(2,2,3); plot(abs(yyy));

% 2 Dimension

clc
clear all
close all

f = imread('image_synt.tif');
subplot(2,2,1); imshow(f);

ft = fft2(f);
S = abs(ft);
subplot(2,2,2); imshow(S, []);

fs = fftshift(ft);
subplot(2,2,3); imshow(abs(fs), []);

flog = log(1 +abs(fs));
subplot(2,2,4); imshow(flog, []);

% Soble

clc
clear all
close all

f = imread('bld.tif');
subplot(2,2,1); imshow(f); xlabel('A gray scale image');

F = fft2(f, 2*size(f,1)-1, 2*size(f,2)-1);

h=fspecial('sobel')';
H=freqz2(h,2*size(f,1)-1,2*size(f,2)-1);

H=fftshift(H);
G=H.*F;
g=real(ifft2(G));
g=g(1:size(f,1), 1:size(f,2));
subplot(2,2,2); imshow(g, []);
xlabel('Result of filtering');

subplot(2,2,3); imshow(abs(g), []);
xlabel('The absolute value of g');
subplot(2,2,4);imshow(abs(g) > 0.2*abs(max(g(:))));
xlabel('A thresholded binary image');

% dominio spaziale
figure
f = imread('bld.tif');
subplot(2,2,1), imshow(f); xlabel('A gray scale image');
gs = imfilter(double(f), double(h));
subplot(2,2,2); imshow(gs, []); xlabel('Result of filtering');
subplot(2,2,3); imshow(abs(gs), []);xlabel('The absolute value of gs');
subplot(2,2,4); imshow(abs(gs) > 0.2*abs(max(gs(:))));xlabel('A thresholded binary image');
%% Visualizzazione filtri highpass
H = highpass_filter('ideal', PQ(1), PQ(2), D0);
subplot(2,2,1); imshow(fftshift(H), []);
xlabel('Ideal highpass filter');
H = highpass_filter('btw', PQ(1), PQ(2), D0);
subplot(2,2,2); imshow(fftshift(H), []);
xlabel('Butterworth highpass filter');
H = highpass_filter('gaussian', PQ(1), PQ(2), D0);
subplot(2,2,[3 4]); imshow(fftshift(H), []);
xlabel('Gaussian highpass filter');
%% Visualizzazione filtri Lowpass
H = lowpass_filter('ideal', PQ(1), PQ(2), D0);
subplot(2,2,1); imshow(fftshift(H), []);
xlabel('Ideal lowpass filter');
H = lowpass_filter('btw', PQ(1), PQ(2), D0);
subplot(2,2,2); imshow(fftshift(H), []);
xlabel('Butterworth lowpass filter');
H = lowpass_filter('gaussian', PQ(1), PQ(2), D0);
subplot(2,2,[3 4]); imshow(fftshift(H), []);
xlabel('Gaussian lowpass filter');
%% Exam Tests 1
% data l'immagine rice.png individuare tutti i chicchi di riso, escludendo
% gli oggetti lungo il bordo, individuare quello più piccolo e quello più
% grande ed infine generare l'istogramma delle aree, usare la funzione hist.

clear
clc
close all;

I=imread('rice.png');
subplot(3,3,1), imshow(I);
title('Original image');

t = graythresh(I);
t=t*255;
g = I>=t;
subplot(3,3,2), imshow(g);
title('Thresholded image');

se = strel('disk', 10);
fo = imopen(I, se);
subplot(3,3,3), imshow(fo);
title('Opened image');

f2 = imsubtract(I, fo);
f2 = imtophat(I, se);
subplot(3,3,4), imshow(f2);
title('Top-hat transformation');

t = graythresh(f2);
t=t*255;
g2 = f2>=t;
subplot(3,3,5), imshow(g2);
title('Thresholded top-hat image');

%elimino gli i chicchi di riso dal bordo

B=imclearborder(g2);
figure, imshow(B);

se=strel('disk',2);
B=imopen(B,se);
se=strel('disk',1);
B=imerode(B,se);
figure, imshow(B);


% calcolo le aree dei chicchi di riso
[L, n]= bwlabel(B);
prop = regionprops(L, 'Area');
all_Area = cat(1, prop.Area);
max=max(all_Area(:));
min=min(all_Area(:));

for i=1:n
    if all_Area(i)==max;
        figure, imshow(L==i);
        break;
    end
end

for i=1:n
    if all_Area(i)==min;
        figure, imshow(L==i);
        break;
    end
end

plot(all_Area(:));
%% Exam test 2
% data l'immagine rice.png individuare tutti i chicchi di riso, escludendo
% gli oggetti lungo il bordo, individuare quello più piccolo e quello più
% grande ed infine generare l'istogramma delle aree, usare la funzione hist.

clear
clc
close all;

I=imread('rice.png');
subplot(1,2,1), imshow(I);
title('Original image');

se = strel('disk', 10);
fo = imopen(I, se);

f2 = imsubtract(I, fo);
f2 = imtophat(I, se);

t = graythresh(f2);
t=t*255;
g2 = f2>=t;
subplot(1,2,2), imshow(g2);
title('Thresholded top-hat image');

% elimino gli i chicchi di riso dal bordo

B=imclearborder(g2);

% elimino le piccole aree rimaste dopo la sogliatura, e separo i chicchi
% di riso uniti

se=strel('disk',2);
B=imopen(B,se);
se=strel('disk',1);
B=imerode(B,se);
figure, imshow(B);


% calcolo le aree dei chicchi di riso
[L, n]= bwlabel(B);
prop = regionprops(L, 'Area');
all_Area = cat(1, prop.Area);
max=max(all_Area);
min=min(all_Area);

for i=1:n
    if all_Area(i)==max;
        figure, imshow(L==i);
        break;
    end
end

for i=1:n
    if all_Area(i)==min;
        figure, imshow(L==i);
        break;
    end
end

figure, hist(all_Area);

%% Exam test 3
% Data l'immagine coins.png' individuare tutti i contorni degli 
% oggetti presenti e visualizzarli. 
% Determinare le monete piccole e le monete grandi marcandole 
% con un simbolo differente 
% (*rosso piccole, *blu grandi)

I = imread('coins.png');
subplot(2,2,1);
imshow(I), title('Immagine originale');

%Sogliatura con Otsu è ideale quando lo sfondo è omogeneo
t = graythresh(I);
BW = I > (t * 255);
subplot(2,2,2);
imshow(BW), title('Immagine sogliata');

%Sono presenti dei buchi che ho riempito con il comando imfill
BW = imfill(BW, 'holes');
subplot(2,2,3)
imshow(BW), title('Immagine con buchi riempiti');

%Per poter ricavare i contorni ho usato il comando bwboundaries
boundaries = bwboundaries(BW);
ncontorni = size (boundaries);
subplot(2,2,4);
imshow(I), title('Immagine con contorni evidenziati');
hold on;
for i=1:ncontorni(1)
    b = boundaries{i};
    plot(b(:,2), b(:,1), 'r');
end

hold off;

%Ricavo la matrice con le etichette delle componenti connesse
[L, num] = bwlabel(BW);

%Ricavo le aree e i centroidi
areas = regionprops(L, 'area');
centroids = regionprops(L, 'centroid');

figure, imshow(I), title('Monete piccole in rosso, monete grandi in blu');
hold on;

%Suddivisione di monete grandi e piccoli con una soglia a scelta
for i=1:num
    if areas(i).Area < 2200
        plot(centroids(i).Centroid(1), centroids(i).Centroid(2), '*r');
    else
        plot(centroids(i).Centroid(1), centroids(i).Centroid(2), '*b');
    end
end

hold off;

%% Exam test 4
% Data l'immagine 'pillsetc.png, identificare gli oggetti di forma
% arrotondata. Classificare gli oggetti presenti in base alla loro rotondità
% visualizzarne il contorno ed il valore di rotondità

f = imread('pillsetc.png');
imshow(f);

%Trasformazione immagine RGB in grayscale
binaryImage = rgb2gray(f);
subplot(2,3,1), imshow(binaryImage);
title('Immagine Originale');
binaryImage = imbinarize(binaryImage);
subplot(2,3,2), imshow(binaryImage);
title('Immagine in b/n');
%Riempimento buchi nell'immagine
binaryImage = imfill(binaryImage, 'holes');
%Riempimento pixel bianchi non appartenenti all'oggetto
se = strel('disk', 2);
binaryImage = imopen(binaryImage, se);
binaryImage = imclose(binaryImage, se);
subplot(2,3,3), imshow(binaryImage);
title('Immagine in b/n con buchi riempiti');

%Evidenziati oggetti rotondi
subplot(2,3,4), imshow(f);
[L, n] = bwlabel(binaryImage);
title('Oggetti rotondi evidenziati');
hold on;
prop = regionprops(L, 'Circularity');
all_Circularity = cat(1, prop.Circularity);
prop = regionprops(L, 'Centroid');
all_Centr = cat(1, prop.Centroid);
for i=1:size(all_Circularity)
    if all_Circularity(i) >= 1
        plot(all_Centr(i,1), all_Centr(i,2), 'Marker', '*', 'MarkerEdgeColor', 'r');
    end
end

%Calcolo contorno degli oggetti
subplot(2,3,5), imshow(f);
title('Contorni oggetti');
hold on
boundaries = bwboundaries(binaryImage);
numberOfBoundaries = size(boundaries, 1);
for k = 1 : numberOfBoundaries
    thisBoundary = boundaries{k};
    plot(thisBoundary(:,2), thisBoundary(:,1), 'r', 'LineWidth', 1);
end
hold off;

%Calcolo rotondità oggetti
subplot(2,3,6), imshow(f);
title('Indice di rotondità');
hold on;
text(all_Centr(:,1), all_Centr(:,2), num2str(all_Circularity(:)), 'Color', 'red', ...
     'HorizontalAlignment', 'center', 'BackgroundColor', 'black', 'FontSize', 7);
%% Exam test 5
% Esame 31/01/2023
% Data l'immagine 'rice.png', non uniformemente illuminata, individuare tutti i singoli
% chicchi di riso in essa presenti
% Escludendo gli oggetti lungo il bordo, determinare l'area del chicco di
% riso.
% Raggruppare i chicchi di riso in 3 macroaree (piccolo, medio, grande) in
% base all'area,  e marcare in modo differente i chicchi a seconda della
% categoria di appartenenza. Medio è +/-10% del valore medio dell'area
% Generare infine l'istogramma delle 3 categorie tramite la funzione
% histogram (o bar)
% chiusura di tutte le finestre aperte e pulizia workspace e command window 
close all
clear 
clc

%leggo l'immagine 
I = imread("rice.png");
figure, imshow(I), title('Immagine Originale');

%tophat (opening e poi sottrae il risultato all'immagine originale)
SE = strel("disk",10);
th = imtophat(I,SE);
figure, imshow(th), title('Tophat');

%Separo i chicchi l'uno dall'altro
t = graythresh(th); %Sogliatura con Otsu
bw = th > (t * 255);  %Binarizzo
SE = strel("disk",1);
bw = imerode(bw,SE);
SE = strel("disk",1);
bw = imopen(bw,SE);

%Pulisco i bordi 
bw = imclearborder(bw);
figure, imshow(bw), title('Chicchi Separati');

%Estraggo tutte le componenti connesse (chicchi)
[L,n] = bwlabel(bw);

%Estrazione centroidi e area dei chicchi
props = regionprops(L, "Centroid", "Area");
centroids = cat(1, props.Centroid);
areas = cat(1, props.Area);

%Calcolo l'area media (total area/ numero di chicchi)
medium_area = sum(areas(:)) / n;

%Inizializzo i counter
chicchi_grandi =0;
chicchi_piccoli = 0;
chicchi_medi =0;

%apro una nuova finestra (uso hold on per non cancellare i punti precedenti)
figure, imshow(I), title('Chicchi grandi in blu, medi rossi, piccoli verdi');
hold on;

%Suddivisione di chicchi medi/piccoli/grandi
for i=1:n
    if areas(i) > (medium_area + (0.1 * medium_area))
        plot(centroids(i,1), centroids(i,2), '*b');
        chicchi_grandi = chicchi_grandi +1;
    elseif areas(i) < (medium_area - (0.1 * medium_area))
        plot(centroids(i,1), centroids(i,2), '*g');
        chicchi_piccoli = chicchi_piccoli +1;
    else
        plot(centroids(i,1), centroids(i,2), '*r');
        chicchi_medi = chicchi_medi +1;
    end
end
hold off;

%creazione istogramma
y = [chicchi_piccoli,chicchi_medi,chicchi_grandi];
x = categorical({'Piccoli', 'Medi', 'Grandi'});
x = reordercats(x,{'Piccoli', 'Medi', 'Grandi'});
figure, bar(x,y), title('Suddivisione Chicchi');
%% Exam test 6
% Data l'immagine 'rice.png', non uniformemente illuminata, individuare tutti
% i singoli chicchi di riso in essa presenti.
% 
% Escludendo gli oggetti lungo il bordo, determinare l'orientazione di
% ciascun chicco di riso.
% 
% Raggruppando le orientazioni nelle quattro direzioni principali
% (orizzontale, verticale, diagonale e antidiagonale), marcare in modo
% differente i chicchi in base alle differenti orientazioni.
% 
% ESEMPIO. Marcare
%     i chicchi orizzontali con un asterisco ROSSO,
%     i chicchi verticali con un asterisco BLU,
%     i chicchi orientati a 45° con un asterisco VERDE e infine
%     i chicchi orientati a -45° con un asterisco GIALLO.
% 
% Il simbolo marcatore verrà posto nel centroide di ogni oggetto.
% 
% Generare, infine, l'istogramma delle 4 orientazioni principali tramite
% la funzione histogram (o bar).
%
% Usare una soglia pari a +/- 23 per separare le varie regioni
% 
% NOTA: ci devono essere 73 chicchi se la segmentazione è stata effettuata correttamente,
% una volta rimossi i chicchi sul bordo

close all
clear 
clc

%leggo l'immagine 
I = imread("rice.png");
subplot(1,2,1), imshow(I), title('Immagine Originale');
% Tophat transformation per uniformare l'illuminazione
se = strel('disk', 10);
% op = imopen(I, se);
% Is = imsubtract(I, op);
Is = imtophat(I, se);
% figure, imshow(Is, []), title('Tophat');
% Applico il metodo di OTSU per la sogliatura
th = graythresh(Is);
th = th*255;
I_tr = (Is>=th);
subplot(1,2,2); imshow(I_tr);title('Thresholded top-hat image');
% Eliminazione oggetti lungo il bordo
I_wb = imclearborder(I_tr);
% figure, imshow(I_wb, []);
se = strel('disk', 2);
I_op = imopen(I_wb, se);
se = strel('disk', 1);
I_er = imerode(I_op, se);
figure, imshow(I_er, []);
% Estrazione componenti connesse (chicchi)

[L,n] = bwlabel(I_er);
props = regionprops(L, "Centroid", 'Orientation');
allAngles = [props.Orientation]; % Extract all orientation angles into one vector.
allCentr = cat(1, props.Centroid);
% contatori
hor = 0; ver = 0; diag = 0; antid = 0;
figure, imshow(I);
hold on
% Colorazione centroidi in base alla direzione
for i = 1 : n
    if (allAngles(i) > 0-23 && allAngles(i) < 0+23)
        plot(allCentr(i, 1), allCentr(i, 2),  'Marker', '*', 'MarkerEdgeColor', 'r');
        hor = hor +1;
    end
    if allAngles(i) >= 45 - 23 && allAngles(i) < 45 + 23
            plot(allCentr(i, 1), allCentr(i, 2),  'Marker', '*', 'MarkerEdgeColor', 'green');
            diag = diag +1;
    end
    if (allAngles(i) > 90-23 && allAngles(i) <= 90 + 23) || allAngles(i) > -90-23 && allAngles(i) <= -90 + 23
            plot(allCentr(i, 1), allCentr(i, 2),  'Marker', '*', 'MarkerEdgeColor', 'b');
            ver = ver +1;
    end
    if allAngles(i) >= -45 - 23 && allAngles(i) < -45 + 23
            plot(allCentr(i, 1), allCentr(i, 2),  'Marker', '*', 'MarkerEdgeColor', 'y');
            antid = antid +1;
    end
end
hold off
% Istogramma
y = [hor ver dig adg];
x = categorical({'Orizzontali', 'Verticali', 'Diagonali', 'Antidiagonali'});
x = reordercats(x, {'Orizzontali', 'Verticali', 'Diagonali', 'Antidiagonali'});
figure; %bar(x,y);
b = bar(x,y);
title('Istogramma direzioni');
b.FaceColor = 'flat';
b.CData(1,:) = [255 0 0];
b.CData(2,:) = [0 0 255];
b.CData(3,:) = [0 255 0];
b.CData(4,:) = [5 10 0];
%% Exam test 7 (6)

close all %chiusura finestre aperte da prima
clear %pulizia variabili presenti da prima
clc %pulizia command window


%lettura immagine
I = imread("rice.png");
imshow(I)

%evidenziamento chicchi di riso uniformando lo sfondo
se = strel("disk",10);
th = imtophat(I,se);
figure, imshow(th);

%separazione chicchi connessi e pulizia chicchi sul bordo
bw = th>40;
se2 = strel("disk",2);
bw = imerode(bw,se2);
se3 = strel("disk",1);
bw = imopen(bw,se3);
bw = imclearborder(bw);
figure, imshow(bw)

%estrazione orientamento e centroidi
[L,n] = bwlabel(bw);
props = regionprops(L, "Centroid", "Orientation");
all_centr = cat(1, props.Centroid);
all_or = cat(1, props.Orientation);

%plotting degli asterischi sopra i chicchi (soglia accettabile per gli angoli 23 gradi)
n_ver=0; n_or=0; n_45=0; n_m45=0;
figure, imshow(I);
for i=1:n
    hold on
    if(all_or(i) > 0-23 && all_or(i) < 0+23)
        plot(all_centr(i,1), all_centr(i,2), 'r*')
        n_or = n_or + 1;
    end
    if((all_or(i) > 90-23 && all_or(i) < 90+23) || (all_or(i) > -90-23 && all_or(i) < -90+23))
        plot(all_centr(i,1), all_centr(i,2), 'b*')
        n_ver = n_ver + 1;
    end
    if(all_or(i) > 45-23 && all_or(i) < 45+23)
        plot(all_centr(i,1), all_centr(i,2), 'g*')
        n_45 = n_45 + 1;
    end
    if(all_or(i) > -45-23 && all_or(i) < -45+23)
        plot(all_centr(i,1), all_centr(i,2), 'y*')
        n_m45 = n_m45 + 1;
    end
end

%creazione dell'istogramma con il numero di chicchi per orientamento
y = [n_ver,n_or,n_45,n_m45];
x = categorical({'Vertical', 'Horizontal', 'Diagonal', 'Antidiagonal'});
x = reordercats(x,{'Vertical', 'Horizontal', 'Diagonal', 'Antidiagonal'});
figure, bar(x,y);
%% test Exam 8
% Data l'immagine 'coins.png' individuare gli oggetti presenti
% determinare per ogni moneta il livello di grigio medio e la sua deviazione standard
% mostrare tali valori nella posizione del centroide di ogni oggetto in due figure distinte
% mediante la funzione bar fare un plot delle deviazioni standard in funzione della label.
% Infine individuare le monete più uniformi, ovvero quelle la cui deviazione standard è al disotto di una certa soglia (es. <20). 
% Mostrare le monete trovate visualizzandole il relativo bounding box
% Attenzione Alcune monete presentano buchi derivanti dall'operazione di soglia

clc
clear
close all

I = imread('coins.png');
subplot(3,2,1);
imshow(I), title('Original');

% Sogliatura con Otsu è ideale quando lo sfondo è omogeneo
t = graythresh(I);
BW = I > (t * 255);
subplot(3,2,2);
imshow(BW), title('Thresholded image');

% Sono presenti dei buchi che ho riempito con il comando imfill
BW = imfill(BW, 'holes');
subplot(3,2,3)
imshow(BW), title('Image without holes');

% Per poter ricavare i contorni ho usato il comando bwboundaries
boundaries = bwboundaries(BW);
ncontorni = size (boundaries);

%Ricavo la matrice con le etichette delle componenti connesse
[L, num] = bwlabel(BW);

% Ricavo i centroidi, bounding box, mean intensity (livello di grigio),
% pixel values(deviazione standard)
centroids = regionprops(L, 'centroid');
totCentroids = cat(1,centroids.Centroid);

boundingBoxes = regionprops(L,'BoundingBox');
totBB = cat(1, boundingBoxes.BoundingBox);

meanGL = regionprops(L,I,'MeanIntensity');
totMeanInt = cat(1, meanGL.MeanIntensity);

pValues = regionprops(L, I, "PixelValues");
totPValues = cat(1, pValues.PixelValues);

% Mostro i valori dei livelli di grigio medi
subplot(3,2,4)
imshow(BW), title('Gray levels');
hold on
for i=1:length(centroids)
        text(totCentroids(i,1), totCentroids(i,2), num2str(round(totMeanInt(i))), 'HorizontalAlignment', 'center', 'Color', 'r');
end
hold off

for i=1: length(pValues)
    s(i).StandDev = std(double(pValues(i).PixelValues));
end

% Standard deviation values
subplot(3,2,5)
imshow(BW), title('Standard deviations');
hold on
for i=1:length(centroids)
        text(totCentroids(i,1), totCentroids(i,2), num2str(round(s(i).StandDev)), 'HorizontalAlignment', 'center', 'Color', 'r');
end
hold off
% Uniforms coins Boundaries
subplot(3,2,6)
imshow(I), title('Uniforms coins Boundaries');
hold on
for i=1: length(pValues)
    if(s(i).StandDev < 20)
        % Boundaries
        thisBoundary = boundaries{i};
        plot(thisBoundary(:,2), thisBoundary(:,1), 'r', 'LineWidth', 1);
    end
end
hold off

% With Bounding Box
figure,
imshow(I), title('Uniforms coins Boundaries');
hold on
for i=1: num
    if(s(i).StandDev < 20)
        % Boundaries
        thisBB = totBB(i,:);
        rectangle('Position', [thisBB(1), thisBB(2), thisBB(3), thisBB(4)], 'EdgeColor','r', 'LineWidth', 2);
    end
end
hold off

% Histogram bars
figure, bar(1:length(pValues), [s.StandDev]);
%% Exam test 10
% Data l'immagine 'rice.png', non uniformemente illuminata, individuare tutti i singoli
% chicchi di riso in essa presenti
% Escludendo gli oggetti lungo il bordo, determinare l'area del chicco di
% riso.
% Raggruppare i chicchi di riso in 3 macroaree (piccolo, medio, grande) in
% base all'area,  e marcare in modo differente i chicchi a seconda della
% categoria di appartenenza. Medio è +/-10% del valore medio dell'area
% Generare infine l'istogramma delle 3 categorie tramite la funzione
% histogram (o bar)
% chiusura di tutte le finestre aperte e pulizia workspace e command window 
clc
clear
close all

I = imread("rice.png");
subplot(2,3,1); imshow(I); title('Original');
% Rendiamo uniforme l'immagine e binarizziamo, eliminiamo gli oggetti
% lungo il bordo ed eventuali legami tra chicchi differenti
se = strel('disk', 10);
I_tp = imtophat(I, se);
th = graythresh(I_tp)*255;
I_bw = I_tp >= th;
I_bw = imclearborder(I_bw);
se = strel('disk', 2);
I_bw = imopen(I_bw, se);
se = strel('disk', 1);
I_bw = imerode(I_bw, se);
subplot(2,3,2); imshow(I_bw); title('Binary image cleaned');
% Usiamo le funzioni bwlabel e regionprops per identificare le componenti
% connesse e identificare l'area di ogni singolo chicco, per poi
% suddividere le aree nelle tre categorie richieste
[L, n] = bwlabel(I_bw);
props = regionprops(L, I, 'Centroid', 'Area');
all_Centr = cat(1, props.Centroid);
all_Areas = cat(1, props.Area);
avg_area = mean(all_Areas);
subplot(2,3,[4 6]); imshow(I); title('Different color areas');
hold on
little = 0; middle = 0; big = 0;
for i = 1 : n
    area = double(all_Areas(i));
    if (all_Areas(i) <= avg_area - avg_area*0.1)
        plot(all_Centr(i,1), all_Centr(i,2), '*r');
        little = little + 1;
    end
    if (all_Areas(i) > avg_area - avg_area*0.1 && all_Areas(i) < avg_area + avg_area*0.1)
        plot(all_Centr(i,1), all_Centr(i,2), '*g');
        middle = middle + 1;
    end
    if (all_Areas(i) > avg_area + avg_area*0.1)
        plot(all_Centr(i,1), all_Centr(i,2), '*b');
        big = big + 1;
    end
end

% Genero l'istogramma
y = [little middle big];
x = categorical({'Little', 'Middle', 'Big'});
x = reordercats(x, {'Little', 'Middle', 'Big'});
figure; b = bar(x,y);
b.FaceColor = "flat";
b.CData(1,:)= [255 0 0];
b.CData(2,:)= [0 255 0];
b.CData(3,:)= [0 0 255];
%% Exam test 11
%Leggere blood2, convertirla in grayscale, segmentarla mediante soglia T=220. Determinare
%quante sono le cellule che non toccano il bordo dell'immagine escludendo
%eventuali regioni non significative. Mostrare sull'immagine originale i
%contorni delle cellule trovate. Distinguere i globuli rossi dai bianchi in
%base al livello di grigio medio. Nel centroide di ogni oggetto
%visualizzare il livello di grigio medio e in un'altra figura mostrare se
%WBC oRBC
clc
clear
close all

I = imread('blood2 (2).jpeg');
figure; imshow(I); title('Original');

I_bn = rgb2gray(I);
figure; imshow(I_bn); title('Gray Image');
th = 220;
I_bw = ~(I_bn > th);
I_bw = imfill(I_bw, "holes");

figure; imshow(I_bw); title('Gray Image');
I_wb = imclearborder(I_bw);
figure; imshow(I_wb); title('Gray Image');
se = strel('disk',2);
I_bw = imerode(I_wb, se);
figure; imshow(I_bw); title('Gray Image');
[L, n] = bwlabel(I_bw);

props = regionprops(L, I_bn, 'Centroid', 'MeanIntensity');
all_Centr = cat(1, props.Centroid);
all_Int = cat(1, props.MeanIntensity);
figure,imshow(I); title('Mean intensity and borders');
hold on
boundaries = bwboundaries(L);
wbc = 0; rbc=0;
for i = 1:n
    mean = round(double(all_Int(i))); 
    if(all_Int(i) < 150)
        thisB = boundaries{i};
        text(all_Centr(i, 1)-10, all_Centr(i,2), num2str(mean), "FontSize", 7, "FontWeight","bold");
        plot(thisB(:,2), thisB(:,1), 'b', 'LineWidth', 1);
        %facoltativo
        wbc = wbc + 1;
    else
        thisB = boundaries{i};
        text(all_Centr(i, 1)-10, all_Centr(i,2), num2str(mean), "FontSize", 7, "FontWeight","bold");
        plot(thisB(:,2), thisB(:,1), 'r', 'LineWidth', 1);
        %facoltativo
        rbc=rbc + 1;
    end
end
hold off
figure,imshow(I); title('Red or white');
hold on
a = 'WBC';
b = 'RBC';
for i = 1:n
    if(all_Int(i) < 150)
       text(all_Centr(i, 1)-10, all_Centr(i,2), a, "FontSize", 7, "FontWeight","bold");
    else
       text(all_Centr(i, 1)-10, all_Centr(i,2), b, "FontSize", 7, "FontWeight","bold");
    end
end
% facoltativo
figure; b = bar(all_Int);
b.FaceColor = "flat";
for k = 1:length(b.CData)
    b.CData(k,:) = [k/10 k/18 k/10];
end

%facoltativo
y = [wbc rbc];
x = categorical({'Bianchi', 'Rossi'});
x = reordercats(x, {'Bianchi', 'Rossi'});

figure; b = bar(x,y);
b.FaceColor = "flat";
b.CData(1, :) = [255 0 120];
b.CData(2, :) = [.5 .8 .97];
 

