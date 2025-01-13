close all; % Chiude tutte le finestre aperte precedentemente
clear;     % Rimuove tutte le variabili presenti nello workspace
clc;       % Pulisce la Command Window

% Applicare la ricostruzione morfologica per elaborare un'immagine con 
% sfondo complesso. L'obiettivo è isolare i tasti dell'immagine 'calculator.tif' 
% e visualizzarli su uno sfondo uniforme.

% Caricamento dell'immagine originale
img = imread('calculator.tif');
subplot(2, 4, 1); imshow(img); title('Immagine Originale');
xlabel('A');

% Erosione dell'immagine con un elemento strutturante orizzontale
img_eroded = imerode(img, ones(1, 71));
% Ricostruzione morfologica utilizzando l'immagine erosa come marker
img_reconstructed1 = imreconstruct(img_eroded, img);
subplot(2, 4, 2); imshow(img_reconstructed1); 
title('Ricostruzione Morfologica di A');
xlabel('B');

% Apertura morfologica dell'immagine originale
img_opened = imopen(img, ones(1, 71));
subplot(2, 4, 3); imshow(img_opened);
title('Apertura Morfologica di A'); xlabel('C');

% Top-hat basato sulla ricostruzione morfologica
tophat_recon = img - img_reconstructed1;
subplot(2, 4, 4); imshow(tophat_recon);
title('Top-hat Ricostruito');
xlabel('D');

% Top-hat classico
tophat_classic = img - img_opened;
subplot(2, 4, 5); imshow(tophat_classic);
title('Top-hat Classico'); xlabel('E');

% Ricostruzione morfologica del top-hat ricostruito
img_reconstructed2 = imreconstruct(imerode(tophat_recon, ones(1, 11)), tophat_recon);
subplot(2, 4, 6); imshow(img_reconstructed2)
title('Ricostruzione Morfologica di D'); xlabel('F');

% Dilatazione dell'immagine ricostruita con un elemento strutturante orizzontale
img_dilated = imdilate(img_reconstructed2, ones(1, 21));
subplot(2, 4, 7); imshow(img_dilated);
title('Dilatazione di F'); xlabel('G');

% Risultato finale della ricostruzione morfologica
final_result = imreconstruct(img_dilated, img_reconstructed2);
subplot(2, 4, 8); imshow(final_result);
title('Risultato Finale (Marker F, Mask G)'); xlabel('H');