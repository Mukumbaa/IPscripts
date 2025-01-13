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
end

f = imread('gel-image.tif');
subplot(1,2,1); imshow(f);
xlabel('Original image');
impixelinfo();

% watershed con marker
g = watershed_marker(f);
subplot(1,2,2), imshow(g);
xlabel('Watershed with marker segmentation');